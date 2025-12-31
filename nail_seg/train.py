from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nail_unet.dataset import NailSegDataset, build_train_transforms, build_val_transforms
from nail_unet.losses import BCEDiceLoss
from nail_unet.metrics import compute_batch_metrics
from nail_unet.models import MobileUNet
from nail_unet.utils import (
    RunPaths,
    count_parameters,
    create_run_dir,
    log_line,
    save_metrics,
    save_sample_grid,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nail segmentation model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--encoder_pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, optimizer, scheduler, scaler, path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", 0.0)


def save_checkpoint(
    run_paths: RunPaths,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    is_best: bool,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    torch.save(state, run_paths.last_ckpt)
    if is_best:
        torch.save(state, run_paths.best_ckpt)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    optimizer=None,
    grad_clip: float | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    mode = "train" if is_train else "val"
    model.train(is_train)

    total_loss = 0.0
    metric_sums = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, desc=mode, leave=False):
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
                memory_format=torch.channels_last,
            )
            masks = masks.to(device, non_blocking=device.type == "cuda")

            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                preds = model(images)
                loss = loss_fn(preds, masks)
                preds_prob = torch.sigmoid(preds)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            batch_metrics = compute_batch_metrics(preds_prob, masks)
            total_loss += loss.item()
            for key in metric_sums:
                metric_sums[key] += batch_metrics[key]
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {key: metric_sums[key] / max(num_batches, 1) for key in metric_sums}
    avg_metrics["loss"] = avg_loss
    return avg_metrics


def resolve_device(device_arg: str) -> torch.device:
    if not device_arg or device_arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU was detected. Use --device cpu instead.")
    return device


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.deterministic)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.set_float32_matmul_precision("high")

    data_dir = Path(args.data_dir)
    train_images = data_dir / "images" / "train"
    train_masks = data_dir / "masks" / "train"
    val_images = data_dir / "images" / "val"
    val_masks = data_dir / "masks" / "val"

    run_paths = create_run_dir()
    log_line(run_paths.log_file, f"Device: {device}")
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else 0
        log_line(run_paths.log_file, f"CUDA device: {torch.cuda.get_device_name(cuda_index)}")

    train_dataset = NailSegDataset(
        train_images, train_masks, build_train_transforms(args.img_size)
    )
    has_val = val_images.exists() and val_masks.exists()
    val_dataset = (
        NailSegDataset(val_images, val_masks, build_val_transforms(args.img_size))
        if has_val
        else None
    )

    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs = {"persistent_workers": True, "prefetch_factor": 2}

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        **worker_kwargs,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            **worker_kwargs,
        )
        if val_dataset
        else None
    )

    model = MobileUNet(
        encoder_pretrained=args.encoder_pretrained,
        apply_sigmoid=False,
    ).to(device, memory_format=torch.channels_last)
    log_line(run_paths.log_file, f"Parameters: {count_parameters(model)}")

    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 50
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_epoch, best_metric = load_checkpoint(
            model, optimizer, scheduler, scaler, resume_path
        )
        log_line(run_paths.log_file, f"Resumed from epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            scaler if device.type == "cuda" else None,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        val_metrics = None
        if val_loader and (epoch + 1) % args.val_every == 0:
            val_metrics = run_epoch(
                model, val_loader, loss_fn, device, scaler if device.type == "cuda" else None
            )

        if scheduler:
            scheduler.step()

        epoch_data = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_data)
        save_metrics(run_paths.metrics_file, history)

        log_line(
            run_paths.log_file,
            json.dumps(epoch_data),
        )

        if args.save_samples:
            images, masks = next(iter(train_loader))
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.no_grad():
                preds = model(images)
                preds_prob = torch.sigmoid(preds)
            save_sample_grid(
                images,
                masks,
                preds_prob,
                run_paths.samples / f"epoch_{epoch+1:03d}.png",
            )

        metric_source = val_metrics if val_metrics else train_metrics
        current_metric = metric_source["iou"]
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric

        save_checkpoint(
            run_paths,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
            best_metric,
            is_best,
        )

    log_line(run_paths.log_file, f"Best IoU: {best_metric:.4f}")


if __name__ == "__main__":
    main()
