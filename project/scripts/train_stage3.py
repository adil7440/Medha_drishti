import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import random
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.dncnn import DnCNN
from models.swinir import SwinIRSmall
from models.swinir_large import SwinIRLarge
from models.restormer import Restormer
from models.mirnet_v2 import MIRNetv2
from models.nafnet import NAFNet
from scripts.stage3_dataset import get_dataloaders
from scripts.loss_functions import HybridLoss

import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = PROJECT_DIR / "configs" / "stage3_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def build_model(model_name: str, model_cfg: dict) -> nn.Module:
    """Factory function to build a model from config."""
    name = model_name.lower().replace("-", "_")

    MODEL_MAP = {
        "dncnn": (DnCNN, DnCNN.__init__.__code__.co_varnames),
        "swinir_small": (SwinIRSmall, SwinIRSmall.__init__.__code__.co_varnames),
        "swinir_large": (SwinIRLarge, SwinIRLarge.__init__.__code__.co_varnames),
        "restormer": (Restormer, Restormer.__init__.__code__.co_varnames),
        "mirnet_v2": (MIRNetv2, MIRNetv2.__init__.__code__.co_varnames),
        "nafnet": (NAFNet, NAFNet.__init__.__code__.co_varnames),
    }

    if name in MODEL_MAP:
        cls, valid_keys = MODEL_MAP[name]
        filtered = {k: v for k, v in model_cfg.items() if k in valid_keys}
        return cls(**filtered)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class WarmupCosineScheduler:
    """Cosine Annealing with Warm Restarts + Linear Warmup."""
    def __init__(self, optimizer, warmup_epochs, T_0, T_mult, eta_min):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.eta_min = eta_min
        self.T_0 = T_0
        self.T_mult = T_mult
        self.current_epoch = 0
        self.current_T = T_0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            elapsed = self.current_epoch - self.warmup_epochs
            t = elapsed % self.current_T
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
                 (1 + np.cos(np.pi * t / self.current_T))
            if elapsed > 0 and elapsed % self.current_T == 0:
                self.current_T = int(self.current_T * self.T_mult)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    config: dict,
    output_dir: Path,
    resume_checkpoint: str = None,
):
    train_cfg = config["training"]
    opt_cfg = config["optimizer"]
    sched_cfg = config["scheduler"]
    loss_cfg = config["loss"]
    max_epochs = train_cfg["max_epochs"]
    patience = train_cfg["early_stopping"]["patience"]
    min_delta = train_cfg["early_stopping"]["min_delta"]
    use_amp = train_cfg.get("mixed_precision", True)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    grad_clip = train_cfg.get("gradient_clip_norm", 1.0)

    print(f"\n{'=' * 70}")
    print(f" Training: {model_name} | Device: {device} | Max Epochs: {max_epochs}")
    print(f"{'=' * 70}")

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = round(param_count * 4 / (1024 ** 2), 2)
    print(f"Parameters: {param_count:,} | Size: {model_size_mb} MB")

    criterion = HybridLoss(
        weights=loss_cfg.get("weights"),
        eps=loss_cfg.get("charbonnier", {}).get("eps", 1e-6),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 2e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=sched_cfg.get("warmup_epochs", 5),
        T_0=sched_cfg.get("T_0", 50),
        T_mult=sched_cfg.get("T_mult", 2),
        eta_min=sched_cfg.get("eta_min", 1e-7),
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))

    start_epoch = 1
    best_val_psnr = -float("inf")
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    checkpoints_dir = output_dir / "checkpoints" / model_name.lower()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Resume training
    if resume_checkpoint and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        best_val_psnr = ckpt.get("val_psnr", -float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    start_train_t = time.time()

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start_t = time.time()
        model.train()
        train_loss = 0.0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for step, (inp, tgt, _) in enumerate(pbar):
            inp, tgt = inp.to(device), tgt.to(device)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    pred = model(inp)
                    loss = criterion(pred, tgt) / grad_accum
                scaler.scale(loss).backward()
            else:
                pred = model(inp)
                loss = criterion(pred, tgt) / grad_accum
                loss.backward()

            train_loss += loss.item() * inp.size(0) * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if grad_clip > 0:
                    if use_amp and device.type == "cuda":
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if use_amp and device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        train_loss /= len(train_loader.dataset)
        current_lr = scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_count = 0

        with torch.no_grad():
            for inp, tgt, _ in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        pred = model(inp)
                        loss = criterion(pred, tgt)
                else:
                    pred = model(inp)
                    loss = criterion(pred, tgt)

                val_loss += loss.item() * inp.size(0)
                for b in range(inp.size(0)):
                    val_psnr += calculate_psnr(pred[b:b+1], tgt[b:b+1])
                val_count += inp.size(0)

        val_loss /= val_count
        val_psnr /= val_count
        epoch_time = round(time.time() - epoch_start_t, 2)
        gpu_mem = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device.type == "cuda" else 0.0

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_psnr": round(val_psnr, 4),
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
            "gpu_mem_mb": gpu_mem,
        }
        history.append(epoch_stats)

        status = ""
        if val_psnr > best_val_psnr + min_delta:
            best_val_psnr = val_psnr
            best_val_loss = val_loss
            patience_counter = 0
            status = " * BEST *"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "model_name": model_name,
            }, checkpoints_dir / "best_checkpoint.pth")
        else:
            patience_counter += 1

        print(
            f"Epoch [{epoch:03d}/{max_epochs:03d}] | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"PSNR: {val_psnr:.4f} dB | LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s | GPU: {gpu_mem:.0f}MB | "
            f"Patience: {patience_counter}/{patience}{status}"
        )

        # Save periodic checkpoint
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "model_name": model_name,
            }, checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pth")

        # Early Stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    total_time = round(time.time() - start_train_t, 2)

    # Save final checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_psnr": val_psnr,
        "model_name": model_name,
    }, checkpoints_dir / "final_checkpoint.pth")

    # Save training history
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    history_data = {
        "model_name": model_name,
        "total_train_time_sec": total_time,
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "best_val_psnr": round(best_val_psnr, 4),
        "best_val_loss": round(best_val_loss, 6),
        "epochs_trained": len(history),
        "history": history,
    }
    with open(logs_dir / f"{model_name.lower()}_history.json", "w") as f:
        json.dump(history_data, f, indent=2)

    print(f"\n[Done] {model_name}: {total_time}s | Best PSNR: {best_val_psnr:.4f} dB | Epochs: {len(history)}")
    return best_val_psnr, total_time


def run_stage3_training(preprocessed_dir: str, config: dict = None):
    if config is None:
        config = load_config()

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    stage3_dir = PROJECT_DIR / config.get("output", {}).get("base_dir", "stage3")
    stage3_dir.mkdir(parents=True, exist_ok=True)

    bs = config.get("dataset", {}).get("batch_size", {})
    batch_size = bs.get("cuda", 4) if device.type == "cuda" else bs.get("cpu", 2)
    target_size = config.get("dataset", {}).get("target_size", 128)
    max_samples = config.get("dataset", {}).get("max_samples", None)
    aug_config = config.get("augmentation", {})

    train_loader, val_loader = get_dataloaders(
        preprocessed_dir, batch_size=batch_size, target_size=target_size,
        max_samples=max_samples, aug_config=aug_config,
    )

    results = {}
    models_cfg = config.get("models", {})

    for model_name, model_cfg in models_cfg.items():
        if not model_cfg.get("enabled", True):
            print(f"[Skip] {model_name} is disabled in config.")
            continue

        print(f"\n>>> Building {model_name}")
        try:
            model = build_model(model_name, model_cfg)
            psnr, train_time = train_single_model(
                model_name, model, train_loader, val_loader,
                device, config, stage3_dir,
            )
            results[model_name] = {"psnr": psnr, "train_time": train_time}
        except Exception as e:
            print(f"[Error] Training {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(" All Models Training Complete")
    for name, res in results.items():
        print(f"  {name}: PSNR={res['psnr']:.4f} dB | Time={res['train_time']:.1f}s")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    prep_dir = PROJECT_DIR / "stage2" / "preprocessed"
    run_stage3_training(str(prep_dir))
