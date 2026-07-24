import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.dncnn import DnCNN
from models.swinir import SwinIRSmall
from scripts.stage3_dataset import get_dataloaders
from scripts.loss_functions import HybridLoss


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Computes batch average PSNR."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    max_epochs: int = 5,
    patience: int = 5,
    output_dir: Path = None
):
    print(f"\n" + "=" * 60)
    print(f" Starting Training: {model_name} on {device}")
    print("=" * 60)

    model = model.to(device)
    criterion = HybridLoss(l1_weight=0.8, ssim_weight=0.2).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    use_cuda = (device.type == "cuda")

    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda) if use_cuda else None

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    checkpoints_dir = output_dir / "checkpoints" / model_name.lower()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    start_train_t = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start_t = time.time()
        model.train()
        train_loss = 0.0

        for inp, tgt, _ in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()

            if use_cuda:
                with torch.amp.autocast('cuda'):
                    pred = model(inp)
                    loss = criterion(pred, tgt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(inp)
                loss = criterion(pred, tgt)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inp.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0

        with torch.no_grad():
            for inp, tgt, _ in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                if use_cuda:
                    with torch.amp.autocast('cuda'):
                        pred = model(inp)
                        loss = criterion(pred, tgt)
                else:
                    pred = model(inp)
                    loss = criterion(pred, tgt)

                val_loss += loss.item() * inp.size(0)
                val_psnr += calculate_psnr(pred, tgt) * inp.size(0)

        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        epoch_time = round(time.time() - epoch_start_t, 2)
        current_lr = scheduler.get_last_lr()[0]

        gpu_mem = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if use_cuda else 0.0

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_psnr": round(val_psnr, 2),
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
            "gpu_mem_mb": gpu_mem
        }
        history.append(epoch_stats)

        print(f"Epoch [{epoch:02d}/{max_epochs:02d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB | Time: {epoch_time}s")

        # Save Best Checkpoint & Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr
            }, checkpoints_dir / "best_checkpoint.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" Early stopping triggered at epoch {epoch} (Patience = {patience}).")
                break

    total_time = round(time.time() - start_train_t, 2)

    # Save History JSON
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / f"{model_name.lower()}_history.json", "w") as f:
        json.dump({"model_name": model_name, "total_train_time_sec": total_time, "history": history}, f, indent=2)

    return best_val_loss, total_time


def run_stage3_training(preprocessed_dir: str, max_epochs: int = 5):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    stage3_dir = PROJECT_DIR / "stage3"
    stage3_dir.mkdir(parents=True, exist_ok=True)

    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(preprocessed_dir, batch_size=4 if device.type == "cuda" else 2, target_size=128, max_samples=40)

    # 1. Train DnCNN (Baseline Model)
    dncnn = DnCNN(in_channels=1, out_channels=1, num_features=64, num_layers=17)
    train_single_model("DnCNN", dncnn, train_loader, val_loader, device, max_epochs=max_epochs, output_dir=stage3_dir)

    # 2. Train SwinIR Small (Primary Model)
    swinir = SwinIRSmall(in_channels=1, out_channels=1, embed_dim=48, num_heads=4, window_size=8)
    train_single_model("SwinIR", swinir, train_loader, val_loader, device, max_epochs=max_epochs, output_dir=stage3_dir)


if __name__ == "__main__":
    prep_dir = PROJECT_DIR / "stage2" / "preprocessed"
    run_stage3_training(str(prep_dir), max_epochs=5)
