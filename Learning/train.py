import os
import random
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils.data_factory import build_dataloaders_single_feature
from .utils.early_stopping import EarlyStopping
from .model_config import get_default_config, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - true)).item()


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()


def mape(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-5) -> float:
    denominator = torch.clamp(torch.abs(true), min=eps)
    return torch.mean(torch.abs((pred - true) / denominator)).item()


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)  # (B, N, T_out)

        optimizer.zero_grad()
        pred = model(x)  # (B, N, T_out)

        loss = criterion(pred, y)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_mape = 0.0
    n_batches = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = criterion(pred, y)
        total_loss += loss.item()
        total_mae += mae(pred, y)
        total_rmse += rmse(pred, y)
        total_mape += mape(pred, y)
        n_batches += 1

    denom = max(n_batches, 1)
    return (
        total_loss / denom,
        total_mae / denom,
        total_rmse / denom,
        total_mape / denom,
    )


def run_training(
    cfg: Dict[str, Any],
    log_callback: Optional[Callable[[str], None]] = None,
    stop_callback: Optional[Callable[[], bool]] = None,
) -> Dict[str, str]:
    def log(msg: str) -> None:
        print(msg)
        if log_callback is not None:
            log_callback(msg)

    device = torch.device(cfg["device"])
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    train_loader, val_loader, test_loader, stats, shapes = build_dataloaders_single_feature(
        raw=data_cfg["data_path"],
        input_steps=data_cfg["input_steps"],
        pred_steps=data_cfg["pred_steps"],
        batch_size=data_cfg["batch_size"],
        shuffle=data_cfg["shuffle"],
        DEVICE=device,
        split_ratios=tuple(data_cfg["split_ratios"]),
    )

    num_nodes = shapes["N_nodes"]
    log(f"Loaded data from {data_cfg['data_path']}, nodes = {num_nodes}")

    model = build_model(cfg, num_nodes=num_nodes, device=device)
    log(str(model))

    train_cfg = cfg["train"]
    log_cfg = cfg["log"]

    if train_cfg["loss"].lower() == "mae":
        criterion = nn.L1Loss()
    elif train_cfg["loss"].lower() == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss: {train_cfg['loss']}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    save_root = log_cfg["save_dir"]
    model_name = cfg["model"]["name"]
    model_save_dir = os.path.join(save_root, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{log_cfg['experiment_name']}_{model_name}_{timestamp}"
    best_model_path = os.path.join(model_save_dir, f"{exp_name}_best.pth")
    last_model_path = os.path.join(model_save_dir, f"{exp_name}_last.pth")

    patience = train_cfg.get("patience", 10)
    use_early_stopping = patience is not None and patience > 0
    early_stopping: Optional[EarlyStopping]
    if use_early_stopping:
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            delta=0.0,
            path=best_model_path,
            mode="min",
        )
    else:
        early_stopping = None

    epochs = train_cfg["epochs"]
    grad_clip = train_cfg.get("grad_clip", 0.0)
    print_every = log_cfg.get("print_every", 50)

    for epoch in range(1, epochs + 1):
        if stop_callback is not None and stop_callback():
            log(f"Training stopped by user before epoch {epoch}.")
            break
        start_time = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip,
        )

        val_loss, val_mae, val_rmse, val_mape = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        elapsed = time.time() - start_time

        log(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAPE: {val_mape:.4f} | "
            f"Time: {elapsed:.2f}s"
        )

        if early_stopping is not None:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                log("Early stopping triggered.")
                break

    torch.save(model.state_dict(), last_model_path)

    if os.path.exists(best_model_path):
        log(f"Loading best model from {best_model_path} for test evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_mae, test_rmse, test_mape = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )
    log(
        f"[Test] Loss: {test_loss:.4f} | "
        f"MAE: {test_mae:.4f} | "
        f"RMSE: {test_rmse:.4f} | "
        f"MAPE: {test_mape:.4f}"
    )
    return {"best": best_model_path, "last": last_model_path}
