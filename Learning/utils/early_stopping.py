import os
from typing import Optional

import numpy as np
import torch


class EarlyStopping:


    def __init__(
        self,
        patience: int = 10,
        verbose: bool = True,
        delta: float = 0.0,
        path: Optional[str] = None,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = np.inf if mode == "min" else -np.inf

    def __call__(self, value: float, model: torch.nn.Module) -> None:
        score = -value if self.mode == "min" else value

        if self.best_score is None:
            self.best_score = score
            self.best_value = value
            self._save_checkpoint(value, model)
        elif score < self.best_score + self.delta:
            # 没有显著提升
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_value = value
            self._save_checkpoint(value, model)
            self.counter = 0

    def _save_checkpoint(self, value: float, model: torch.nn.Module) -> None:
        if self.path is None:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if self.verbose:
            print(f"Validation metric improved to {value:.6f}. Saving model to {self.path}")
        torch.save(model.state_dict(), self.path)


