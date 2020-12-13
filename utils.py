import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_checkpoint(model, model_name, logger):
    logger.write(os.path.join("logs", f"{model_name}.txt"))
    torch.save(model.state_dict(), os.path.join("checkpoints", f"{model_name}.pt"))


def generator(model, tokenizer, data_loader, device):
    for (inputs, labels) in tqdm(data_loader):
        labels = labels.to(device)
        tokens = tokenizer(
            list(inputs), truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        outputs = model(**tokens)
        yield labels, outputs


class EarlyStopMonitor:
    def __init__(self, patience, mode="min"):
        assert mode in {"min", "max"}, "`mode` must be one of 'min' or 'max'"
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.stop = False

    def __call__(self, score):
        if self.mode == "max":
            score = -score
        if self.best_score is None:
            self.best_score = score
        elif self.best_score > score:
            self.counter = 0
            self.best_score = score
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.stop = True

    def get_best_score(self):
        return abs(self.best_score)


class Logger:
    def __init__(self, num_epochs, log_interval):
        self.log_interval = log_interval
        self.num_epochs = num_epochs
        self.log = []

    def __call__(self, epoch, train_loss, val_loss):
        self.log.append(
            f"Epoch [{epoch}/{self.num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f} "
        )

    def write(self, out_path):
        with open(out_path, "w+") as f:
            f.write("\n".join(self.log))

