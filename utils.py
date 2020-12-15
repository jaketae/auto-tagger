import os
import random

import numpy as np
import torch
from tqdm.auto import tqdm


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_checkpoint(model, save_title, logger):
    logger.write(os.path.join("logs", f"{save_title}.txt"))
    torch.save(model.state_dict(), os.path.join("checkpoints", f"{save_title}.pt"))


def generator(model, data_loader):
    for (inputs, labels) in tqdm(data_loader):
        outputs = model(inputs)
        labels = labels.to(model.device)
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
        log_string = (
            f"Epoch [{epoch}/{self.num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f} "
        )
        if epoch % self.log_interval == 0:
            tqdm.write(log_string)
        self.log.append(log_string)

    def write(self, out_path):
        with open(out_path, "w+") as f:
            f.write("\n".join(self.log))

