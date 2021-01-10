import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_model(model, save_title, logger):
    logger.write(os.path.join("logs", f"{save_title}.txt"))
    torch.save(model.state_dict(), os.path.join("checkpoints", f"{save_title}.pt"))
    with open(os.path.join("checkpoints", f"{save_title}.json"), "w+") as f:
        json.dump(model.config, f)


def load_model(model_name, tags, save_title):
    from model import BertForPostClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join("checkpoints", f"{save_title}.json")) as f:
        config = json.load(f)
    model = BertForPostClassification(model_name, tags, **config).to(device)
    model.load_state_dict(torch.load(os.path.join("checkpoints", f"{save_title}.pt")))
    model.eval()
    return model


def generator(model, data_loader):
    for (inputs, labels) in tqdm(data_loader):
        outputs = model(inputs)
        labels = labels.to(model.device)
        yield labels, outputs


def word_counter(sentence):
    return len(sentence.split())


def get_tags():
    return list(pd.read_csv("data/val.csv").columns[2:])


def chunkify(body, max_len, min_len):
    chunk = ""
    chunks = []
    word_count = 0
    sentences = body.split(".")
    for sentence in sentences:
        if not sentence:
            continue
        sentence += "."
        count = word_counter(sentence)
        if word_count <= max_len and word_count + count > max_len:
            chunks.append(chunk.lstrip())
            chunk = sentence
            word_count = count
        else:
            chunk += sentence
            word_count += count
    if chunk and word_count >= min_len:
        chunks.append(chunk.lstrip())
    return chunks


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
        self.best = None

    def __call__(self, epoch, train_loss, val_loss):
        log_string = (
            f"Epoch [{epoch}/{self.num_epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f} "
        )
        if epoch % self.log_interval == 0:
            tqdm.write(log_string)
        self.log.append(log_string)
        if self.best is None or val_loss < self.best:
            self.best = val_loss

    def write(self, out_path):
        with open(out_path, "w+") as f:
            f.write("\n".join(self.log))

