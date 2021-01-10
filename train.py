import argparse
import os
import warnings

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import make_loader
from model import BertForPostClassification
from utils import EarlyStopMonitor, Logger, generator, save_model, set_seed


def main(args):
    set_seed()
    loader_config = {
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "min_len": args.min_len,
    }
    train_loader = make_loader("train", **loader_config)
    tags, val_loader = make_loader("val", return_tags=True, **loader_config)
    if args.load_title:
        model = load_model(args.model_name, tags, args.load_title)
        if model.max_len != args.max_len:
            warnings.warn("`max_len` of model and data loader do not match")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForPostClassification(
            args.model_name,
            tags,
            args.max_len,
            args.min_len,
            freeze_bert=args.freeze_bert,
        ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": 3e-5, "eps": 1e-8},
            {"params": model.classifier.parameters()},
        ]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_epochs * len(train_loader),
    )
    monitor = EarlyStopMonitor(args.patience)
    logger = Logger(args.num_epochs, args.log_interval)
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, scheduler)
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion)
        logger(epoch, train_loss, val_loss)
        monitor(val_loss)
        if monitor.stop:
            break
    save_model(model, f"{args.save_title}", logger)


def run_epoch(model, data_loader, criterion, optimizer=None, scheduler=None):
    if optimizer is None:
        assert (
            scheduler is None
        ), "If `scheduler` is provided, you must also specify an `optimizer`"
    total_loss = 0
    for (labels, outputs) in generator(model, data_loader):
        loss = criterion(outputs, labels)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilroberta-base",
        choices=[
            "bert-base",
            "distilbert-base",
            "roberta-base",
            "distilroberta-base",
            "allenai/longformer-base-4096",
        ],
    )
    parser.add_argument("--save_title", type=str)
    parser.add_argument("--load_title", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--max_len", type=int, default=256, help="maximum length of each text"
    )
    parser.add_argument(
        "--min_len", type=int, default=128, help="minimum length of each text"
    )
    parser.add_argument("--freeze_bert", dest="freeze_bert", action="store_true")
    parser.add_argument("--unfreeze_bert", dest="freeze_bert", action="store_false")
    parser.set_defaults(freeze_bert=True)
    args = parser.parse_args()
    main(args)
