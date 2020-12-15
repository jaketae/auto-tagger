import argparse
import os

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import make_loader
from model import BertForPostClassification
from utils import EarlyStopMonitor, Logger, generator, save_checkpoint, set_seed


def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join("data", args.data_dir)
    train_loader = make_loader("train", data_dir, args.batch_size)
    val_loader = make_loader("val", data_dir, args.batch_size)
    _, label = iter(train_loader).next()
    num_labels = label.size(1)
    model = BertForPostClassification(
        args.model_name, num_labels, args.dropout, args.freeze_bert
    ).to(device)
    if args.weight_path:
        model.load_state_dict(torch.load(os.path.join("checkpoints", args.weight_path)))
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
        train_loss = run_epoch(
            model, train_loader, device, criterion, optimizer, scheduler
        )
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, criterion)
        logger(epoch, train_loss, val_loss)
        monitor(val_loss)
        if monitor.stop:
            break
    save_checkpoint(model, f"{args.data_dir}_{args.model_name}", logger)


def run_epoch(model, data_loader, device, criterion, optimizer=None, scheduler=None):
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
        default="roberta-base",
        choices=["roberta-base", "distilroberta-base", "allenai/longformer-base-4096",],
    )
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--weight_path", type=str, default="")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--freeze_bert", dest="freeze_bert", action="store_true")
    parser.add_argument("--unfreeze_bert", dest="freeze_bert", action="store_false")
    parser.set_defaults(freeze_bert=True)
    args = parser.parse_args()
    main(args)
