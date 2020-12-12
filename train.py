import argparse
import os

import torch
from tqdm.auto import tqdm
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from dataset import make_loaders
from model import get_model
from utils import EarlyStopMonitor, Logger, save_checkpoint, set_seed


def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BertForPostClassification(
        args.model_name, args.num_labels, args.dropout, args.freeze_bert
    ).to(device)
    if args.weight_path:
        model.load_state_dict(torch.load(args.weight_path))
    train_loader = make_loaders(os.path.join("data", "train.csv"), args.batch_size)
    val_loader = make_loaders(os.path.join("data", "val.csv"), args.batch_size)
    criterion = torch.nn.BCEWithLogits()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_epochs * len(train_loader),
    )
    monitor = EarlyStopMonitor(args.patience)
    logger = Logger(args.num_epochs, args.log_interval)
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = run_epoch(
            train_loader, tokenizer, model, device, criterion, optimizer, scheduler
        )
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(val_loader, tokenizer, model, device, criterion)
        logger(epoch, train_loss, val_loss)
        monitor(val_loss)
        if monitor.stop:
            save_checkpoint(model, args.model_name, logger)
            break
    if not monitor.stop:
        save_checkpoint(model, args.model_name, logger)


def run_epoch(
    data_loader, tokenizer, model, device, criterion, optimizer=None, scheduler=None
):
    if optimizer is None:
        assert (
            scheduler is None
        ), "If `scheduler` is provided, you must also specify an `optimizer`"
    total_loss = 0
    for (inputs, labels) in tqdm(data_loader):
        labels = labels.to(device)
        tokens = tokenizer(
            inputs, truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        outputs = model(**tokens)
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
        default="roberta",
        choices=["roberta", "distilbert", "longformer"],
    )
    parser.add_argument("--weight_path", type=float, default=""),
    parser.add_argument("--num_labels", type=int, default=10),
    parser.add_argument("--dropout", type=float, default=0.5),
    parser.add_argument("--num_epoch", type=int, default=5),
    parser.add_argument("--log_interval", type=int, default=1),
    parser.add_argument("--batch_size", type=int, default=16),
    parser.add_argument("--freeze_bert", type=bool, default=True),
    parser.add_argument("--patience", type=int, default=2),
    args = parser.parse_args()
    main(args)
