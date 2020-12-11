import argparse

import torch
from tqdm.auto import tqdm
from transformers import AdamW, AutoTokenizer

from dataset import make_loaders
from model import get_model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BertForPostClassification(
        args.model_name, args.num_hidden, args.num_labels, args.dropout
    ).to(device)
    if args.freeze_bert:
        model.freeze_bert()
    train_loader, test_loader = make_loaders(
        args.file_path, args.batch_size, args.test_size
    )
    criterion = torch.nn.BCEWithLogits()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_history = []
    for epoch in args.num_epochs:
        loss = run_epoch(train_loader, tokenizer, model, criterion, optimizer, device)
        loss_history.append(loss)
    torch.save(model.state_dict(), os.path.join("save", model_name))


def run_epoch(data_loader, tokenizer, model, device, criterion, optimizer=None):
    total_loss = 0
    for (inputs, labels) in tqdm(data_loader):
        labels = labels.to(device)
        tokens = tokenizer(
            inputs, truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        outputs = model(**tokens)
        loss = criterion(outputs["logits"], labels)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta",
        choices=["roberta", "distilbert", "longformer"],
    )
    parser.add_argument("--num_hidden", type=int, default=784),
    parser.add_argument("--num_labels", type=int, default=10),
    parser.add_argument("--dropout", type=float, default=0.5),
    parser.add_argument("--num_epoch", type=int, default=5),
    parser.add_argument("--batch_size", type=int, default=16),
    parser.add_argument("--test_size", type=float, default=0.1),
    parser.add_argument("--file_path", type=str, default="data.csv"),
    parser.add_argument("--freeze_bert", type=bool, default=True),
    parser.add_argument("--verbose", type=bool, default=False),
    args = parser.parse_args()
    main(args)
