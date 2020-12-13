import argparse

import torch
from tqdm.auto import tqdm

from dataset import make_loader


def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = make_loader("test", args.batch_size)
    _, label = iter(train_loader).next()
    num_labels = label.size(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BertForPostClassification(
        args.model_name, num_labels, args.dropout, args.freeze_bert
    ).to(device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()
    print(get_accuracy(model, tokenizer, test_loader, device))


@torch.no_grad()
def get_accuracy(model, tokenizer, test_loader, device):
    num_samples = 0
    num_correct = 0
    for (inputs, labels) in tqdm(test_loader):
        labels = labels.to(device)
        tokens = tokenizer(
            list(inputs), truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        outputs = model(**tokens)
        num_samples += labels.size(0)
        num_correct += (labels == outputs).sum().item()
    return num_correct / num_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        choices=[
            "roberta-base",
            "distilbert-base-uncased",
            "allenai/longformer-base-4096",
        ],
    )
    parser.add_argument("--weight_path", type="str", help="path to model weigts")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
