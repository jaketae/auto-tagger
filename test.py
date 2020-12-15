import argparse
import os

import torch
from transformers import AutoTokenizer

from dataset import make_loader
from model import BertForPostClassification
from utils import generator, set_seed


def main(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join("data", args.data_dir)
    test_loader = make_loader("test", data_dir, args.batch_size)
    _, label = iter(test_loader).next()
    num_labels = label.size(1)
    model = BertForPostClassification(args.model_name, num_labels, 0).to(device)
    model.load_state_dict(torch.load(os.path.join("checkpoints", args.weight_path)))
    model.eval()
    print(get_accuracy(model, test_loader, device))


@torch.no_grad()
def get_accuracy(model, test_loader, device):
    num_samples = 0
    num_correct = 0
    for (labels, outputs) in generator(model, test_loader):
        outputs = outputs > 0
        num_samples += torch.numel(labels)
        num_correct += (labels == outputs).sum().item()
    return num_correct / num_samples


def get_hamming_accuracy(model, tokenizer, test_loader, device):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        choices=["roberta-base", "distilroberta-base", "allenai/longformer-base-4096",],
    )
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--weight_path", type=str, help="path to model weigts")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
