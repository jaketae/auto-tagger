import argparse
import os

import torch
from transformers import AutoTokenizer

from dataset import make_loader
from utils import generator, load_model, set_seed


def main(args):
    set_seed()
    tags, test_loader = make_loader(
        "test", args.data_dir, args.batch_size, return_tags=True
    )
    model = load_model(args.model_name, tags, args.save_title)
    accuracy = get_accuracy(model, test_loader)
    hamming_accuracy = get_hamming_accuracy(model, test_loader)
    print(f"Accuracy: {accuracy:.4f}, Hamming Accuracy: {hamming_accuracy:.4f}")


@torch.no_grad()
def get_accuracy(model, test_loader):
    num_samples = 0
    num_correct = 0
    for (labels, outputs) in generator(model, test_loader):
        outputs = outputs > 0
        num_samples += torch.numel(labels)
        num_correct += (labels == outputs).sum().item()
    return num_correct / num_samples


@torch.no_grad()
def get_hamming_accuracy(model, test_loader):
    scores = []
    for (labels, outputs) in generator(model, test_loader):
        score = _hamming_accuracy(labels, outputs)
        scores.append(score)
    return sum(scores) / len(scores)


def _hamming_accuracy(labels, outputs):
    for i in range(labels.size(0)):
        set_true = set(torch.where(labels[i])[0].tolist())
        set_pred = set(torch.where(outputs[i])[0].tolist())
        denominator = len(set_true.union(set_pred))
        if denominator == 0:
            return 1
        numerator = len(set_true.intersection(set_pred))
        return numerator / denominator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        choices=["roberta-base", "distilroberta-base", "allenai/longformer-base-4096",],
    )
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--save_title", type=str, help="title of saved file")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
