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
    test_loader = make_loader("test", args.data_dir, args.batch_size)
    _, label = iter(test_loader).next()
    num_labels = label.size(1)
    model = BertForPostClassification(args.model_name, num_labels, 0).to(device)
    model.load_state_dict(torch.load(os.path.join("checkpoints", args.weight_path)))
    model.eval()
    accuracy = get_accuracy(model, test_loader)
    hamming_accuracy = get_hamming_accuracy(model, test_loader)
    print(f"Accuracy: {accuracy}, Hamming Accuracy: {hamming_accuracy}")


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
        for i in range(labels.size(0)):
            set_true = set(torch.where(labels[i])[0].tolist())
            set_pred = set(torch.where(outputs[i])[0].tolist())
            denominator = len(set_true.union(set_pred))
            if denominator == 0:
                scores.append(1)
            else:
                numerator = len(set_true.intersection(set_pred))
                scores.append(numerator / denominator)
    return sum(scores) / len(scores)


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
