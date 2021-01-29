import argparse

import torch
from transformers import AutoTokenizer, pipeline

from utils import get_all_tags, set_seed


def main(args):
    # https://github.com/joeddav/zero-shot-demo/blob/master/app.py
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tags = get_all_tags()
    test_loader = make_loader(
        "test", args.batch_size, args.max_len, args.min_len, return_tags=False,
    )
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline(
        "zero-shot-classification", model=model_name, tokenizer=tokenizer, device=device
    )
    for (sequence, _) in tqdm(test_loader):
        result = classifier(sequence, tags, multi_class=True)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/bart-large-mnli",
        choices=["facebook/bart-large-mnli", "roberta-large-mnli",],
    )
    args = parser.parse_args()
    main(args)
