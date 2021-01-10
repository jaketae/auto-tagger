import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from utils import chunkify


def set_body(row, chunk):
    row["body"] = chunk
    return row


class BlogDataset(Dataset):
    def __init__(self, file_path, max_len, min_len):
        df = pd.read_csv(file_path)
        df_dict = df.to_dict("records")
        new_df = [
            set_body(row.copy(), chunk)
            for row in df_dict
            for chunk in chunkify(row["body"], max_len, min_len)
        ]
        self.data = pd.DataFrame(new_df).set_index("title")

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row["body"], torch.tensor(row[2:]).to(float)

    def __len__(self):
        return len(self.data)


def make_loader(mode, batch_size, max_len, min_len, return_tags=False):
    assert mode in {
        "train",
        "val",
        "test",
    }, "`mode` must be one of 'train', 'val', or 'test'"
    dataset = BlogDataset(
        os.path.join("data", f"{mode}.csv"), max_len, min_len
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if not return_tags:
        return data_loader
    tags = list(dataset.data.columns[2:])
    return tags, data_loader

