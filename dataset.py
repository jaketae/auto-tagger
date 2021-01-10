import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class BlogDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row["body"], torch.tensor(row[2:]).to(float)

    def __len__(self):
        return len(self.data)


def make_loader(mode, data_dir, batch_size, return_tags=False):
    assert mode in {
        "train",
        "val",
        "test",
    }, "`mode` must be one of 'train', 'val', or 'test'"
    dataset = BlogDataset(os.path.join("data", data_dir, f"{mode}.csv"))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if not return_tags:
        return data_loader
    tags = list(dataset.data.columns[2:])
    return tags, data_loader


def get_tags(data_dir):
    df = pd.read_csv(os.path.join("data", data_dir, "val.csv"))
    return df.columns.values.tolist()[2:]
