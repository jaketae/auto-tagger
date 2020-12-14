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


def make_loader(mode, data_dir, batch_size):
    assert mode in {
        "train",
        "val",
        "test",
    }, "`mode` must be one of 'train', 'val', or 'test'"
    dataset = BlogDataset(os.path.join(data_dir, f"{mode}.csv"))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
