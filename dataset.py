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


def make_loader(file_path, batch_size):
    dataset = BlogDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
