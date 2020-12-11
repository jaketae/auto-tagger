import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class BlogDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row["body"], torch.tensor(row[2:]).to(float)

    def __len__(self):
        return len(self.data)


def make_loaders(file_path, test_size, batch_size):
    dataset = BlogDataset(file_path)
    train_idx, test_idx = train_test_split(
        range(len(dataset)), test_size=test_size, shuffle=True
    )
    train_dataset = Subset(dataset, indices=train_idx)
    test_dataset = Subset(dataset, indices=test_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

