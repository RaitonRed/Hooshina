import pandas as pd
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=64):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(self.tokenizer.encode(row["user"], self.max_len), dtype=torch.long)
        y = torch.tensor(self.tokenizer.encode(row["bot"], self.max_len), dtype=torch.long)
        return x, y