import pandas as pd
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
        x = self.tokenizer.encode(row["message"], self.max_len)
        y = self.tokenizer.encode(row["reply_message"], self.max_len)
        return x, y