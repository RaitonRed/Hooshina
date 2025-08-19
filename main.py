from src.dataset import Dataset
from src.tokenizer import Tokenizer
from src.model import Model
from src.train import train
from src.chat import chat
import pandas as pd

if __name__ == "__main__":
    # آماده‌سازی دیتاست
    df = pd.read_csv("data/proccessed/telegram.csv")
    tokenizer = Tokenizer()
    tokenizer.build_vocab(df["user"].tolist() + df["bot"].tolist())

    dataset = Dataset("data/proccessed/telegram.csv", tokenizer)
    model = Model(vocab_size=len(tokenizer.word2id))

    # آموزش
    train(model, dataset, epochs=3)

    # ورود به چت
    chat(model, tokenizer)
