import re

class Tokenizer:
    def __init__(self):
        self.word2id = {"<pad>": 0, "<unk>": 1}
        self.id2word = {0: "<pad>", 1: "<unk>"}

    def build_vocab(self, texts, min_freq=2):
        freq = {}
        for t in texts:
            for tok in self.tokenize(t):
                freq[tok] = freq.get(tok, 0) + 1

        for w, c in freq.items():
            if c >= min_freq and w not in self.word2id:
                idx = len(self.word2id)
                self.word2id[w] = idx
                self.id2word[idx] = w

    def tokenize(self, text):
        return re.findall(r"\w+|[^\s\w]", str(text))

    def encode(self, text, max_len):
        tokens = self.tokenize(text)
        ids = [self.word2id.get(t, 1) for t in tokens][:max_len]
        return ids + [0] * (max_len - len(ids))

    def decode(self, ids):
        return " ".join([self.id2word.get(i, "<unk>") for i in ids if i != 0])
