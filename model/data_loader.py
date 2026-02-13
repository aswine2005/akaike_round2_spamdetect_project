import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.preprocessing import clean_text
class SpamDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def encode(self, text):
        tokens = text.split()
        seq = [self.vocab.get(word, self.vocab["<UNK>"]) for word in tokens]
        if len(seq) < self.max_len:
            seq += [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return torch.tensor(seq)
    def __getitem__(self, idx):
        return self.encode(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.float)