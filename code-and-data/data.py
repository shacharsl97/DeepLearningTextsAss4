from __future__ import annotations
from typing import Iterator
import torch
from torch import nn
import torch.nn.functional as F
import random
import glob
from tqdm import tqdm
import pickle
import os

class CharTokenizer:
    def __init__(self):
        self.symbols = ["<PAD>"]
        self.tokens = set()
        self.vocab = list(self.symbols)
        self.stoi = {s:i for i, s in enumerate(self.vocab)}

    def pad_id(self): return self.stoi["<PAD>"]

    def get_id(self, tok: str): return self.stoi[tok]

    def vocab_size(self): return len(self.vocab)
        
    def train(self, sequences: list[str]) -> None:
        for seq in sequences:
            for symbol in self._tokenize_to_symbols(seq):
                self.tokens.add(symbol)

        self.vocab = list(self.symbols) + list(sorted(self.tokens))
        self.stoi = {s:i for i, s in enumerate(self.vocab)}


    def _tokenize_to_symbols(self, text: str) -> list[str]:
        return list(text)

    def tokenize(self, text: str) -> list[int]:
        seq: list[str] = self._tokenize_to_symbols(text)
        return [self.stoi[s] for s in seq]

    def detokenize(self, tokens: list[int], keep_symbols = True) -> str:
        strs: list[str] = [self.vocab[t] for t in tokens]
        if not keep_symbols:
            strs = [s for s in strs if len(s) == 1]
        return "".join(strs)

    def save(self, path: str) -> None:
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'symbols': self.symbols,
                'tokens': list(self.tokens),
                'vocab': self.vocab,
                'stoi': self.stoi
            }, f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> 'CharTokenizer':
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = CharTokenizer()
        tokenizer.symbols = data['symbols']
        tokenizer.tokens = set(data['tokens'])
        tokenizer.vocab = data['vocab']
        tokenizer.stoi = {s: int(i) for s, i in data['stoi'].items()}
        return tokenizer

class RandomOrderDataIterator:
    def __init__(self, data, desired_length):
        self.desired_length = desired_length
        self.data: list[list[int]] = [seq for seq in data if len(seq) > self.desired_length]

    def __iter__(self):
        if len(self.data) == 0: return
        while True:
            seq = random.choice(self.data)
            idx = random.randint(0, len(seq) - self.desired_length)
            yield seq[idx:idx + self.desired_length]


# This both creates the tokenizer and uses it to tokenize the data.
# In a real system you'd like to split it to two separate functions.
# Feel free to separate it to two functions also in this code.
def load_data(path: str):
    """
    Loads tokenized data and tokenizer from disk if available, otherwise processes and saves them.
    Returns: (tokenizer, data)
    """
    tokenized_path = os.path.join(path, 'tokenized_data.pkl')
    tokenizer_path = os.path.join(path, 'tokenizer.json')
    if os.path.exists(tokenized_path) and os.path.exists(tokenizer_path):
        print(f"Loading tokenized data from {tokenized_path} and tokenizer from {tokenizer_path}...")
        with open(tokenized_path, 'rb') as f:
            data = pickle.load(f)
        tokenizer = CharTokenizer.load(tokenizer_path)
        print("Tokenized data loaded.")
        return tokenizer, data
    else:
        print("Tokenized data not found. Training tokenizer and tokenizing data with tqdm progress bars...")
        tokenizer = CharTokenizer()
        files = list(glob.glob(f"{path}/*.txt"))
        for fname in tqdm(files, desc="Training tokenizer"):
            with open(fname) as fh:
                text = fh.read()
                tokenizer.train(text)
        data = []
        for fname in tqdm(files, desc="Tokenizing data"):
            with open(fname) as fh:
                text = fh.read()
                data.append(tokenizer.tokenize(text))
        with open(tokenized_path, 'wb') as f:
            pickle.dump(data, f)
        tokenizer.save(tokenizer_path)
        print(f"Tokenized data and tokenizer saved to {tokenized_path} and {tokenizer_path}.")
        return tokenizer, data

def batch_items(data_iter: Iterator[list[int]], batch_size: int = 2) -> Iterator[torch.LongTensor]:
    batch = []
    for seq in data_iter:
        idx = 0
        batch.append(seq)
        if len(batch) >= batch_size:
            yield torch.tensor(batch, dtype=torch.long)
            batch = []
    if len(batch) > 0:
        yield torch.tensor(batch, dtype=torch.long)

