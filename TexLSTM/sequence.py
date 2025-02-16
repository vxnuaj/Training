import json
import pickle as pkl
import torch
from tqdm import tqdm
import numpy as np

# --- OPEN FILES
with open('../pre_data/char_to_idx.json', 'r') as f1, open('../pre_data/idx_to_char.json', 'r') as f2:
    char_to_index = json.load(f1)
    index_to_char = json.load(f2)

with open('../pre_data/tokenized_files.pkl', 'rb') as f:
    tokenized_files = pkl.load(f)

# --- PARAMS
seq_length = 150

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise ValueError("cuda not available mannn...")

print(f"Generating sequences using {device}")

def create_sequences(chars, seq_length):
    indices = torch.tensor([char_to_index[char] for char in chars], device=device)
    num_sequences = len(indices) - seq_length
    input_seqs = torch.stack([indices[i:i + seq_length] for i in range(num_sequences)]).to(device)
    target_seqs = torch.stack([indices[i + 1:i + 1 + seq_length] for i in range(num_sequences)]).to(device)
    return input_seqs, target_seqs

# --- create numerical sequences

with open('../pre_data/numeric_sequences.pkl', 'wb') as f:
    for chars in tqdm(tokenized_files, desc='> Processing files'):
        input_seqs, target_seqs = create_sequences(chars, seq_length) # NOTE [:1000] will be removed once I rent a100 -- 8gb ram is not enough to process entire dataset.
        for i in range(input_seqs.size(0)):
            numeric_seq = (input_seqs[i].tolist(), target_seqs[i].tolist())
            pkl.dump(numeric_seq, f)


# create training / test datasets and input / targets

print('> Creating, Splitting, and Saving Train / Test Set')

X, y = [], []
n = .8 # train split size, relative to entire dataset.

with open('../pre_data/numeric_sequences.pkl', 'rb') as f:
    try:
        while True:
            input_seq, target = pkl.load(f)
            X.append(input_seq)
            y.append(target)
    except EOFError:
        pass

X = torch.tensor(X, device=device)
y = torch.tensor(y, device=device)

split_idx = int(n * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]


torch.save(X_train, '../data/X_train.pt')
torch.save(y_train, '../data/y_train.pt')
torch.save(X_test, '../data/X_test.pt')
torch.save(y_test, '../data/y_test.pt')
