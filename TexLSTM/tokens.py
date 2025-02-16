import os
import json
import pickle as pkl
from collections import Counter

latex_files = [f"../latex/{f}" for f in os.listdir('../latex/') if f.endswith('.tex')]
raw_data = [ ]

for file in latex_files:
    
    with open(file, 'r') as f:
        content = f.read()
        raw_data.append(content)

# each elemenent in raw_data is the string representation for the .tex files. 119 total.

def tokenize(content):
    return list(content)

tokenized_files = [tokenize(text) for text in raw_data]

tokens = [tok for sublist in tokenized_files for tok in sublist] 
vocab = Counter(tokens)

# generate bidirectional dicts to get char's based off of index and idnex based off of char's

char_to_index = {char: idx for idx, (char, _) in enumerate(vocab.items())}
index_to_char = {idx:char for char,idx in char_to_index.items()}

with open('../pre_data/char_to_idx.json', 'w') as f1, open('../pre_data/idx_to_char.json', 'w') as f2:
    json.dump(char_to_index, f1)
    json.dump(index_to_char, f2)

with open('../pre_data/tokenized_files.pkl', 'wb') as f:
    pkl.dump(tokenized_files, f)