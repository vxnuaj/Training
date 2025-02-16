import torch
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def _spacepunc(text):
    """Preprocess text for punctuation spacing and special tokens."""
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = []

    for i, char in enumerate(text.lower()):
        if i > 0 and no_space(char, text[i - 1]):
            out.append(' ' + char)
        else:
            out.append(char)

    return '<bos> ' + ''.join(out) + ' <eos>'

def preprocess_batch(data_batch):
    """Preprocess a batch of text data."""
    fr_list = [item['translation']['fr'] for item in data_batch]
    en_list = [item['translation']['en'] for item in data_batch]
    
    fr_list = [_spacepunc(fr) for fr in fr_list]
    en_list = [_spacepunc(en) for en in en_list]
    
    return fr_list, en_list

def build_vocabulary(all_texts, min_freq=2):
    """Build vocabulary from all texts with minimum frequency threshold."""
    word_freq = {}
    for text in all_texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {'<pad>', '<unk>', '<bos>', '<eos>'} | {word for word, freq in word_freq.items() if freq >= min_freq}
    
    idx_to_word = {idx: word for idx, word in enumerate(sorted(vocab))}
    word_to_idx = {word: idx for idx, word in idx_to_word.items()}
    
    return idx_to_word, word_to_idx

def process_split(data_split, batch_size=5000):
    """Process a single data split using existing vocabularies."""
    fr_list, en_list = [], []
    
    data_list = [{'translation': {'fr': item['translation']['fr'], 
                                'en': item['translation']['en']}} 
                for item in data_split]
    
    with ProcessPoolExecutor() as executor:
        data_chunks = [data_list[i:i + batch_size] 
                      for i in range(0, len(data_list), batch_size)]
        
        for result in tqdm(executor.map(preprocess_batch, data_chunks), 
                          total=len(data_chunks)):
            fr_list.extend(result[0])
            en_list.extend(result[1])
    
    return fr_list, en_list

def save_data(base_path, split_name, fr_seqs, en_seqs, fr_vocab=None, en_vocab=None):
    """Save processed data and vocabularies."""
    os.makedirs(os.path.join(base_path, split_name), exist_ok=True)
    
    with open(os.path.join(base_path, split_name, 'fr_seqs_raw.json'), 'w') as f:
        json.dump(fr_seqs, f)
    with open(os.path.join(base_path, split_name, 'en_seqs_raw.json'), 'w') as f:
        json.dump(en_seqs, f)
    
    if fr_vocab and en_vocab:
        fr_idx_to_word, fr_word_to_idx = fr_vocab
        en_idx_to_word, en_word_to_idx = en_vocab
        
        with open(os.path.join(base_path, split_name, 'fr_idx_to_word.json'), 'w') as f:
            json.dump(fr_idx_to_word, f)
        with open(os.path.join(base_path, split_name, 'fr_word_to_idx.json'), 'w') as f:
            json.dump(fr_word_to_idx, f)
        with open(os.path.join(base_path, split_name, 'en_idx_to_word.json'), 'w') as f:
            json.dump(en_idx_to_word, f)
        with open(os.path.join(base_path, split_name, 'en_word_to_idx.json'), 'w') as f:
            json.dump(en_word_to_idx, f)

def collate_fn(batch, fr_word_to_idx, en_word_to_idx):
    fr_texts, en_texts = zip(*batch)
    
    # convert txts to idxs
    fr_seqs = [torch.tensor([fr_word_to_idx.get(word, fr_word_to_idx['<unk>']) for word in fr.split()]) for fr in fr_texts]
    en_seqs = [torch.tensor([en_word_to_idx.get(word, en_word_to_idx['<unk>']) for word in en.split()]) for en in en_texts]
    
    # pad seqs to max seq len in batch.
    fr_seqs = pad_sequence(fr_seqs, batch_first=True, padding_value=fr_word_to_idx['<pad>'])
    en_seqs = pad_sequence(en_seqs, batch_first=True, padding_value=en_word_to_idx['<pad>'])
    
    return fr_seqs, en_seqs

def main():
    base_path = '../data'
    os.makedirs(base_path, exist_ok=True)
    
    print("Loading Dataset from Hugging Face")
    data = load_dataset('opus_books', 'en-fr')
    
    full_dataset = data['train'].train_test_split(test_size=0.2, seed=42)
    train_data = full_dataset['train']
    test_splits = full_dataset['test'].train_test_split(test_size=0.5, seed=42)
    validation_data = test_splits['train']
    holdout_data = test_splits['test']
    
    print("processing training data")
    train_fr, train_en = process_split(train_data)
    
    print("building vocab ")
    fr_vocab = build_vocabulary(train_fr)
    en_vocab = build_vocabulary(train_en)
    #fr_word_to_idx = fr_vocab[1]
    #en_word_to_idx = en_vocab[1]
    
    print("saving training data")
    save_data(base_path, 'train', train_fr, train_en, fr_vocab, en_vocab)
    
    print("preprocessing validation")
    val_fr, val_en = process_split(validation_data)
    save_data(base_path, 'validation', val_fr, val_en)
    
    print("processing holdout ")
    holdout_fr, holdout_en = process_split(holdout_data)
    save_data(base_path, 'holdout', holdout_fr, holdout_en)
    
    print("Finished!")

if __name__ == '__main__':
    main()
