import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SequenceDataset(Dataset):
    def __init__(self, fr_sequences, en_sequences, fr_word_to_idx, en_word_to_idx):
       
        self.fr_sequences = fr_sequences
        self.en_sequences = en_sequences
        self.fr_word_to_idx = fr_word_to_idx
        self.en_word_to_idx = en_word_to_idx
        
    def __len__(self):
        return len(self.fr_sequences)
    
    def __getitem__(self, idx):
        fr_sentence = self.fr_sequences[idx]
        en_sentence = self.en_sequences[idx]
       
        # Tokenize and convert words to indices using the word_to_idx
        fr_indices = [self.fr_word_to_idx.get(word, self.fr_word_to_idx['<pad>']) for word in fr_sentence.split()]
        en_indices = [self.en_word_to_idx.get(word, self.en_word_to_idx['<pad>']) for word in en_sentence.split()]
        
        return torch.tensor(fr_indices), torch.tensor(en_indices)