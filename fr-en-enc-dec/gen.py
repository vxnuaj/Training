import torch
import json
from model import Seq2Seq

device = ('cuda')
seed = 0

checkpoint_path = '../checkpoints/adam/checkpoint2/checkpoint2/checkpoint2/epoch_5.pth' # TODO

encoder_params = (256, 256) # second idx is size of the encoded state
decoder_params = (256, 256)
in_vocab_size = 37836 # en vocab size
out_vocab_size = 48012 # fr vocab size
encoder_embedding_dim = 256
decoder_embedding_dim = 256
batch_first = True
eval_epoch_freq = 1 # frequency of validation testing
batch_size = 1

sample_seed = 101

with open('../data/train/en_seqs_raw.json', 'r') as f:
    en_seqs_raw = json.load(f)

with open('../data/train/fr_seqs_raw.json', 'r') as f:
    fr_seqs_raw = json.load(f)

with open('../data/train/en_word_to_idx.json', 'r') as f:
    en_word_to_idx = json.load(f)

with open('../data/train/fr_word_to_idx.json', 'r') as f:
    fr_word_to_idx = json.load(f)

with open('../data/train/fr_idx_to_word.json', 'r') as f:
    fr_idx_to_word = json.load(f)

fr_bos_token = fr_word_to_idx['<bos>']
fr_eos_token = fr_word_to_idx['<eos>']
fr_pad_token = fr_word_to_idx['<pad>']

en_bos_token = en_word_to_idx['<bos>']
en_eos_token = en_word_to_idx['<eos>']
en_pad_token = en_word_to_idx['<pad>']

checkpoint = torch.load(checkpoint_path, weights_only = True)

model = Seq2Seq(
    encoder_params = encoder_params,
    decoder_params = decoder_params,
    in_vocab_size = in_vocab_size,
    out_vocab_size = out_vocab_size,
    encoder_embedding_dim = encoder_embedding_dim,
    decoder_embedding_dim = decoder_embedding_dim,
    batch_size = batch_size,
    bos_token = fr_bos_token,
    fr_eos_token = fr_eos_token,
    fr_pad_token = fr_pad_token,
    en_pad_token = en_pad_token,
    batch_first = batch_first,
    seed = seed
).to(device)

model.load_state_dict(checkpoint)

sample = en_seqs_raw[sample_seed]
sample_target = fr_seqs_raw[sample_seed]
tok_sample_tensor = torch.tensor([en_word_to_idx[word] for word in sample.split()]).unsqueeze(0).to(device)
sample_target_tensor = torch.tensor([fr_word_to_idx[word] for word in sample_target.split()]).unsqueeze(0).to(device)

pred = model.generate(tok_sample_tensor, sample_target_tensor, fr_bos_token).argmax(dim = 2).squeeze(0)

out_str_list = [fr_idx_to_word[str(int(tok))] for tok in pred]
pred_str = (' ').join(out_str_list)

print(f"INPUT: {sample}")
print(f"TARGET: {sample_target}")
print(f"OUTPUT: {pred_str}")