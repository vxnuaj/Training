'''
NOTE best to overfit on small segment of data prior to real training
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from model import Seq2Seq
from dataloader import SequenceDataset

import random
import json
import wandb
import os

from tqdm import tqdm 

seed = 0
torch.manual_seed(seed)

assert os.path.exists('../checkpoints/adam/checkpoint2/checkpoint2/checkpoint2'), FileNotFoundError('path does not exist')

assert torch.cuda.is_available(), RuntimeError('no cuda!')
device = ('cuda')

# TODO adjust up to 90% vram.
batch_size = 64
test_batch_size = 64 
#

shuffle = False

sample_idx = random.randint(0, 1000) # idx of sequence seed grabbed from X_test which will be used to generate sequence.

# model params
encoder_params = (256, 256) # second idx is size of the encoded state
decoder_params = (256, 256)
in_vocab_size = 37836 # en vocab size
out_vocab_size = 48012 # fr vocab size
encoder_embedding_dim = 256
decoder_embedding_dim = 256
batch_first = True
eval_epoch_freq = 1 # frequency of validation testing

# learnign rate schedule params
epochs = 8
learning_rate = .0001 
lr_iter_milestones = [5.5, 6., 7.5, 8.] # every half epoch, we adjust lr rate.
gamma = .5

# load metadata, data and dataloader

with open('../data/train/fr_idx_to_word.json', 'r') as f:
    fr_idx_to_word = json.load(f)
    
with open('../data/train/en_idx_to_word.json', 'r') as f:
    en_idx_to_word = json.load(f)    

with open('../data/train/fr_word_to_idx.json', 'r') as f:
    fr_word_to_idx = json.load(f)
    
with open('../data/train/en_word_to_idx.json', 'r') as f:
    en_word_to_idx = json.load(f)    


def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

train_fr = load_data('../data/train/fr_seqs_raw.json')
train_en = load_data('../data/train/en_seqs_raw.json')

val_fr = load_data('../data/validation/fr_seqs_raw.json')
val_en = load_data('../data/validation/en_seqs_raw.json')

def collate_fn(batch, fr_word_to_idx, en_word_to_idx):
    fr_seqs, en_seqs = zip(*batch)
    
    fr_seqs = pad_sequence(fr_seqs, batch_first=True, padding_value=fr_word_to_idx['<pad>'])
    en_seqs = pad_sequence(en_seqs, batch_first=True, padding_value=en_word_to_idx['<pad>'])
    
    return fr_seqs, en_seqs

train_dataset = SequenceDataset(train_fr, train_en, fr_word_to_idx, en_word_to_idx)
val_dataset = SequenceDataset(train_fr[0:13000], train_en[0:1300], fr_word_to_idx, en_word_to_idx)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,  
    collate_fn=lambda batch: collate_fn(batch, fr_word_to_idx, en_word_to_idx),
    shuffle=True
)

val_dataloader = DataLoader(
    train_dataset,
    batch_size=test_batch_size,
    collate_fn=lambda batch: collate_fn(batch, fr_word_to_idx, en_word_to_idx),
    shuffle=False
)

# init model training params

fr_bos_token = fr_word_to_idx['<bos>']
fr_eos_token = fr_word_to_idx['<eos>']
fr_pad_token = fr_word_to_idx['<pad>']

en_bos_token = en_word_to_idx['<bos>']
en_eos_token = en_word_to_idx['<eos>']
en_pad_token = en_word_to_idx['<pad>']

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

checkpoint = torch.load('../checkpoints/adam/checkpoint2/checkpoint2/epoch_2.pth', weights_only= True)
model.load_state_dict(checkpoint)

criterion = nn.CrossEntropyLoss()

opt = optim.Adam(
    model.parameters(),
    lr = learning_rate
)

scheduler = MultiStepLR(
    opt, 
    milestones = lr_iter_milestones,
    gamma = gamma
    )

# TRAINING LOOP

run = wandb.init(
    project = 'MT-Encoder-Decoder',
    config = {
        'learning_rate': learning_rate,
        'epochs': epochs
        },
    name = 'adamv1'
)

total_steps = 0

model.train()

for epoch in (pbar:= tqdm(range(epochs), total = epochs, desc = "Epoch Progress")):
    
    for step, (target_seq, in_seq) in tqdm(enumerate(train_dataloader), total = len(train_dataloader), desc = 'Batch Progress'):
      
        if in_seq.size(0) != batch_size:
            continue
       
        total_steps += 1
        
        in_seq = in_seq.to(device)
        target_seq = target_seq.to(device) 
        
        logits = model(in_seq, target_seq)
        loss = criterion(logits.view(-1, out_vocab_size), target_seq.view(-1))
        pplx = torch.exp(loss)
        
        loss.backward()
        opt.step()
        opt.zero_grad() 
       
        pbar.set_description(f"Loss: {loss} | PPLX: {pplx}") 
        
        wandb.log({'loss': loss, 'pplx':pplx})
        
        if (epoch + step / len(train_dataloader)) in lr_iter_milestones:
            scheduler.step()
    
    torch.save(model.state_dict(), f = f'../checkpoints/adam/checkpoint2/checkpoint2/checkpoint2/epoch_{epoch}.pth')
    
    '''if epoch % eval_epoch_freq == 0:
       
        model.eval() 
        with torch.no_grad():
            
            total_loss = 0
            total_pplx = 0
            total_steps_test = 0  
            
            for step_test, (test_target_seq, test_in_seq) in tqdm(enumerate(val_dataloader), total = len(val_dataloader), desc = 'Evaluating'):
               
                if test_in_seq.size(0) != test_batch_size:
                    continue
                
                total_steps_test += 1 
                 
                test_in_seq = test_in_seq.to(device)
                test_target_seq = test_target_seq.to(device)
                
                logits = model.test(test_in_seq, test_target_seq)
                loss = criterion(logits.view(-1, out_vocab_size), test_target_seq.view(-1))
                pplx = torch.exp(loss)
                
                total_loss += loss
                total_pplx += pplx
           
            torch.save(model.state_dict(), f = f'../checkpoints/epoch_{epoch}.pth')
               
            avg_test_loss = total_loss / total_steps_test
            avg_test_pplx = total_pplx / total_steps_test
            
            wandb.log({'testloss': avg_test_loss, 'testpplx': avg_test_pplx})
            
            sample_str = train_en[sample_idx]
           
            sample_tok = [en_word_to_idx.get(word, en_word_to_idx['<pad>']) for word in sample_str.split()]
            pad_seq = [en_idx_to_word.get(idx, en_idx_to_word['1722']) for idx in sample_str.split()]
            sample_tensor = torch.tensor(sample_tok).unsqueeze(0).to(device)
     
#            print(sample_tok)  
#            print(pad_seq)
      
            sample_target_str = train_fr[sample_idx]
            
            
            sample_target_tok = [fr_word_to_idx.get(word, fr_word_to_idx['<pad>']) for word in sample_target_str.split()]
            sample_target_tensor = torch.tensor(sample_tok).unsqueeze(0).to(device)
         
            with torch.no_grad():
                logits = model.generate(sample_tensor, sample_target_tensor, fr_bos_token)
                pred = logits.argmax(dim = 2).view(-1) # TODO right?
               
                
            pred_str_list = [fr_idx_to_word[str(int(tok))] for tok in pred]
            pred_str = (' ').join(pred_str_list)'''
           
         
        