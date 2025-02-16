import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader

from TeXLSTM import TeXLSTM
from dataloader import SequenceDataset

import wandb
import time
import json
import warnings


warnings.filterwarnings('ignore', category = FutureWarning)

seed = 0
torch.manual_seed(seed)

print('\nINITIALIZING ----------------------------')

# init device

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise ValueError("cuda not available mannn...")

print(f"Training on {device}")

# load train / test data and dataloaders

print('LOADING DATA & VOCAB ----------------------------')

batch_size = 1
shuffle = False

X_train = torch.load('../data/X_train.pt').to(device)
y_train = torch.load('../data/y_train.pt').to(device)
X_test = torch.load('../data/X_test.pt').to(device)
y_test = torch.load('../data/y_test.pt').to(device)


dataset = SequenceDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

# init idx_to_char.

with open('../pre_data/idx_to_char.json', 'r') as f:
    idx_to_char = json.load(f)


# init model params and model

print('INSTANTIATING MODEL ----------------------------')

seq_len = 150
vocab_size = 97
embedding_dim = 100
n_lstm_cells = 2
n_units = (512, 512, vocab_size)

model = TeXLSTM(
    vocab_size = vocab_size,
    embedding_dim = embedding_dim,
    n_lstm_cells = n_lstm_cells,
    n_units = n_units,
    device = device,
    seed = seed,
).to(device)

# init training params

epochs = 20
lr = .005 

total_batches = X_train.size(0)
eval_iter = 40 # the iteration frequency at which we eval the model and generate sample latex.
sample_iter = 40 * 3 # the iteration frequency at which we sample latex from the model. must be a multiple of eval_iter
sample_train_pause = 3 # pause between model.train and model.eval()
max_length = 500 # must be less than input sequence size
temperature = 1 # will not be used if greedy = True
stochastic = True # stochastic samplign with temperature
greedy = False # greedy sampling
# sample_seq_len = 30 # length of input sequence during intermediary sampling -- NOTE not used if we define sample_seq below (also check training loop)
sample_seq = torch.tensor([0, 11, 8, 23, 1, 2, 6, 24, 25, 26, 4, 10, 8, 2, 5, 13]).to(device)
eos = torch.tensor([0, 8, 2, 24, 6, 24, 25, 26, 4, 10, 8, 2, 5, 13], dtype = torch.long).to(device)

checkpoint_epoch = 2

optim = opt.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()

model.train()

# init sampling function

def sample(model, input_seq, max_length, temperature, stochastic = True, greedy = False, eos:torch.Tensor = None):
    
    '''
    input_seq: a tensor, corresponding to a single sequence.
    ''' 
    
    assert stochastic != greedy, "stochastic sampling can not be True if greedy sampling is True"
    assert temperature > 0, "temperature must be > 0"
    assert isinstance(eos, torch.Tensor), 'eos must be torch.Tensor'
    assert isinstance(input_seq, torch.Tensor), 'input_seq must be torch.Tensor'
 
    len_eos = len(eos)
    gen = input_seq.tolist()
    input_seq = input_seq.unsqueeze(0)
  
    for _ in range(max_length):
   
        with torch.no_grad():
            out = model(input_seq)

        next_token_logits = out[:, -1,  :]
  
        if stochastic:
            next_token_probs = F.softmax(next_token_logits / temperature, dim = -1) 
            next_token = torch.multinomial(next_token_probs, num_samples = 1).item()
        elif greedy:
            next_token_probs = F.softmax(next_token_logits, dim = -1)
            next_token = torch.argmax(next_token_probs).item()
            
        gen.append(next_token)
        
        input_seq = torch.cat((input_seq, torch.tensor([[next_token]], dtype = torch.long).to(device)), dim = 1).to(device)
        
        if input_seq[-len_eos:].equal(eos):
            break

    generated_text = ''.join([idx_to_char[str(idx)] for idx in gen])

    return generated_text

print('TRAINING  ----------------------------\n')

starting_epoch = 0
ending_epoch = starting_epoch + epochs

wandb.init(
    project = 'TeXLSTM',
    name = f"TeXLSTM-epoch-{starting_epoch}-{ending_epoch}",
    config = {
        "learning_rate": lr,
        "architecture": "Stacked 2-layer LSTM",
        "dataset": 'The Stacks Project',
        "Starting Epoch": 0,
        "Ending Epoch": 0 + epochs
    }
)

total_steps = 0
sample_idx = 0

for epoch in range(epochs):
    
    for iteration, batch in enumerate(dataloader): 
      
        total_steps += 1
       
        seq = batch[0].to(device)
        labels = batch[1].to(device)
       
        logits = model(seq) 
        loss = criterion(logits.view(-1, 97), labels.view(-1))  # right
        pplx = torch.exp(loss).item()
       
        loss.backward()
        optim.step()
        optim.zero_grad() 
       
        wandb.log({"train_loss": loss.item(), "train_pplx": pplx}) 
        
        if (iteration + 1) % 10 == 0:
            print(f"Epoch: {epoch} | Total Steps: {total_steps} | Iteration: {iteration + 1} | Loss: {loss.item()} | PPLX: {pplx}")
             
        if (total_steps % eval_iter) == 0 or total_steps == 0:

            model.eval()   
            
            print(f"\nEvaluating after {total_steps} steps | Epoch {epoch} Iteration {iteration}\n")
            
            with torch.no_grad():
           
                X_test = X_test.to(device)
                logit = model(X_test)
               
                pred = F.softmax(logit, dim = -1).argmax(dim = -1)
                
                loss_test = criterion(logit.view(-1, vocab_size), y_test.view(-1)) 
                pplx_test = torch.exp(loss_test)
         
            print(f"Eval Loss: {loss_test} | Eval PPLX: {pplx_test}")

            wandb.log({'val_loss': loss_test.item(), 'val_pplx': pplx_test})
            
            #idx = torch.randint( high = X_test.size(0), size = (1, )).item()   
            #sample_seq = X_test[idx][0:sample_seq_len]
            
            if (total_steps % sample_iter) == 0:
                
                sample_seq = sample_seq 
            
                generated_text = sample(
                    model,
                    input_seq = sample_seq,
                    max_length = max_length, 
                    temperature = temperature,
                    stochastic = stochastic,
                    greedy = greedy,
                    eos = eos
                    ) 
            
                print(generated_text)
            
                sample_idx += 1 
                
                with open(f'../sampling/sample_{sample_idx}_epoch_{epoch + 1}.txt', 'w') as f:
                    f.write(generated_text) 
                
                time.sleep(sample_train_pause)
                
            model.train() 

    if (epoch + 1) % checkpoint_epoch == 0:
           
        print(f"\nSaving checkpoint at epoch {epoch + 1} with final loss: {loss}") 
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimize_state_dict': optim.state_dict(),
            'loss': loss
        }, f = f'../checkpoints/checkpoint_epoch_{epoch + 1}')