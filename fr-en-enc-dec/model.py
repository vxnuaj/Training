import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    
    def __init__(
        self, 
        encoder_params,
        decoder_params,
        in_vocab_size, 
        out_vocab_size,
        encoder_embedding_dim,
        decoder_embedding_dim,
        batch_size,
        bos_token,
        en_pad_token,
        fr_pad_token,
        fr_eos_token,
        batch_first = True,
        seed = 0,
        ):
      
        self.device = 'cuda' 
      
        torch.manual_seed(seed)
          
        self.batch_size = batch_size
        self.bos_token = torch.tensor(bos_token).expand(batch_size) 
        self.fr_eos_token = fr_eos_token
       
        super().__init__()
       
        self.encoder_embedding = nn.Embedding(
            num_embeddings = in_vocab_size, 
            embedding_dim = encoder_embedding_dim,
            padding_idx = en_pad_token
        ).to(self.device)
        
        self.decoder_embedding = nn.Embedding(
            num_embeddings = out_vocab_size,
            embedding_dim = decoder_embedding_dim,
            padding_idx = fr_pad_token
        ).to(self.device)
       
        self.encoder = nn.ModuleList([
            nn.LSTM(
                input_size = encoder_embedding_dim,
                hidden_size = encoder_params[0],
                batch_first = batch_first 
            ).to(self.device),
            
            nn.LSTM(
                input_size = encoder_params[0],
                hidden_size = encoder_params[1],
                batch_first = batch_first
            ).to(self.device)
                        
        ])
        
        self.decoder = nn.ModuleList([
            nn.LSTM(
                input_size = decoder_embedding_dim,
                hidden_size = decoder_params[0],
                batch_first = batch_first 
            ).to(self.device),
           
            nn.LSTM(
                input_size = decoder_params[0],
                hidden_size = decoder_params[1],
                batch_first = batch_first
            ).to(self.device) 
        ])
        
        self.linear = nn.Linear(
            in_features = decoder_params[1],
            out_features = out_vocab_size,
        ).to(self.device)
    
    def forward(self, x, target_sequence:torch.Tensor):
       
        '''
        - bos should be the index in the vocab for the bos token, presented as as torch.tensor.
        - target_sequence should be the tensor of target sequences of dims: (batch_size, max_seq_len). 
            note that max_seq_len is the size of the sequence with longest length. all other shorter sequences are padded with the <pad> token.
        '''
     
        x = self.encoder_embedding(x.to(torch.long)).to(self.device)
        
        x, (h1, c1) = self.encoder[0](x) 
        _, (h2, c2) = self.encoder[1](x)

        # reminder, we are sampling autoregressively until <eos>

        outputs = []

        for t in range(target_sequence.size(1)): 
           
            if t == 0:
                decoder_input = self.bos_token.to(self.device)
                
            else:
                decoder_input = target_sequence[:, t - 1].to(self.device)
               
            decoder_input_embedding = self.decoder_embedding(decoder_input.to(torch.long)).to(self.device).unsqueeze(1)
                
            out, (h1, c1) = self.decoder[0](decoder_input_embedding, (h1, c1))  # must be the last hidden state and last cell state
            out, (h2, c2) = self.decoder[1](out, (h2, c2))
            
            out = self.linear(out) # batch_size, 1, vocab_size 
           
            outputs.append(out) 
      
        outputs = torch.cat(outputs, dim = 1) # batch_size, seq_len, vocab_size 
       
        return outputs 
   
    def test(self, x, target_sequence):
        
        x = self.encoder_embedding(x.to(torch.long)).to(self.device)
        
        x, (h1, c1) = self.encoder[0](x)
        _, (h2, c2) = self.encoder[1](x) 
        
        outputs = [ ]
        
        for t in range(target_sequence.size(1)):
            if t == 0:
                decoder_input = self.bos_token.to(self.device)
            else:
                decoder_input = pred.to(self.device).squeeze(1, 2)
           
            decoder_input_embedding = self.decoder_embedding(decoder_input.to(torch.long)).to(self.device).unsqueeze(1)

            out, (h1, c1) = self.decoder[0](decoder_input_embedding, (h1, c1))
            out, (h2, c2) = self.decoder[1](out, (h2, c2))
            
            out = self.linear(out)
            pred = out.argmax(dim = -1).unsqueeze(1) 
    
            if (pred == self.fr_eos_token).all():
                break
            
            outputs.append(out)
       
        outputs = torch.cat(outputs, dim = 1) 
            
        return outputs
            
    def generate(self, x, target_sequence, bos_token):
        
        x = self.encoder_embedding(x.to(torch.long)).to(self.device)
       
        assert int(x.size(0)) == 1, ValueError('input seq to gen func must be batch_size = 1') 
        
        x, (h1, c1) = self.encoder[0](x) 
        _, (h2, c2) = self.encoder[1](x)

        # reminder, we are sampling autoregressively until <eos>

        outputs = []

        for t in range(target_sequence.size(1)):  
           
            if t == 0:
                decoder_input = torch.tensor(bos_token).unsqueeze(0).to(self.device)
            else:
                
                decoder_input = target_sequence[:, t - 1].to(self.device) 
                
#                decoder_input = pred.to(self.device).squeeze(1, 2)
               
            decoder_input_embedding = self.decoder_embedding(decoder_input.to(torch.long)).unsqueeze(1) 
                
            out, (h1, c1) = self.decoder[0](decoder_input_embedding, (h1, c1))  # must be the last hidden state and last cell state | 
            out, (h2, c2) = self.decoder[1](out, (h2, c2)) # same as above.
            
            out = self.linear(out) # batch_size, 1, vocab_size
            pred = out.argmax(dim = -1).unsqueeze(1)
          
            outputs.append(out) 
            
            if (pred == self.fr_eos_token).all():
                break
      
        outputs = torch.cat(outputs, dim = 1) # batch_size, target seq_len, vocab_size 
        
        return outputs 
    