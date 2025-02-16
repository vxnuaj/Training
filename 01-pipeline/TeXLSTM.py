import torch
import torch.nn as nn

class TeXLSTM(nn.Module):
    
    def __init__(
        self,
        vocab_size:int,
        embedding_dim:int,
        n_lstm_cells:int,
        n_units:tuple,
        seed:int,
        device:str,
        batch_first:bool = True,
        ):


        super().__init__()

        '''
        n_units: count of units for all layers (aside from the embedding layer). 
        n_units[i] determiens the number of units for the ith layer 
        (we're counting from the first lstm layer, not from the embedding layer.)
        
        ''' 
       
        torch.manual_seed(seed) 
        
        # init model variables
        self.n_lstm_cells = n_lstm_cells 
        self.device = device
        
        # init layers 
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim
            )
        
        self.lstm_layers = nn.ModuleList()
      
        for i in range(n_lstm_cells): 
           
            if i == 0:
            
                self.lstm_layers.append(
                    nn.LSTM(
                        input_size = embedding_dim,
                        hidden_size = n_units[i],
                        batch_first = batch_first
                    )
                    )
                
            else:
                
                self.lstm_layers.append(
                    nn.LSTM(
                        input_size = n_units[i - 1],
                        hidden_size = n_units[i],
                        batch_first = batch_first 
                    )
                )

        self.linear = nn.Linear(
            in_features = n_units[-2],
            out_features = n_units[-1]
        )


    def forward(self, x):
      
        x = x.to(self.device)
        x = self.embedding(x.long())

        for i in range(self.n_lstm_cells):
            
            x, (h, _) = self.lstm_layers[i](x) 
     
               
        x = self.linear(x) 
        
        return x # TODO is the final output right?