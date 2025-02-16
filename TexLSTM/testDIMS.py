import torch
from TeXLSTM import TeXLSTM
from torchinfo import summary

batch_size = 1
seq_len = 190

model = TeXLSTM(vocab_size = 97, embedding_dim = 150, n_lstm_cells = 2, n_units = (512, 512, 97), seed = 0,)

summary(model, input_size = (batch_size, seq_len))