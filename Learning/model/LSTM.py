import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, num_layers, seq_length, pre_len):
        super(LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.seq_length = seq_length
        self.pre_len = pre_len

        lstm_input_size = num_nodes * in_dim

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_nodes * pre_len)

    def forward(self, x):
        x = x.permute(0, 2 ,1 ,3)
        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)

        x = x.reshape(B, T, N * C)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        pred = self.fc(last_hidden)

        pred = pred.reshape(B, N, self.pre_len)
        return pred
