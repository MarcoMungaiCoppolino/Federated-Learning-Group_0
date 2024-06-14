import torch.nn as nn
import torch.nn.functional as F


class ShakespeareLSTM(nn.Module):
    def __init__(self, args):
        super(ShakespeareLSTM, self).__init__()

        embedding_dim = 8
        hidden_size = 256
        num_LSTM = 2
        input_length = 80
        self.n_cls = 80

        self.embedding = nn.Embedding(input_length, embedding_dim)
        self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, self.n_cls)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2) # lstm accepts in this style
        output, (h_, c_) = self.stacked_LSTM(x)
        # Choose last hidden layer
        output = self.dropout(output)
        last_hidden = output[:,-1,:]
        x = self.fc(last_hidden)

        return x

__all__ = ["ShakespeareLSTM"]