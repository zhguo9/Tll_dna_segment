import torch
import torch.nn as nn
from torchcrf import CRF


class BILSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, n_class: int, embedding_dim: int = 128, rnn_units: int = 128,
                 drop_rate: float = 0.5):
        super(BILSTMCRF, self).__init__()
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, rnn_units, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(drop_rate)
        self.dense = nn.Linear(rnn_units * 2, n_class)
        self.crf = CRF(n_class, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x

    def loss_function(self, y_pred, y_true):
        mask = torch.ne(y_true, -1)
        return -self.crf(y_pred, y_true, mask=mask, reduction='mean')

    def predict(self, x):
        emissions = self.forward(x)
        return self.crf.decode(emissions)

