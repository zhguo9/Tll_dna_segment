import torch
import torch.nn as nn
from torchcrf import CRF

class BILSTMCRF(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 n_class: int,
                 embedding_dim: int = 128,
                 rnn_units: int = 128,
                 drop_rate: float = 0.5,
                 input_dim: int = 33,
                 hidden_dim: int = 5,
                 ):
        super(BILSTMCRF, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        self.vocab_size = vocab_size
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.bilstm = nn.LSTM(input_size=self.embedding_dim,
                              hidden_size=self.rnn_units,
                              num_layers=1,
                              bidirectional=True,
                              batch_first=True)
        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(rnn_units * 2, self.n_class)
        self.crf = CRF(self.n_class, batch_first=True)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        # print(x.shape)
        x = self.linear(x)
        # print("111",x.shape)
        # x = self.crf(x, y)
        # print("222",x.shape,x)
        return x

    def get_feature(self, x):
        # print(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def loss(self, x, y):
        outputs = self.forward(x)
        loss = -self.crf(outputs, y)
        return loss

    def predict(self, x):
        emission = self.forward(x)
        return self.crf.decode(emission)

