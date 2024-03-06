import torch.nn as nn
import torch
from torch.autograd import Function
from torchcrf import CRF

class BILSTMCRF(nn.Module):
    def __init__(self, vocab_size, n_class, embedding_dim=128, rnn_units=128, drop_rate=0.5):
        super(BILSTMCRF, self).__init__()
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate

        # 定义模型层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(rnn_units * 2, n_class)  # 双向LSTM输出的维度是 rnn_units * 2
        self.crf = CRF(n_class)

    def forward(self, x):
        # 前向传播
        x = x.to(torch.int64)
        # print("Input x:", x)
        mask = torch.ones(x.size()).to(x.device)
        # print("Mask:", mask)
        embedded = self.embedding(x)
        # print("Embedded input:", embedded)
        lstm_out, _ = self.lstm(embedded)
        # print("LSTM output:", lstm_out)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        # print("Logits:", logits)
        logits = logits.long()
        # print(self.crf.decode(logits))
        return self.crf.decode(logits)

if __name__ == "__main__":
    # 示例用法
    vocab_size = 10000
    n_class = 2
    embedding_dim = 128
    rnn_units = 128
    drop_rate = 0.5

    # 创建模型
    model = BILSTMCRF(vocab_size, n_class, embedding_dim, rnn_units, drop_rate)

    # 打印模型结构
    print(model)