import torch
import torch.nn as nn
from torchcrf import CRF

# 定义标签
labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# 构造数据
sentences = [
    ['我', '在', '上', '海', '复', '旦', '大', '学', '学', '习'],
    ['他', '是', '中', '国', '科', '学', '院', '的', '研', '究']
]
tags = [
    ['O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O'],
    ['O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O']
]

# 将字符和标签转换为数字索引
indexed_sentences = [[i for i in range(len(sentence))] for sentence in sentences]
indexed_tags = [[label2id[tag] for tag in tag_seq] for tag_seq in tags]

# 将数据转换为PyTorch张量
sentences_tensor = torch.tensor(indexed_sentences, dtype=torch.long)
tags_tensor = torch.tensor(indexed_tags, dtype=torch.long)


# 定义BiLSTM-CRF模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def forward(self, sentences):
        embeds = self.word_embeds(sentences)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def loss(self, sentences, tags):
        lstm_feats = self.forward(sentences)
        return -self.crf(lstm_feats, tags)

    def predict(self, sentences):
        print(sentences)
        lstm_feats = self.forward(sentences)
        return self.crf.decode(lstm_feats)


# 初始化模型
vocab_size = len(sentences[0])  # 假设所有句子长度相同
tag_to_ix = label2id
embedding_dim = 10
hidden_dim = 20
model = BiLSTM_CRF(vocab_size, tag_to_ix, embedding_dim, hidden_dim)

# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(20):
    optimizer.zero_grad()
    loss = model.loss(sentences_tensor, tags_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# 预测并解码
with torch.no_grad():
    pred_tags = model.predict(sentences_tensor)
    for sentence, true_tags, pred_tags in zip(sentences, tags, pred_tags):
        print("Sentence:", sentence)
        print("True Tags:", true_tags)
        print("Pred Tags:", [id2label[tag] for tag in pred_tags])
        print()