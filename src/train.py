import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import nltk
nltk.download('punkt')


# 准备数据集
# 假设已有英文文本数据，每行为一句话，已经分词
# 例如：["This is a sentence .", "Another sentence here ."]
sentences = ["This is a sentence .",
             "Another sentence here .",
             "Audrii body was .",]  # 英文句子列表



# 分词统计
words = [word for sentence in sentences for word in word_tokenize(sentence)]
word_counts = Counter(words)

# 构建词汇表
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print(vocab)
word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # 将词映射为索引
print(word_to_idx)
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
print(idx_to_word)
vocab_size = len(word_to_idx) + 1  # 词汇表大小，加1是为了留出一个位置给未登录词

# 将句子转换为数字表示
data = []
for sentence in sentences:
    data.append([word_to_idx[word] for word in word_tokenize(sentence)])


# 定义数据集类
class WordSegmentationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2)





# 实例化模型
embedding_dim = 100
hidden_dim = 128
model = BiLSTM(vocab_size, embedding_dim, hidden_dim)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数
optimizer = optim.Adam(model.parameters())

# 将数据转换为Tensor并定义DataLoader
train_dataset = WordSegmentationDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_data in train_loader:
        batch_data = torch.tensor(batch_data).long()
        labels = torch.zeros_like(batch_data).float()  # 标签是是否为分词位置，这里简化为都是0
        optimizer.zero_grad()
        logits = model(batch_data)
        loss = criterion(logits.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

# 将数据转换为Tensor并定义DataLoader
test_dataset = WordSegmentationDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 不需要shuffle

# 用于存储预测结果和真实标签
all_preds = []
all_labels = []

# 评估模型
model.eval()
with torch.no_grad():
    for batch_data in test_loader:
        batch_data = torch.tensor(batch_data).long()
        labels = torch.zeros_like(batch_data).float()  # 标签是是否为分词位置，这里简化为都是0
        logits = model(batch_data)
        preds = (torch.sigmoid(logits) > 0.5).int()  # 二分类预测，大于0.5为1，小于等于0.5为0
        all_preds.extend(preds.squeeze().tolist())
        all_labels.extend(labels.squeeze().tolist())

# 计算模型性能指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
