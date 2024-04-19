import torch
import torch.nn as nn

# 定义词汇表大小和嵌入维度
vocab_size = 5
embedding_dim = 3

# 创建一个nn.Embedding实例
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 定义输入张量，包含两个整数索引
input_tensor = torch.LongTensor([0, 2])  # 对应单词为'apple'和'cherry'

# 查找对应的词嵌入向量
embedded_vector = embedding_layer(input_tensor)

print(embedded_vector)
