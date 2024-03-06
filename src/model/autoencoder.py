import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 数据预处理
data = [
    ['t', 'B'],
    ['h', 'I'],
    ['e', 'I'],
    ['i', 'I'],
    ['r', 'I'],
    ['e', 'B'],
    ['s', 'I'],
    ['t', 'I'],
    ['a', 'I']
]

# 构建字典
char_to_index = {char: idx for idx, char in enumerate('abcdefghijklmnopqrstuvwxyz')}
index_to_char = {idx: char for char, idx in char_to_index.items()}
pos_to_index = {'B': 0, 'I': 1}
index_to_pos = {0: 'B', 1: 'I'}

# 将数据转换为张量
def data_to_tensor(data):
    inputs = []
    targets = []
    for char, pos in data:
        inputs.append(char_to_index[char])
        targets.append(pos_to_index[pos])
    return torch.tensor(inputs), torch.tensor(targets)

# 定义Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x
#
# # 定义训练函数
# def train(model, criterion, optimizer, inputs, targets, epochs=1000):
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         outputs = model(inputs.float())
#         loss = criterion(outputs, targets.float())
#         loss.backward()
#         optimizer.step()
#         if (epoch+1) % 100 == 0:
#             print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
#
# # 设置模型参数
# input_size = len(char_to_index)
# hidden_size = 3
# model = Autoencoder(input_size, hidden_size)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # 准备数据
# inputs, targets = data_to_tensor(data)
#
# # 训练模型
# train(model, criterion, optimizer, inputs, targets)
#
# # 提取特征
# encoded_inputs = model.encoder(inputs.float())
# for idx, encoding in enumerate(encoded_inputs):
#     print(f'Character: {index_to_char[inputs[idx].item()]} - Position: {index_to_pos[targets[idx].item()]} - Encoded: {encoding.detach().numpy()}')
