import torch
import torch.nn as nn
import torch.nn.functional as F
class TextFeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_size):
        super(TextFeatureExtractor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, num_filters, filter_size, padding=filter_size//2)  # 添加padding以保持长度
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Linear(32, 5)
        self.decoder = nn.Linear(5, 32)
    def forward(self, x):
        # x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        # x = x.mean(dim=1)  # 取平均以获得碱基嵌入的序列表示 [batch_size, embedding_dim]
        encoded = self.encoder(x.float())  # [batch_size, latent_dim]
        decoded = self.decoder(encoded)
        decoded = torch.clamp(decoded, min=0, max=26)
        return encoded, decoded

class DNAFeatureExtractor(nn.Module):
    def __init__(self, num_bases, embedding_dim, num_filters, filter_size):
        super(DNAFeatureExtractor, self).__init__()
        self.embedding = nn.Embedding(num_bases, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, num_filters, filter_size, padding=filter_size//2)  # 添加padding以保持长度
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Linear(32, 5)
        self.decoder = nn.Linear(5, 32)

    def forward(self, x):
        # x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        # x = x.mean(dim=1)  # 取平均以获得碱基嵌入的序列表示 [batch_size, embedding_dim]
        encoded = self.encoder(x.float())  # [batch_size, latent_dim]
        decoded = self.decoder(encoded)
        decoded = torch.clamp(decoded, min=0, max=4)
        return encoded, decoded


# class TextFeatureExtractor(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, num_filters, filter_size):
#         super(TextFeatureExtractor, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.conv = nn.Conv1d(embedding_dim, num_filters, filter_size, padding=filter_size//2)  # 添加padding以保持长度
#         self.pool = nn.AdaptiveAvgPool1d(1)
#
#     def forward(self, x):
#         x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
#         x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_length] 为Conv1d准备
#         x = self.conv(x)  # [batch_size, num_filters, seq_length]
#         x = self.pool(x)  # [batch_size, num_filters, 1]
#         x = x.squeeze(2)  # [batch_size, num_filters]
#         return x
#
# class DNAFeatureExtractor(nn.Module):
#     def __init__(self, num_bases, embedding_dim, num_filters, filter_size):
#         super(DNAFeatureExtractor, self).__init__()
#         self.embedding = nn.Embedding(num_bases, embedding_dim)
#         self.conv = nn.Conv1d(embedding_dim, num_filters, filter_size, padding=filter_size//2)  # 添加padding以保持长度
#         self.pool = nn.AdaptiveAvgPool1d(1)
#
#     def forward(self, x):
#         x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
#         x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_length] 为Conv1d准备
#         x = self.conv(x)  # [batch_size, num_filters, seq_length]
#         x = self.pool(x)  # [batch_size, num_filters, 1]
#         x = x.squeeze(2)  # [batch_size, num_filters]
#         return x
