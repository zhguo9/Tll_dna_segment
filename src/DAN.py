import torch
import torch.nn as nn
import torch.optim as optim
from dataset.sourceDataset import TextDataset
from torch.autograd import Function
import torchvision
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from model.BiLSTM import BILSTMCRF
from torchvision import datasets, transforms
import hydra
import numpy as np











# 定义一个简单的文本转换函数
def text_to_tensor(text):
    # 在这里你可以实现自己的文本转换逻辑，例如将文本转换为张量
    # 这里只是一个示例，假设文本是单词，并且转换为张量是简单地将每个字母的ASCII码作为张量元素
    tensor = torch.tensor([ord(char) for char in text])
    return tensor


@hydra.main(config_path="configs", config_name="config.yaml",version_base="1.1")
def train(cfg: DictConfig) -> None:
    # 创建模型
    # base_model = getattr(torchvision.models, cfg.model.name)(pretrained=True)
    encode_model = Autoencoder()
    base_model = BILSTMCRF(100,2)
    model = DANModel(base_model, cfg.model.num_classes, cfg.model.num_layers)

    # 创建优化器
    optimizer = getattr(torch.optim, cfg.optimizer.name)(model.parameters(), lr=cfg.optimizer.lr)

    # 加载数据
    transform = transforms.Lambda(lambda x: text_to_tensor(x))
    train_dataset = TextDataset(file_path='C:\\Users\silence\Documents\Git\\transfer-dna\data\DNA_Coded_Train.txt', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # 在训练过程中，你需要同时考虑分类器损失和 MK-MMD 损失
    num_epochs = cfg.epochs
    for epoch in range(num_epochs):
        for batch_idx, (source_data, target_data) in enumerate(train_loader):
            source_inputs, source_labels = source_data
            target_inputs, _ = target_data

            source_logits, target_logits, intermediate_source_outputs, intermediate_target_outputs = model(
                source_inputs, target_inputs)

            classifier_loss = classifier_criterion(source_logits, source_labels)
            mmd_loss = mmd_criterion(intermediate_source_outputs, intermediate_target_outputs)

            total_loss = classifier_loss + mmd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()