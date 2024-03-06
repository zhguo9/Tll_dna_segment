import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from dataset.sourceDataset import SourceDataset
from dataset.targetDataset import TargetDataset
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from model.autoencoder import Autoencoder
import hydra
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from torchvision import datasets, transforms
from model.BiLSTM import BILSTMCRF
from model.kernels import GaussianKernel
from model.mkLoss import MultipleKernelMaximumMeanDiscrepancy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MK-MMD Loss
def mmd_loss(source_features, target_features):
    source_features_mean = torch.mean(source_features, dim=0)
    target_features_mean = torch.mean(target_features, dim=0)

    source_mmd = torch.mean(torch.sum((source_features - source_features_mean) ** 2, dim=1))
    target_mmd = torch.mean(torch.sum((target_features - target_features_mean) ** 2, dim=1))

    return source_mmd + target_mmd


# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, autoencoder, bilstm):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.bilstm = bilstm

    def forward(self, x):
        x = x.to(torch.float32)
        encoded_input = self.autoencoder.encoder(x)
        feature = self.autoencoder.decoder(encoded_input)
        lstm_output = self.bilstm(x)
        return feature, lstm_output


# 定义一个简单的文本转换函数
def text_to_tensor(text):
    # 在这里你可以实现自己的文本转换逻辑，例如将文本转换为张量
    # 这里只是一个示例，假设文本是单词，并且转换为张量是简单地将每个字母的ASCII码作为张量元素
    tensor = torch.tensor([ord(char) for char in text])
    return tensor

# 定义分类器损失和 MK-MMD 损失
classifier_criterion = nn.CrossEntropyLoss()

@hydra.main(config_path="configs", config_name="config.yaml",version_base="1.1")
def train(cfg: DictConfig) -> None:
    # 创建模型
    autoencoder = Autoencoder(cfg.autoencoder.input_size, cfg.autoencoder.hidden_size)
    bilstm = BILSTMCRF(cfg.bilstm.voca_size, cfg.bilstm.n_class)
    model = CombinedModel(autoencoder, bilstm)
    model.to(device)

    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    #define loss
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=False
    )

    # 加载数据
    transform = transforms.Lambda(lambda x: text_to_tensor(x))
    source_dataset = SourceDataset(file_path=r'C:\\Users\silence\Documents\Git\\transfer-dna\data\dataset.txt', transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=cfg.batch_size, shuffle=True)

    target_dataset = TargetDataset(file_path=r'C:\Users\silence\Documents\Git\transfer-dna\data\processedData\output.txt', transform=transform)
    target_loader = DataLoader(target_dataset, batch_size=cfg.batch_size, shuffle=True)


    # 在训练过程中，你需要同时考虑分类器损失和 MK-MMD 损失
    num_epochs = cfg.epochs
    for epoch in range(num_epochs):
        # Initialize variables for accuracy calculation
        correct = 0
        total = 0



        # Use tqdm for progress bar
        source_loader_iterator = tqdm.tqdm(source_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for i in range(cfg.iters_per_epoch):
            source_x, source_label = next(iter(source_loader))
            target_x = next(iter(target_loader))
            #转移到gpu
            source_x = source_x.to(device)
            source_label = source_label.to(device)
            target_x = target_x.to(device)

            source_f, source_y = model(source_x)
            target_f, target_y = model(target_x)
            # print("feature of source: ", torch.flatten(source_f))
            # print("feature of target: ", torch.flatten(target_f))


            source_label = source_label.float()
            source_label = torch.squeeze(source_label)
            source_y = torch.as_tensor(source_y)
            source_y = source_y.float()
            source_y = source_y.to(device)
            source_y = torch.squeeze(source_y)
            # print("label of source: ", source_label)
            # print("output of source: ", source_y)
            cls_loss = F.cross_entropy(source_y, source_label)
            transfer_loss = mkmmd_loss(source_f, target_f)
            # print("classification loss: ", cls_loss.item())
            # print("transfer loss: ", transfer_loss)
            loss = cls_loss + transfer_loss * cfg.trade_off

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch)

            # current_lr = optimizer.param_groups[0]['lr']
            # print("当前学习率：", current_lr)

            # Calculate accuracy
            source_y = torch.unsqueeze(source_y, dim = 1)
            # print("source_l: ", source_label)
            # print("source_y: ", source_y)
            # _, predicted = torch.max(source_y.data, 1)
            # print("predicted: ", predicted)
            # predicted = predicted.to(device)
            total += source_label.size(0)
            equal = torch.squeeze(source_y) == source_label
            # print("equal: ", equal)
            correct += torch.sum(equal).item()
            # print("correct:",correct,"total:",total)


            # Update progress bar description with accuracy
            source_loader_iterator.set_postfix(loss=loss.item(), accuracy=100 * correct / total, refresh=True)


        # Print epoch summary
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    train()
