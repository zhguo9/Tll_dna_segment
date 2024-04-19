import torch
import torch.nn as nn
import tqdm
import os
import torch.nn.functional as F
from src.dataset.processText import EngDataset
from src.dataset.processDna import DNADataset
from torch.optim.lr_scheduler import LambdaLR
import hydra
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from torchvision import datasets, transforms
from src.model.BiLSTM import BILSTMCRF
from src.model.kernels import GaussianKernel
from torchcrf import CRF
import os
from src.model.mkLoss import MultipleKernelMaximumMeanDiscrepancy
import sys
import datetime
import logging
import warnings
from src.model.getFeature import TextFeatureExtractor, DNAFeatureExtractor



# 定义一个简单的文本转换函数
def text_to_tensor(text):
    tensor = torch.tensor(text)
    return tensor

current_dir = os.path.dirname(os.path.abspath(__file__))
path_eng = os.path.join(current_dir, 'data\\EngDataset.txt')
path_dna = os.path.join(current_dir, 'data\\processedData\\output.txt')
path_data_dir = os.path.join(current_dir, 'data\\sourceData\\')

checkpoint_dir = os.path.join(current_dir, 'checkoutpoints')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

@hydra.main(config_path="src/configs", config_name="config.yaml", version_base="1.1")
def train(cfg: DictConfig) -> None:
    # 创建模型
    eng_encoder = TextFeatureExtractor(cfg.eng_extractor.voca_size,
                                       cfg.eng_extractor.embedding_dim,
                                       cfg.eng_extractor.num_filter,
                                       cfg.eng_extractor.filter_size,
                                       )
    eng_encoder.to(device)
    dna_encoder = DNAFeatureExtractor(cfg.dna_extractor.voca_size,
                                      cfg.dna_extractor.embedding_dim,
                                      cfg.dna_extractor.num_filter,
                                      cfg.dna_extractor.filter_size,
                                      )
    dna_encoder.to(device)
    eng_lstm = BILSTMCRF(cfg.bilstm.voca_size, cfg.bilstm.n_class)
    dna_lstm = BILSTMCRF(4, 2)
    eng_lstm.to(device)
    dna_lstm.to(device)

    # Define your optimizer
    optimizer1 = torch.optim.Adam(eng_lstm.parameters(), lr=0.001)
    lr_scheduler1 = LambdaLR(optimizer1, lr_lambda=lambda epoch: 0.95 ** epoch)

    optimizer2 = torch.optim.Adam(dna_lstm.parameters(), lr=0.001)
    lr_scheduler2 = LambdaLR(optimizer2, lr_lambda=lambda epoch: 0.95 ** epoch)

    optimizer3 = torch.optim.Adam(eng_encoder.parameters(), lr=0.001)
    lr_scheduler3 = LambdaLR(optimizer3, lr_lambda=lambda epoch: 0.95 ** epoch)

    optimizer4 = torch.optim.Adam(dna_encoder.parameters(), lr=0.001)
    lr_scheduler4 = LambdaLR(optimizer4, lr_lambda=lambda epoch: 0.95 ** epoch)

    #define loss
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=False
    )

    # 加载数据
    transform = transforms.Lambda(lambda x: text_to_tensor(x))
    source_dataset = EngDataset(file_path=path_eng, transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=cfg.batch_size, shuffle=True)
    target_dataset = DNADataset(file_path=path_dna, seq_length=32)
    target_loader = DataLoader(target_dataset, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        total_correct = 0
        total_samples = 0

        for i in range(cfg.iters_per_epoch):
            # 加载数据
            source_x, source_label = next(iter(source_loader))
            target_x = next(iter(target_loader))
            # print(source_x, target_x)
            source_label = source_label.to(device)
            source_x = source_x.to(device)
            target_x = target_x.to(device)

            f_s, x_s = eng_encoder(source_x)
            f_t, x_t = dna_encoder(target_x)

            y_s = eng_lstm.predict(x_s)
            y_s = torch.as_tensor(y_s)
            y_s = y_s.to(device)
            batch_correct = torch.sum((y_s == source_label).int()).item()
            total_correct += batch_correct
            total_samples += source_label.size(0) * 32

            # print(f_s)
            # print(f_t)
            class_loss = eng_lstm.loss(x_s, source_label)
            feature_loss = mkmmd_loss(f_s, f_t)
            total_loss = class_loss + feature_loss
            print(class_loss, feature_loss)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            lr_scheduler1.step(epoch)
            lr_scheduler2.step(epoch)
            lr_scheduler3.step(epoch)
            lr_scheduler4.step(epoch)
        # 计算平均准确率
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}, Accuracy: {accuracy}, Loss: {total_loss}")

        # 在固定的轮数增加检查点
        if (epoch + 1) % cfg.checkpoint_interval == 0:
            # 构建检查点文件名
            checkpoint_path = os.path.join(checkpoint_dir, f"encoder_epoch_{epoch+1}.pt")
            # 保存模型状态
            torch.save(dna_encoder.state_dict(), checkpoint_path)
            # 构建检查点文件名
            checkpoint_path = os.path.join(checkpoint_dir, f"lstm_epoch_{epoch+1}.pt")
            # 保存模型状态
            torch.save(dna_lstm.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

if __name__ == "__main__":
    # 忽略特定类别的警告
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()