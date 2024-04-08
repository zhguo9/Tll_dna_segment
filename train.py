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
    model = BILSTMCRF(cfg.bilstm.voca_size, cfg.bilstm.n_class)
    model.to(device)

    # 检查是否存在检查点文件
    # checkpoint_file = "model_epoch_600.pt"
    # checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    # if os.path.exists(checkpoint_path):
    #     # 加载模型参数
    #     model.load_state_dict(torch.load(checkpoint_path))
    #     print(f"Model parameters loaded from checkpoint file: {checkpoint_path}")

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
    source_dataset = EngDataset(file_path=path_eng, transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=cfg.batch_size, shuffle=True)
    target_dataset = DNADataset(file_path=path_dna, seq_length=32)
    target_loader = DataLoader(target_dataset, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        total_correct = 0
        total_samples = 0

        for i in range(cfg.iters_per_epoch):
            source_x, source_label = next(iter(source_loader))
            target_x = next(iter(target_loader))
            source_label = source_label.to(device)

            source_x = source_x.to(device)
            target_x = target_x.to(device)

            y_s = model.predict(source_x)
            y_t = model.predict(target_x)

            f_s = model.get_feature(source_x)
            f_t = model.get_feature(target_x)

            y_s = torch.tensor(y_s)
            y_s = y_s.to(device)

            # 计算准确率
            # print("y:",source_y[0],"\n", "l:", source_label[0])
            batch_correct = torch.sum((y_s == source_label).int()).item()

            total_correct += batch_correct
            total_samples += source_label.size(0) * 32
            # print(batch_correct, total_correct, total_samples)

            # 计算损失
            cls_loss = model.loss(source_x, source_label)
            # cls_loss = torch.nn.functional.cross_entropy(y_s.float(), source_label.float())
            align_weight = 100.0
            tf_loss = mkmmd_loss(f_s, f_t)
            total_loss = cls_loss + align_weight * tf_loss
            print("class_loss: ",cls_loss, "feature loss: ", tf_loss * align_weight, "total loss: ", total_loss)

            # 计算梯度，优化参数
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch)
        # 计算平均准确率
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}, Accuracy: {accuracy}, Loss: {total_loss}")

        # 在固定的轮数增加检查点
        if (epoch + 1) % cfg.checkpoint_interval == 0:
            # 构建检查点文件名
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            # 保存模型状态
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

if __name__ == "__main__":
    # 忽略特定类别的警告
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()