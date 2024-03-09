import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from dataset.processText import EngDataset
from dataset.processDna import DNADataset
from torch.optim.lr_scheduler import LambdaLR
import hydra
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from torchvision import datasets, transforms
from model.BiLSTM import BILSTMCRF
from model.kernels import GaussianKernel
from model.mkLoss import MultipleKernelMaximumMeanDiscrepancy
from torchcrf import CRF

# 定义一个简单的文本转换函数
def text_to_tensor(text):
    tensor = torch.tensor(text)
    return tensor


@hydra.main(config_path="configs", config_name="config.yaml",version_base="1.1")
def train(cfg: DictConfig) -> None:
    # 创建模型
    model = BILSTMCRF(cfg.autoencoder.input_size, cfg.autoencoder.hidden_size, cfg.bilstm.voca_size, cfg.bilstm.n_class)
    model.to(device)

    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    #define loss
    class_loss = nn.CrossEntropyLoss()
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=False
    )

    # 加载数据
    transform = transforms.Lambda(lambda x: text_to_tensor(x))
    source_dataset = EngDataset(file_path=r'C:\\Users\silence\Documents\Git\\transfer-dna\data\dataset.txt', transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=cfg.batch_size, shuffle=True)
    target_dataset = DNADataset(file_path=r'C:\Users\silence\Documents\Git\transfer-dna\data\dnaData\output.txt', seq_length=33)
    target_loader = DataLoader(target_dataset, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        # Initialize variables for accuracy calculation
        correct = 0
        total = 0

        # Use tqdm for progress bar
        source_loader_iterator = tqdm.tqdm(source_loader, desc=f'Epoch {epoch + 1}/{cfg.epochs}', leave=False)

        for i in range(cfg.iters_per_epoch):
            source_x, source_label = next(iter(source_loader))
            target_x = next(iter(target_loader))
            source_x = source_x.to(device)
            source_label = source_label.to(device)
            target_x = target_x.to(device)

            source_f, source_y = model(source_x, source_label)
            target_f, _ = model(target_x)
            source_y = source_y.to(device)

            # 计算损失
            cls_loss = source_y
            # transfer_loss = mkmmd_loss(source_f, target_f)
            transfer_loss = 0
            total_loss = cls_loss + transfer_loss * cfg.trade_off

            # 计算梯度，优化参数
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch)

            # Calculate accuracy
            source_y = torch.unsqueeze(source_y, dim = 1)
            total += source_label.size(0)
            equal = torch.squeeze(source_y) == source_label
            correct += torch.sum(equal).item()

            # Update progress bar description with accuracy
            source_loader_iterator.set_postfix(loss=total_loss.item(), accuracy=100 * correct / total, refresh=True)

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{cfg.epochs}, Loss: {total_loss.item()}, Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()