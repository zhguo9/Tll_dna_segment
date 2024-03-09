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
from torchcrf import CRF

def text_to_tensor(text):
    tensor = torch.tensor(text)
    return tensor

@hydra.main(config_path="configs", config_name="config.yaml",version_base="1.1")
def train(cfg: DictConfig) -> None:
    model = BILSTMCRF(cfg.autoencoder.input_size, cfg.autoencoder.hidden_size, cfg.bilstm.voca_size, cfg.bilstm.n_class)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Lambda(lambda x: text_to_tensor(x))
    source_dataset = EngDataset(file_path=r'C:\\Users\silence\Documents\Git\\transfer-dna\data\dataset.txt', transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        source_x, source_label = next(iter(source_loader))
        source_x = source_x.to(device)
        source_label = source_label.to(device)

        source_f, source_y = model(source_x)
        source_y = source_y.to(device)

        cls_loss = 0
        transfer_loss = 0
        loss = cls_loss + transfer_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()