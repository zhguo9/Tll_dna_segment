import numpy as np
import csv
import torch
import json
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
from src.dataset.createTestDataSet import fna2Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
path_dna = os.path.join(current_dir, 'data\\processedData\\output.txt')
path_data_dir = os.path.join(current_dir, 'data\\sourceData\\')
checkpoint_dir = os.path.join(current_dir, 'checkoutpoints')

mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

@hydra.main(config_path="src/configs", config_name="config.yaml", version_base="1.1")
def inference(cfg: DictConfig) -> None:
    fna_path = cfg.fna_path
    tsv_path = fna_path.replace(".fna", ".tsv")
    fna2Dataset(fna_path,
                tsv_path,
                'C:\\Guo\\Git\\transfer-dna\\data\\processedData\\output.txt')

    # 创建模型
    model = BILSTMCRF(cfg.bilstm.voca_size, cfg.bilstm.n_class)
    model.to(device)

    # 加载模型参数
    checkpoint_file = "model_epoch_10.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    if os.path.exists(checkpoint_path):
        # 加载模型参数
        model.load_state_dict(torch.load(checkpoint_path))
        # print(f"Model parameters loaded from checkpoint file: {checkpoint_path}")

    # 加载待推理的数据集
    inference_dataset = DNADataset(file_path=path_dna, seq_length=32)
    inference_loader = DataLoader(inference_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 推理过程
    model.eval()  # 将模型设置为评估模式，关闭 dropout 和 batch normalization
    words = []
    predictions = []
    total = 30
    i = 0
    for data in inference_loader:
        # print(data)
        tmp = data.tolist()
        words.append(tmp)
        inputs = data.to(device)
        outputs = model.predict(inputs)
        # print("intput:",inputs[0],
        #       "output:",outputs[0])
        predictions.append(outputs)
    words = words[:-1]
    words = np.array(words)
    words = words.flatten()
    predictions = predictions[:-1]
    predictions = np.array(predictions)
    predictions = predictions.flatten()
    # print(words.shape, predictions.shape)
    start = 0
    end = 0
    result = []
    for i in range(len(predictions)):
        if predictions[i] == 1:
            pass
        else:
            end = i
            fragment = "".join(mapping[n] for n in words[start:end])
            result.append(fragment)
            start = end
    # print(type(result), result)
    # print(predictions[0:100],predictions.shape)
    begin_position = 16
    correct = 0
    whole = 0
    for i in range(200):
        if predictions[begin_position] == 0:
            correct += 1
        else:
            pass
        begin_position += 32
        whole += 1
        if begin_position > len(predictions):
            break
    print(correct, whole, correct / whole)
    # print(predictions[6143])
    # print(predictions[6144])
    # for i in range(0, 32 * 10, 32):
    #     group = predictions[i: i + 32]
    #     print(group[16])
    # print(result)
    # 把结果以json返回给后端
    result = "  |  ".join(result)
    # print(type(result))
    for i in range(0, len(result), 70):
        print(result[i: i+70])
    # result_json = json.dumps(result)

    print(len(predictions))
    # print(result_json)
    # for r in result:
    #     print(r)
    # file_path = "C:\\Guo\\Git\\transfer-dna\\result.csv"
    # try:
    #     with open(file_path, "w", newline="") as f:
    #         writer = csv.writer(f)
    #         for word in result:
    #             # print(word)
    #             writer.writerow([word])
    #     print("分词结果已保存到文件:", file_path)
    # except Exception as e:
    #     print("写入文件时出错:", e)



if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference()
