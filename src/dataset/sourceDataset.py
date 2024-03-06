import torch
from torch.utils.data import Dataset, DataLoader

# 创建标签到数字的映射
label_to_number = {"B": 0, "I": 1}

class SourceDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        self.labels = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():
                    try:
                        word, label = line.strip().split()
                        self.data.append(word)
                        self.labels.append(label)
                    except ValueError:
                        print("error line:", line)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            word = self.transform(word)

        # 转换为数字元组
        numeric_tuple = tuple(label_to_number[l] for l in label)
        label = torch.as_tensor(numeric_tuple, dtype=torch.long)

        return word, label

def main():
    # 创建 TextDataset 实例
    dataset = SourceDataset(file_path=r"C:\Users\silence\Documents\Git\transfer-dna\data\dataset.txt")

    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 遍历 DataLoader 并打印每个 batch 的数据
    for batch in dataloader:
        words, labels = batch
        print("Batch Words:", words)
        print("Batch Labels:", labels)

if __name__ == "__main__":
    main()

