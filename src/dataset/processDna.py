import os
import torch
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
path_data_dir = os.path.join(current_dir, '..\\..\\data\\dnaData\\')


class DNADataset(Dataset):
    def __init__(self, file_path, seq_length=32):
        self.char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.seq_length = seq_length
        self.data = self._read_data(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]

        # 填充或截断序列到指定长度
        if len(sequence) < self.seq_length:
            sequence = sequence + 'A' * (self.seq_length - len(sequence))
        else:
            sequence = sequence[:self.seq_length]

        # 将字符转换为索引
        index_sequence = [self.char_to_idx[char] for char in sequence]

        return torch.tensor(index_sequence, dtype=torch.long)

    def _read_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            sequence = ''
            for line in f:
                char = line.strip()
                # print(char[0])
                if char[0] in self.char_to_idx:
                    sequence += char[0]
                    if len(sequence) == self.seq_length:
                        data.append(sequence)
                        sequence = ''

            # 处理最后一个序列
            if sequence:
                data.append(sequence)
        # print(data)
        return data

def main():
    # 创建 DNADataset 实例
    train_dataset = DNADataset(file_path=os.path.join(path_data_dir, "output.txt"))
    # test_dataset = DNADataset(file_path=os.path.join(path_data_dir, "test.txt"))

    # 创建 DataLoader 实例
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 遍历 DataLoader 并打印每个 batch 的数据
    for batch in train_dataloader:
        print("Batch data:", batch)
        break  # 只打印第一个 batch

if __name__ == "__main__":
    main()