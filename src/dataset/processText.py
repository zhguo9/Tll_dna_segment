import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
path_vocab = os.path.join(current_dir, '..\\..\\data\\vocab\\vocab.txt')
path_data_dir = os.path.join(current_dir, '..\\..\\data\\sourceData\\')


def get_w2i(vocab_path=path_vocab):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return {word.strip(): idx for idx, word in enumerate(f) if word.strip()}


def get_tag2index():
    return {"B": 0, "I": 1, "[PAD]": 2}


class EngDataset(Dataset):
    def __init__(self, file_path, max_len=33, transform=None):
        self.w2i = get_w2i()
        self.tag2index = get_tag2index()
        self.vocab_size = len(self.w2i)
        self.tag_size = len(self.tag2index)
        self.unk_index = self.w2i.get('[UNK]', 65)
        self.pad_index = self.w2i.get('[PAD]', 64)
        self.max_len = max_len
        self.transform = transform

        self.data, self.labels = self._text_to_indexs(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            word = self.transform(word)

        word = torch.as_tensor(word, dtype=torch.long)
        label = torch.as_tensor(label, dtype=torch.long)

        return word, label

    def _text_to_indexs(self, file_path):
        data, labels = [], []
        with open(file_path, 'r') as f:
            line_data, line_label = [], []
            for line in f:
                if line != '\n':
                    w, t = line.split()
                    line_data.append(self.w2i.get(w, self.unk_index))
                    line_label.append(self.tag2index.get(t, 2))
                else:
                    if len(line_data) < self.max_len:
                        line_data = [self.pad_index] * (self.max_len - len(line_data)) + line_data
                        line_label = [2] * (self.max_len - len(line_label)) + line_label
                    else:
                        line_data = line_data[:self.max_len]
                        line_label = line_label[:self.max_len]
                    data.append(line_data)
                    labels.append(line_label)
                    line_data, line_label = [], []
        return data, labels


def main():
    # 创建 EngDataset 实例
    train_dataset = EngDataset(file_path=os.path.join(path_data_dir, "trainEnglish.txt"))
    test_dataset = EngDataset(file_path=os.path.join(path_data_dir, "testEnglish.txt"))

    # 创建 DataLoader 实例
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 遍历 DataLoader 并打印每个 batch 的数据
    for batch in train_dataloader:
        words, labels = batch
        print("Batch Words:", words)
        print("Batch Labels:", labels)
        break  # 只打印第一个 batch


if __name__ == "__main__":
    main()