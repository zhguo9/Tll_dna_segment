from torch.utils.data import Dataset, DataLoader

class TargetDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    self.data.append(line.strip())
                except ValueError:
                    print("error line:", line)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data[idx]

        if self.transform:
            word = self.transform(word)

        return word

def main():
    # 创建 TextDataset 实例
    dataset = TargetDataset(file_path=r"C:\Users\silence\Documents\Git\transfer-dna\data\processedData\output.txt")

    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 遍历 DataLoader 并打印每个 batch 的数据
    for batch in dataloader:
        words = batch
        print("Batch Words:", words)
if __name__ == "__main__":
    main()

