from Bio import SeqIO
import os


base_dir = r"C:\Guo\Git\transfer-dna\data"
input_fna_file = os.path.join(base_dir, "originalFna", "test.fna")  # 替换为你的FNA文件名
input_tsv_file = os.path.join(base_dir, "sourceData", "k12.tsv")
output_txt_file = os.path.join(base_dir, "processedData", "TestDataSet.txt")  # 替换为输出的TXT文件名

def reverse_complement(sequence):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(sequence))


def fna2Dataset(input_fna_file, input_tsv_file, output_txt_file):
    prefix = 16
    suffix = 16
    # 读取FNA文件
    try:
        with open(input_fna_file, 'r') as file:
            sequence = "".join(line.strip() for line in file.readlines() if not line.startswith(">"))
    except FileNotFoundError:
        return "FNA文件不存在或无法读取"

    # 读取位点信息并查找序列
    try:
        with open(input_tsv_file, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return "TSV文件不存在"

    # print(sequence)
    # print(lines)
    results = []
    hashTable = set()
    i = 1
    for line in lines:
        # print(i)
        i = i + 1
        parts = line.strip().split('\t')
        # print(parts)
        if len(parts) >= 3 and parts[8] != "protein-coding":  # 添加筛选条件
            start_position = int(parts[1])
            end_position = int(parts[2])
            strand = parts[4]  # 提取正反链信息
            # 筛去重复的行
            if start_position in hashTable:
                continue
            else:
                hashTable.add(start_position)
            # print(start_position)
            # print(hashTable)
            # if 1 <= start_position <= len(sequence) and 1 <= end_position <= len(sequence) and start_position <= end_position:
            # 把start附近的截取
            subsequence = sequence[start_position - prefix: start_position + suffix]
            if strand == "minus":  # 处理反向序列
                subsequence = reverse_complement(subsequence)
            if len(subsequence) <= 1000:  # 添加长度筛选条件
                subsequence_with_context = f"{subsequence}"
                results.append(subsequence_with_context.upper())
                # print(subsequence)
                # print(start_position)

            # 把end前后的截取
            subsequence = sequence[end_position - prefix + 1: end_position + suffix + 1]
            if strand == "minus":  # 处理反向序列
                subsequence = reverse_complement(subsequence)
            if len(subsequence) <= 1000:
                subsequence_with_context = f"{subsequence}"
                results.append(subsequence_with_context.upper())
                # print(subsequence)
                # print(end_position)
    print(len(results))
    with open(output_txt_file, 'w') as out_file:
        for result in results:
            for i in range(len(result)):
                if(i == prefix):
                    out_file.write(f"{result[i]} B\n")
                else:
                    out_file.write(f"{result[i]} I\n")

if __name__ == "__main__":
    fna2Dataset(input_fna_file, input_tsv_file, output_txt_file)