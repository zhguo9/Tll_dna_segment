from Bio import SeqIO
import os

# 指定起点路径
base_dir = r"C:\Guo\Git\transfer-dna\data"

# 构建绝对路径
input_fna_file = os.path.join(base_dir, "sourceData", "h20.fna")  # 替换为你的FNA文件名
output_txt_file = os.path.join(base_dir, "processedData", "output.txt")  # 替换为输出的TXT文件名


# 读取FNA文件
def extract_tokens_from_fna(input_file, output_file):
    with open(input_file, "r") as f_in:
        with open(output_file, "w") as f_out:
            for record in SeqIO.parse(f_in, "fasta"):
                sequence = str(record.seq)
                tokens = [sequence[i:i+1] for i in range(len(sequence))]
                f_out.write("\n".join(tokens))
                f_out.write("\n")

extract_tokens_from_fna(input_fna_file, output_txt_file)
