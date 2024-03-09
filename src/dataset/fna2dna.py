from Bio import SeqIO

# 读取FNA文件
def extract_tokens_from_fna(input_file, output_file):
    with open(input_file, "r") as f_in:
        with open(output_file, "w") as f_out:
            for record in SeqIO.parse(f_in, "fasta"):
                sequence = str(record.seq)
                tokens = [sequence[i:i+1] for i in range(len(sequence))]
                f_out.write("\n".join(tokens))
                f_out.write("\n")

# 使用示例
input_fna_file = r"C:\Users\silence\Documents\Git\\transfer-dna\data\sourceData\h20.fna"  # 替换为你的FNA文件路径
output_txt_file = r"C:\Users\silence\Documents\Git\\transfer-dna\data\processedData\output.txt"  # 替换为输出的TXT文件路径
extract_tokens_from_fna(input_fna_file, output_txt_file)
