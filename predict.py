import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..\\'))




from keras.regularizers import L1L2
# from keras.engine.topology import Layer
from tensorflow.python.keras.layers import Layer
from keras import backend as K

from keras.models import Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from keras_contrib.layers import CRF
from keras.models import load_model
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy


if __name__ == '__main__':

    from DataProcess.process_data import DataProcess
    import numpy as np
    from keras.utils.vis_utils import plot_model


    dp = DataProcess(max_len=65)#最大长65
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    model = load_model('gypBi_CRF_EnFr.h5', custom_objects={"CRF": CRF, 'crf_loss': crf_loss,'crf_viterbi_accuracy': crf_viterbi_accuracy})########################
    outfile = open("F:/研究生/研究方向DNA/ncbiDNA/result/temp6.txt", "a")

    # 对比测试数据的tag
    y = model.predict(test_data)#y是预测的标签值

    label_indexs = []
    pridict_indexs = []

    num2tag = dp.num2tag()
    i2w = dp.i2w()
    count = 0
    cor_count = 0
    count_r = 0
    cor_count_r = 0
    cor_count_r1=0###############3
    cnt = 0
    cor = 0
    for i, x_line in enumerate(test_data):
        str = ""
        for j, index in enumerate(x_line):#对测试数据的每一行
            if index >= 0 and index<=63:
                char = i2w.get(index, ' ')
                t_line = y[i]
                t_index = np.argmax(t_line[j])
                tag = num2tag.get(t_index, 'O')
                if j>=2:
                    t_index_1 = np.argmax(t_line[j-2])
                    tag_1 = num2tag.get(t_index_1, 'O')
                if j >=1:
                    t_index_2 = np.argmax(t_line[j-1])
                    tag_2 = num2tag.get(t_index_2, 'O')
                if j <= len(y[i])-2:
                    t_index_3 = np.argmax(t_line[j+1])
                    tag_3 = num2tag.get(t_index_3, 'O')
                if j <= len(y[i]) - 3:
                    t_index_4 = np.argmax(t_line[j+2])
                    tag_4 = num2tag.get(t_index_4, 'O')
                pridict_indexs.append(t_index)

                t_line = test_label[i]
                t_index = np.argmax(t_line[j])
                org_tag = num2tag.get(t_index, 'O')
                label_indexs.append(t_index)
                if tag=='B':
                    count = count+1
                    str = str+' '+char
                    if org_tag=='B':
                        cor_count = cor_count+1
                else:
                    str = str+char
                if org_tag == 'B':#原标签为B，增加边界总数
                    count_r = count_r + 1
                    if tag == 'B':#如果模型预测的当前字符标签为"B"，则递增cor_count_r，即正确预测的词边界数量。
                        cor_count_r = cor_count_r + 1
                        cor_count_r1=cor_count_r1+1
                    elif j >= 2 and tag_1 =='B':#当前字符之前2个位置的字符预测的标签为"B"，则递增cor_count_r。
                        cor_count_r = cor_count_r + 1
                    elif j >= 1 and tag_2 == 'B':#当前字符之前一个位置的字符预测的标签为"B"，则递增cor_count_r。
                        cor_count_r = cor_count_r + 1
                    elif j <= len(y[i]) - 1 and tag_3 == 'B':#当前字符之后一个位置的字符预测的标签为"B"，则递增cor_count_r。
                        cor_count_r = cor_count_r + 1
                    elif j <= len(y[i]) - 1 and tag_4== 'B':#当前字符之后两个位置的字符预测的标签为"B"，则递增cor_count_r。
                        cor_count_r = cor_count_r + 1
                cnt = cnt+1
                if tag == org_tag:
                    cor = cor+1
        #print(str)
        outfile.write(str+'\n')#################
    outfile.close()
    outfile1 = open("F:/研究生/研究方向DNA/ncbiDNA/result/temp6.txt", "r")############
    content = outfile1.read()
    outfile1.close()
    os.remove("F:/研究生/研究方向DNA/ncbiDNA/result/temp6.txt")#删除分词结果文件
    words = content.split()
    word_count = len(words)
    total_length = sum(len(word) for word in words)
    average_length = total_length / word_count

    print("平均词长：", average_length)
    print("随机概率：", 1/average_length)

    print('字符级准确率：')
    print(cor/cnt)
    print('切分总数：')
    print(count)
    print('词边界级准确率：')
    print(cor_count/count)
    print('词边界级召回率：')
    print(cor_count_r / count_r)
    print('词边界级召回率（精确）：')
    print(cor_count_r1 / count_r)
    print('精确预测的：')
    print(cor_count_r1)
    print('真实边界总数：')
    print(count_r)
    print('词边界级F1指数：')
    print(2*(cor_count/count)*(cor_count_r / count_r)/((cor_count/count)+(cor_count_r / count_r)))

# 首先，通过变量count_r记录实际的词边界数量（标签中出现的B标记）。然后，在遍历每个字符时，
# 判断当前字符的标签是否为"B"，如果是，则递增count_r。
#
# 然后，通过变量cor_count_r记录模型正确预测的词边界数量。在判断当前字符标签是否为"B"时，
# 如果模型正确预测出当前字符的标签为"B"，则递增cor_count_r。
# 类似地，如果在当前字符之前和之后两个位置上的标签被正确预测为"B"，同样递增cor_count_r。
#
# 最后，召回率的计算公式为：召回率 = 正确预测的词边界数量 / 实际词边界数量。
# 在这段代码中，召回率的计算为cor_count_r / count_r。