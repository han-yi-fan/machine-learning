# BERT_data_processing.py 基于BERT神经网络模型的产品评论情感二分类方法数据预处理程序
# author：韩轶凡    version：4.0

import numpy as np
from transformers import BertTokenizer

######## 参数设置 ########

# 设置比例切分训练集、测试集
proportion = 0.8

######## 数据预处理 ########

# 读取BERT预训练词表
bert_pre_tokenizer='F:\\机器学习\\uncased_L-12_H-768_A-12'

# 逐行读取训练集文本文件
data = []
with open("test.txt","rb") as f:
    for line in f:
        sentence = str(line)[2:-5]
        data.append(sentence)

# 统计每条文本的长度，统计占比80%的文本长度，便于后期做padding
# length = []
# for i in data:
#     length.append(len(i.split(" ")))
# length.sort()
# print(length[int(0.5*len(length))],length[int(0.8*len(length))])

# 为每条文本加起始标志CLS与结尾标志SEP
# 按预训练词表对句子做分词与向量化
sentences = ['[CLS]' + sent + '[SEP]' for sent in data]
tokenizer = BertTokenizer.from_pretrained(bert_pre_tokenizer,do_lower_case=True)
tokenized_sents = [tokenizer.tokenize(sent) for sent in sentences]
input_ids = [tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents]

# 定义句子padding最大长度
MAX_LEN = 200

# PADDING
padding = []
for i in input_ids:
    if len(i) >= MAX_LEN:
        i = i[:MAX_LEN-1]
        i.append(102)
        padding.append(i)
    else:
        rest = MAX_LEN - len(i)
        for num in range(rest):
            i.append(0)
        padding.append(i)
input_ids = np.array(padding, dtype="long")

# 读取生成标签矩阵
# label = []
# with open('train_label.txt', 'r') as f:
#     for line in f:
#         label.append(line)
# labels = np.array([int(i) for i in label])

# 文本向量矩阵第一列插入标签，方便后续切分操作，打乱顺序，保证正确性
# features_with_labels = np.insert(input_ids,0,values=labels,axis=1)
# np.random.shuffle(features_with_labels)

# 拆分训练集、测试集
# train_set = features_with_labels[:200]
# test_set = features_with_labels[:200]
test_set = input_ids

# 拆分标签
# train_label = train_set[:,[0]]
# train_set = np.delete(train_set,0,axis=1)
# test_label = test_set[:,[0]]
# test_set = np.delete(test_set,0,axis=1)

# 保存矩阵
# np.savetxt('bert_train_label.txt', train_label, fmt="%d", delimiter=" ")
# np.savetxt('bert_train_set.txt', train_set, fmt="%d", delimiter=" ")
# np.savetxt('bert_test_label.txt', test_label, fmt="%d", delimiter=" ")
np.savetxt('bert_test_set.txt', test_set, fmt="%d", delimiter=" ")

# 打印矩阵大小
# print(train_label.shape,train_set.shape)
# print(test_label.shape,test_set.shape)
print(test_set.shape)

