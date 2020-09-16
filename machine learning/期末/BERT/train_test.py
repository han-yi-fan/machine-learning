# BERT.py 基于BERT模型的神经网络产品评论情感二分类方法
# author：韩轶凡   version：7.0
# 模型框架：pytorch

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig,BertForSequenceClassification
from torch.utils.data.sampler import RandomSampler,SequentialSampler
from tqdm import trange

# 指定随机种子
torch.manual_seed(666)
torch.cuda.manual_seed(666)
np.random.seed(666)

# 参数设置
batch_size = 4
lr = 2e-5
epochs = 4

# 读取处理好的两集及标签，转化为tensor类型
train_label = np.loadtxt("bert_train_label.txt",dtype="long")
train_set = np.loadtxt("bert_train_set.txt",dtype="long")
test_label = np.loadtxt("bert_test_label.txt",dtype="long")
test_set = np.loadtxt("bert_test_set.txt",dtype="long")

train_set = torch.tensor(train_set).long()
train_label = torch.tensor(train_label).long()
test_set = torch.tensor(test_set).long()
test_label = torch.tensor(test_label).long()

# 建立mask
train_masks = []
for seq in train_set:
    seq_mask = [ float( i > 0 ) for i in seq]
    train_masks.append(seq_mask)
train_masks = np.array(train_masks)
train_masks = torch.tensor(train_masks).long()

# 生成两个数据集的dataloader
train_data = TensorDataset(train_set, train_masks, train_label)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_set, test_label)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# 读取预训练的模型参数
modelConfig = BertConfig.from_pretrained('F:\\机器学习\\uncased_L-12_H-768_A-12\\bert_config.json')
model = BertForSequenceClassification.from_pretrained('F:\\机器学习\\uncased_L-12_H-768_A-12', config=modelConfig)

# 优化算法、初始化参数
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# 优化器选择
optimizer = torch.optim.Adam(optimizer_grouped_parameters,lr=lr)

# 精确率函数
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

########## 训练阶段 ##########
torch.cuda.empty_cache()
model.to(torch.device("cuda"))
# 保存loss
train_loss_set = []
for _ in trange(epochs, desc="Epoch"):

    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(torch.device("cuda")) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()

        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]
        train_loss_set.append(loss.item())
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss / nb_tr_steps))

########## 测试阶段 ##########
# 冻结网络参数，防止更新
model.eval()

with open("test_prelabel.txt","w") as f:
    for batch in test_dataloader:
        batch = tuple(t.to(torch.device("cuda")) for t in batch)
        b_input_ids, b_labels = batch
        # 测试部分不会被track梯度
        with torch.no_grad():
            logits = model( b_input_ids, token_type_ids=None )[0]
        # 模型结果矩阵
        preds = logits.detach().cpu().numpy()
        pred_flat = np.argmax(preds, axis=1).flatten()
        for i in pred_flat:
            f.write(str(i))
            f.write("\n")
print("done")






