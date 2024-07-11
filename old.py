import os
import requests
import torch  # python 机器学习库
import torch.nn as nn  # 神经网络
import torch.nn.functional as F
import tiktoken
import math

# 数据集下载
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/blob/main/sales_textbook.txt?download=true'
    with open('sales_textbook.txt', 'wb') as f:
        f.write(requests.get(url).content)

with open('sales_textbook.txt', 'r') as f:
    text = f.read()

# 数据集tokenization，并生成对应的张量
# enc = tiktoken.get_encoding("o200k_base")
enc = tiktoken.get_encoding("cl100k_base")
tokenized_text = enc.encode(text)  # 拿到的是一个list
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)  # 生成一维张量
# print(enc.decode([919]))

# 数据集分成两部分：90%用于训练，剩下的用于验证
train_end_idx = int(len(tokenized_text) * .9)
train_data = tokenized_text[:train_end_idx]
valid_data = tokenized_text[train_end_idx:]

# 人为设定的参数 超参数 hyperparameters
batch_size = 4  # 一次训练的样本数量
context_len = 16  # 样本长度（16个词）
d_model = 64  # 维度 ？
n_head = 4  # 把Q,K,V维度切成四份


###################################################################
# word embbeding 词嵌入 将词转换为向量，方便计算机进行计算，通过不同词向量之间夹角的余弦值，计算相似性
###################################################################
data = train_data
idxs = torch.randint(low=0, high=len(data) - context_len, size=(batch_size,))  # 取四个随机整数下标;[low, high),size要传tuple类型
# x用于预测下一个单词，y用于检验x是否预测正确，所以y要比x整体错后一个单词。
x_batch = torch.stack([data[idx:idx + context_len] for idx in idxs])  # 4个长度为16的一维张量，stack成一个4*16的二维张量
y_batch = torch.stack([data[idx + 1:idx + 1 + context_len] for idx in idxs])

max_token_value = tokenized_text.max().item()  # 数据集中最大的token
input_embedding_lookup_table = nn.Embedding(max_token_value + 1, d_model)
# print('input_embedding_lookup_table', input_embedding_lookup_table)
x_batch_embedding = input_embedding_lookup_table(x_batch)  # [4, 16, 64]三维张量
y_batch_embedding = input_embedding_lookup_table(y_batch)  # [4, 16, 64]三维张量

###################################################################
# positional encoding 加入位置信息
###################################################################
position_encoding_lookup_table = torch.zeros(context_len, d_model)  # [16, 64] 二维张量，全部填充0
position = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1)  # arange：生成一个长度16的一维张量；unsqueeze(1)：升维成[16,1]的二维张量
# print('position', position)
# print(position.shape)

print(torch.arange(0, d_model, 2, dtype=torch.float))
# 分母 10000^(2i / d_model), 长度为32的一维张量
denominator = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000) / d_model))
# print('denominator', denominator)
# 对偶数维执行 sin(pos/10000^(2i/d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * denominator)
# 对奇数维执行 cos(pos/10000^(2i/d_model))
position_encoding_lookup_table[:, 1::2] = torch.cos(position * denominator)
# unsqueeze(0)：升维成三维张量[1, 16, 63]; expand：将第一维扩展为四个(拷贝)，二三维不变。最终变成[4, 16, 64]的三维张量
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1)

# print('position_encoding_lookup_table', position_encoding_lookup_table)
# print('position_encoding_lookup_table', position_encoding_lookup_table.shape)

x = x_batch_embedding + position_encoding_lookup_table
y = y_batch_embedding + position_encoding_lookup_table

###################################################################
# Multi-Head Attention 多头注意力机制
###################################################################

# 权重 [64, 64]
Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

# 对QKV进行y=wx+b线性变换
Q = Wq(x)
K = Wk(x)
V = Wv(x)

# 分头： [4, 16, 64] => [4, 16, 4, 16] => [4, 4, 16, 16]
Q = Q.reshape(batch_size, context_len, n_head, d_model//n_head).permute(0, 2, 1, 3)
K = K.reshape(batch_size, context_len, n_head, d_model//n_head).permute(0, 2, 1, 3)
V = V.reshape(batch_size, context_len, n_head, d_model//n_head).permute(0, 2, 1, 3)

# K.transpose(-2, -1),最后两维转置
# MatMul: Q * (K的转置)，得出的是注意力矩阵，矩阵里的每一个值，是两个单词之间的注意力
# Scale: 根号dk分之一
# [4, 4, 16, 16]
output = Q @ K.transpose(-2, -1) / math.sqrt(d_model//n_head)

# torch.ones：生成一个[16, 16]的二维张量，并且所有值都填充1
# torch.triu diagonal=1：保持主对角线以上的元素不变，主对角线以下包括主对角线的元素都重设为0
# bool(): 转换为bool值
# mask: [16, 16]，主对角线以上的元素值都为True，主对角线及以下的值都为false
mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()

# Mask
# masked_fill：根据output中元素的位置，找到对应的mask中的元素，如果值为True，将output中对应位置的元素值设为负无穷
# output：主对角线以上的值都为负无穷
output = output.masked_fill(mask, float('-inf'))

# softMax dim=-1表示对行进行softmax
attention_score = F.softmax(output, dim=-1)

# MatMul V
# A包含了所有元素之间的相互关系，这也是为什么叫自注意力层的原因
A = attention_score @ V

# [4, 4, 16, 16] => [4, 16, 4, 16] => [4, 16, 64]
A = A.permute(0, 2, 1, 3).reshape(batch_size, context_len, d_model)

Wo = nn.Linear(d_model, d_model)
# 对A进行y=wx+b的线性变换
output = Wo(A)

# 残差连接 residual connections
output = output + x

# 层归一化 layer normalization
layer_norm = nn.LayerNorm(d_model)
layer_norm_output = layer_norm(output)

# 前馈网络 feedforward network ?
output = nn.Linear(d_model, d_model * 4)(layer_norm_output)
output = nn.ReLU()(output)
output = nn.Linear(d_model * 4, d_model)(output)

# 第二次残差连接
output = layer_norm_output + output

# 第二次层归一化
output = layer_norm(output)

###################################################################
# Linear 最终的线性变换
###################################################################
# [4, 16, 100070) 字典表里每一个字对应当前字的注意力，注意力越大，说明下一个字是他的概率越高
output = nn.Linear(d_model, max_token_value + 1)(output)

# 再进行一次softmax
logits = F.softmax(output, dim=-1)

###################################################################
# 验证结果
###################################################################
# 原始第一句x
x_idxs = [idx.item() for idx in x_batch[0]]
print('x:', [enc.decode([idx]) for idx in x_idxs])
# 原始第一句y
y_idxs = [idx.item() for idx in y_batch[0]]
print('y:', [enc.decode([idx]) for idx in y_idxs])

#预测出的第一句
# argmax 找出当前行概率最大的一个值
list16 = list(range(0, 16))
predicted_idxs = [torch.argmax(logits[0, idx]).item() for idx in list16]
res = [enc.decode([idx]) for idx in predicted_idxs]
print('预测结果:', res)

