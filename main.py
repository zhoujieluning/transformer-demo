import os
import requests
import torch  # python 机器学习库
import torch.nn as nn  # 神经网络
import torch.nn.functional as F
import tiktoken
import math

# Hyperparameters
batch_size = 4
content_len = 16
d_model = 64
num_blocks = 8  # transformer blocks
num_heads = 4
learning_rate = 1e-3
dropout = 0.1
max_iters = 100  # Total of training iterations
eval_interval = 50  # How often to evaluate
eval_iters = 5  # Number of iterations to average for each evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# Load the training data
with open('sales_textbook.txt', 'r') as f:
    text = f.read()

# Tokenize the text
enc = tiktoken.get_encoding('cl100k_base')
tokenized_text = enc.encode(text)
max_token_value = max(tokenized_text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)

# Split into train and validation
train_size = int(len(tokenized_text) * .9)
train_data = tokenized_text[:train_size]
valid_data = tokenized_text[train_size:]


class FeedforwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.Relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    # 创建类的实例的时候，会调用init方法；调用类的实例时候，执行forward方法。
    def forward(self, x):
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model // num_heads)
        self.Wk = nn.Linear(d_model, d_model // num_heads)
        self.Wv = nn.Linear(d_model, d_model // num_heads)
        self.register_buffer('mask', torch.tril(torch.ones(content_len, content_len)))  # 将mask当作一个常量保存

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        attention = Q @ K.transpose(-2, -1) / math.sqrt(d_model // num_heads)
        attention.masked_fill(self.mask == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = attention @ V
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(num_heads)])
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        heads = [head(x) for head in self.heads]
        # 合头
        out = torch.cat(heads, dim=2)
        out = self.Wo(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention()
        self.feedforward_network = FeedforwardNetwork()

    def forward(self, x):
        # 论文中顺序：MultiHead => 残差连接 => LayerNorm => Feedforward => 残差连接 => LayerNorm
        # 实际大模型应用中采用的顺序有所不同：LayerNorm => MultiHead => 残差连接 => LayerNorm => Feedforward => 残差连接
        x = self.layer_norm1(x)
        x = x + self.multi_head_attention(x)
        x = self.layer_norm2(x)
        x = x + self.feedforward_network(x)
        return x


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_lookup_table = nn.Embedding(max_token_value + 1, d_model)
        # Run all the transformer blocks
        # Different from original paper, here we add a final layer norm after all the blocks
        self.transformer_blocks = nn.Sequential(*([TransformerBlock() for _ in range(num_blocks)] + [nn.LayerNorm(d_model)]))
        self.model_out_linear = nn.Linear(d_model, max_token_value + 1)

    # targets决定是否做训练还是推理
    def forward(self, idx, targets=None):
        # B: batch T: 样本长度；本例中idx形状：[4, 16] 4条数据，每条16个单词
        B, T = idx.shape
        positional_embedding_lookup_table = torch.zeros(content_len, d_model, device=device)
        position = torch.arange(0, content_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000) / d_model))
        positional_embedding_lookup_table[:, 0::2] = position * denominator
        positional_embedding_lookup_table[:, 1::2] = position * denominator
        # T第一次进来可能只走了三个单词
        position_embedding = positional_embedding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)

        logits = self.model_out_linear(x)

        if targets is not None:
            # [4, 16, 64]
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table
            idx_crop = idx[:, -self.context_length:]
            # Get predictions
            logits, loss = self(idx_crop)
            # Get the last time step from logits where the dimensions of the logits are (B,T,C)
            logits_last_timestep = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample from the probabilities' distribution.
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes idx_next to idx
            idx = torch.cat((idx, idx_next), dim=1)


model = LanguageModel().to(device)


def get_batch(typ: str):
    data = train_data if typ == 'train' else valid_data
    idxs = torch.randint(0, len(data) - content_len, (batch_size,))
    x = torch.stack([data[idx:idx + content_len] for idx in idxs])
    y = torch.stack([data[idx + 1:idx + 1 + content_len] for idx in idxs])
    return x, y


# 禁用梯度计算
@torch.no_grad()
def estimate_loss():
    out = {}
    # 模型设置为评估模式
    model.eval()
    for typ in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(typ)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[typ] = losses.mean()
    model.train()
    return out


optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model-ckpt.pt')

model.eval()
start = "The salesperson"
start_ids = enc.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=10)
print('---------------')
print(enc.decode(y[0].tolist()))
print('---------------')
