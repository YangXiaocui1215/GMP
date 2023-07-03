import torch.nn as nn
import torch
import numpy as np

class Attention_for_Senti_Prompt(nn.Module):
    def __init__(self, n_head=8, model_dim=768, drop_rate=0.2):
        # n_head 有几层注意力机制
        # model_dim 模型的维度
        # drop_rate 随机丢弃率
        super().__init__()
        self.n_head = n_head
        self.head_dim = model_dim // n_head     # 32//4=8
        self.wq = nn.Linear(model_dim, n_head * self.head_dim)  # [4*8]
        self.wk = nn.Linear(model_dim, n_head * self.head_dim)
        self.wv = nn.Linear(model_dim, n_head * self.head_dim)

        self.o_dense = nn.Linear(model_dim, model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, k, v, mask=None):
        # residual connect
        # q: [4, 1, 768]
        # k=v=[batch_size,seq_len, emb_dim]=[4, 3, 768]
        residual = query    # 残差

        # linear projection
        key = self.wk(k)    # [batch_size,seq_len, num_heads * head_dim]
        value = self.wv(v)  # [batch_size,seq_len, num_heads * head_dim]
        query = self.wq(query)  # [batch_size,seq_len, num_heads * head_dim]

        # 将头分离出来
        # [step,n_head,n,head_dim] = [batch_size,头的数量，seq_len,每个头的维度]
        query = self.split_heads(query) # [4,1,8,96]
        key = self.split_heads(key)     # [4,3,8,96]
        value = self.split_heads(value) # [4,3,8,96]
        
        # 自注意力机制 点乘 
        context = self.scaled_dot_product_attention(
            query, key, value, mask)    # [batch_size,seq_len, model_dim]

        # 再经过一个线性变化
        o = self.o_dense(context)       # [batch_size,seq_len, model_dim]
        # 随机使得一些权重失效
        o = self.o_drop(o)
        # layer normalization
        o = self.layer_norm(residual+o)
        return o

    def split_heads(self, x):
        x = torch.reshape(
            x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        # x = [step,n_head,n,head_dim]
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, query, k, v, mask=None):
        # query: [4, 8, 1, 96]
        # k=v: [4, 8, 3, 96]
        dk = torch.tensor(k.shape[-1]).type(torch.float) ##96
        # import pdb; pdb.set_trace()
        score = torch.matmul(query, k.permute(0, 1, 3, 2)) / (torch.sqrt(dk) + 1e-8)                 # [step, n_head, n, n]=[32, 4, 11, 11]
        if mask is not None:
            score = score.masked_fill_(mask, -np.inf) ##[4, 8, 1, 3]
        self.attention = torch.softmax(score, dim=-1)    ##[4, 8, 1, 3]
        context = torch.matmul(self.attention, v)   # [step, num_head, n, head_dim]: [4, 8, 1, 96]
        context = context.permute(0, 2, 1, 3)       # [batch_size,seq_len, num_head, head_dim]: [4, 1, 8, 96]
        context = context.reshape((context.shape[0], context.shape[1], -1)) ##[4, 1, 768]
        return context                              # [batch_size,seq_len, model_dim]




if __name__ == "__main__":
    attention = Attention_for_Senti_Prompt()
    device = torch.device('cuda:0' )
    query = torch.randn(4, 1, 768)
    key = torch.randn(4, 3, 768)
    value = key

    xx = attention(query, key, value)
    print(xx.shape)
