import torch 
import torch.nn as nn
import math
import torch
from torch import nn

class NormalizationLayer(nn.Module):
    def __init__(self, features: int, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean)/(std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, features: int, hidden_features: int):
        super().__init__()
        self.linear1 = nn.Linear(features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, features)
    
    def forward(self,x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, features: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, features)
        self.features = features
        self.vocab_size = vocab_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.features)

class PositionalEncoding(nn.Module):
    def __init__(self, features: int, seq_len: int, dropout: float):
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, features)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, features, 2).float() * (-math.log(10000.0) / features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = NormalizationLayer(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, features: int, num_heads: int, dropout: float):
        super().__init__()
        assert features % num_heads == 0, "Features must be divisible by number of heads"
        self.features = features
        self.num_heads = num_heads
        self.head_dim = features // num_heads

        self.query = nn.Linear(features, features, bias=False)
        self.key = nn.Linear(features, features, bias=False)
        self.value = nn.Linear(features, features, bias=False)
        self.out = nn.Linear(features, features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        d_h = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_h)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        return torch.matmul(attn_weights, value), attn_weights

    def forward(self, q, k, v, mask=None):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        
        Q = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        K = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        x, self.attn_weights = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.features)

        return self.out(x) 
    
class EncoderBlock(nn.Module):
    def __init__(self, features : int, attention_block: MultiHeadAttention, feed_forward : FeedForwardBlock, dropout : float):
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.attention_block(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, features : int, layers : nn.ModuleList):
        super().__init__()
        





