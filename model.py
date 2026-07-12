import torch 
import torch.nn as nn
import math
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
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        return torch.matmul(attn_weights, value), attn_weights

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.features)

        return self.out(context) 

