import torch 
import math
from torch import nn

class NormalizationLayer(nn.Module):
    def __init__(self, features: int, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) #weight
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
        pe = pe.unsqueeze(0) #(1,seq, features)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (b, sep_len, emb_d)
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

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
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
        self.layers = layers
        self.norm = NormalizationLayer(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, features : int, attention_block : MultiHeadAttention, cross_attention : MultiHeadAttention, feed_forward : FeedForwardBlock, dropout : float):
        super().__init__()
        self.attention = attention_block
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, trgt_mask):
        x = self.residual_connection[0](x, lambda x : self.attention(x,x,x, trgt_mask))
        x = self.residual_connection[1](x, lambda x : self.cross_attention(x,encoder_output,encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, features : int, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = NormalizationLayer(features)

    def forward(self, x, encoder_output, src_mask, trgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, features : int, vocab_size : int):
        super().__init__()
        self.proj = nn.Linear(features, vocab_size)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder, src_emb : InputEmbedding, trgt_emb : InputEmbedding, src_pos : PositionalEncoding, trgt_pos : PositionalEncoding, projection_layer : ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.trgt_emb = trgt_emb
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output : torch.Tensor, src_mask: torch.Tensor, trgt : torch.Tensor, trgt_mask : torch.Tensor):
        trgt = self.trgt_emb(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_output, src_mask, trgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    @staticmethod
    def build_transformer(src_vocab: int, trgt_vocab: int, src_seq: int, trgt_seq: int, features : int = 512, N : int = 6, h : int = 8, dropout : float = 0.1, d_ff: int = 2048):
        src_emb = InputEmbedding(src_vocab, features)
        trgt_emb = InputEmbedding(trgt_vocab, features)

        src_pos = PositionalEncoding(features, src_seq, dropout)
        trgt_pos = PositionalEncoding(features, trgt_seq, dropout)

        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttention(features, h, dropout)
            encoder_feed_forward_block = FeedForwardBlock(features, d_ff)
            encoder_block = EncoderBlock(features, encoder_self_attention_block, encoder_feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)
        
        decoder_blocks = []
        for _ in range (N):
            decoder_self_attention = MultiHeadAttention(features, h, dropout)
            decoder_cross_attention = MultiHeadAttention(features, h, dropout)
            decoder_feed_forward = FeedForwardBlock(features, d_ff)
            decoder_block = DecoderBlock(features, decoder_self_attention, decoder_cross_attention, decoder_feed_forward, dropout)
            decoder_blocks.append(decoder_block)
        
        encoder = Encoder(features, nn.ModuleList(encoder_blocks))
        decoder = Decoder(features, nn.ModuleList(decoder_blocks))

        projection_layer = ProjectionLayer(features, trgt_vocab)

        transformer = Transformer(encoder, decoder, src_emb, trgt_emb, src_pos, trgt_pos, projection_layer)
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer
    
        





