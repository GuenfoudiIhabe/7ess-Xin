import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, head_dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum('i,j->ij', pos, inv_freq)
        self.register_buffer('cos', torch.cos(sinusoid_inp))
        self.register_buffer('sin', torch.sin(sinusoid_inp))

    def forward(self, q, k):
        b, h, seq_len, d = q.shape
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        q_ = q.view(b, h, seq_len, d//2, 2)
        k_ = k.view(b, h, seq_len, d//2, 2)

        q_rot = torch.stack([
            q_[...,0] * cos.unsqueeze(0).unsqueeze(0) - q_[...,1] * sin.unsqueeze(0).unsqueeze(0),
            q_[...,0] * sin.unsqueeze(0).unsqueeze(0) + q_[...,1] * cos.unsqueeze(0).unsqueeze(0)
        ], dim=-1).reshape_as(q)

        k_rot = torch.stack([
            k_[...,0] * cos.unsqueeze(0).unsqueeze(0) - k_[...,1] * sin.unsqueeze(0).unsqueeze(0),
            k_[...,0] * sin.unsqueeze(0).unsqueeze(0) + k_[...,1] * cos.unsqueeze(0).unsqueeze(0)
        ], dim=-1).reshape_as(k)

        return q_rot, k_rot


class ParallelAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1, max_seq_len=512):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("Embedding dim must be divisible by number of heads")
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.to_q = nn.Linear(emb_dim, emb_dim)
        self.to_k = nn.Linear(emb_dim, emb_dim)
        self.to_v = nn.Linear(emb_dim, emb_dim)
        self.to_out = nn.Linear(emb_dim, emb_dim)

        self.rotary = RotaryPositionalEncoding(self.head_dim, max_seq_len)
        self.attn_drop = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(()))
        self.stored_attn = None

    def forward(self, x_q, x_k, x_v, mask=None):
        b, seq_len, _ = x_q.size()
        q = self.to_q(x_q).view(b, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.to_k(x_k).view(b, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.to_v(x_v).view(b, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        q, k = self.rotary(q, k)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        self.stored_attn = attn

        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(b, seq_len, -1)
        return self.to_out(out)


class DenseProjectionLayer(nn.Module):
    def __init__(self, in_features, expansion, dropout=0.1):
        super().__init__()
        hidden = in_features * expansion
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, heads, ff_expansion, dropout=0.1):
        super().__init__()
        self.attn = ParallelAttentionBlock(emb_dim, heads, dropout)
        self.ffn = DenseProjectionLayer(emb_dim, ff_expansion, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x


class EmotionAnalysisModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim=256,
                 stack_depth=6,  # changed from depth
                 attn_heads=8,   # changed from heads
                 ff_expansion=2, # changed from ff_exp
                 max_len=512,
                 dropout=0.1,
                 num_classes=2): # added num_classes parameter
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, attn_heads, ff_expansion, dropout)
            for _ in range(stack_depth)
        ])
        self.classifier = nn.Linear(emb_dim, num_classes)  # dynamic class count
        self.attn_maps = []

    def forward(self, tokens, mask=None):
        self.attn_maps = []
        x = self.embed(tokens)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, mask)
            self.attn_maps.append(layer.attn.stored_attn)
        x = x.mean(dim=1)
        return self.classifier(x)

    def get_attention_weights(self):
        return self.attn_maps


SentimentTransformer = EmotionAnalysisModel(vocab_size=30000)
