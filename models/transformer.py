import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SequencePositionEncoder(nn.Module):
    def __init__(self, emb_dim, seq_length=512):
        super().__init__()
        
        pos_enc_matrix = torch.zeros(seq_length, emb_dim)
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        scaling = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        
        pos_enc_matrix[:, 0::2] = torch.sin(pos * scaling)
        pos_enc_matrix[:, 1::2] = torch.cos(pos * scaling)
        pos_enc_matrix = pos_enc_matrix.unsqueeze(0)
        
        self.register_buffer('pos_enc', pos_enc_matrix)
        
    def forward(self, inp):
        return inp + self.pos_enc[:, :inp.size(1)]


class ParallelAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_attn_heads, dropout_prob=0.1):
        super().__init__()
        
        if emb_dim % num_attn_heads != 0:
            raise ValueError(f"Embedding dimension {emb_dim} must be divisible by number of attention heads {num_attn_heads}")
            
        self.emb_dim = emb_dim
        self.num_attn_heads = num_attn_heads
        self.head_dim = emb_dim // num_attn_heads
        
        self.q_map = nn.Linear(emb_dim, emb_dim)
        self.k_map = nn.Linear(emb_dim, emb_dim)
        self.v_map = nn.Linear(emb_dim, emb_dim)
        self.output_map = nn.Linear(emb_dim, emb_dim)
        
        self.attn_drop = nn.Dropout(dropout_prob)
        self.stored_attn = None
        
    def forward(self, q_vec, k_vec, v_vec, mask_input=None):
        batch = q_vec.size(0)
        
        q = self.q_map(q_vec).view(batch, -1, self.num_attn_heads, self.head_dim).transpose(1, 2)
        k = self.k_map(k_vec).view(batch, -1, self.num_attn_heads, self.head_dim).transpose(1, 2)
        v = self.v_map(v_vec).view(batch, -1, self.num_attn_heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask_input is not None:
            energy = energy.masked_fill(mask_input == 0, -1e9)
        
        attn = F.softmax(energy, dim=-1)
        attn = self.attn_drop(attn)
        
        self.stored_attn = attn
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.emb_dim)
        
        return self.output_map(context)


class DenseProjectionLayer(nn.Module):
    def __init__(self, in_features, expansion_factor, drop_rate=0.1):
        super().__init__()
        
        hidden_dim = in_features * expansion_factor
        
        self.dense1 = nn.Linear(in_features, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, in_features)
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x):
        return self.dense2(self.drop(F.relu(self.dense1(x))))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, ff_expansion, dropout=0.1):
        super().__init__()
        
        self.attn_mechanism = ParallelAttentionBlock(emb_dim, n_heads, dropout)
        self.projection_network = DenseProjectionLayer(emb_dim, ff_expansion, dropout)
        self.norm_attn = nn.LayerNorm(emb_dim)
        self.norm_ff = nn.LayerNorm(emb_dim)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out = self.attn_mechanism(x, x, x, mask)
        x = self.norm_attn(x + self.drop_attn(attn_out))
        
        ff_out = self.projection_network(x)
        x = self.norm_ff(x + self.drop_ff(ff_out))
        
        return x


class EmotionAnalysisModel(nn.Module):
    def __init__(self, vocab_size, num_classes=2, emb_dim=256, stack_depth=6, attn_heads=8, ff_expansion=2, 
                 max_len=512, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoder = SequencePositionEncoder(emb_dim, max_len)
        self.input_drop = nn.Dropout(dropout)
        
        self.enc_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, attn_heads, ff_expansion, dropout)
            for _ in range(stack_depth)
        ])
        
        self.output_layer = nn.Linear(emb_dim, num_classes)
        self.num_classes = num_classes
        print(f"Model configured for {num_classes} sentiment classes")
        
        self.attn_weights = []
        
    def forward(self, x):
        self.attn_weights = []
        
        embedded = self.token_embedding(x)
        encoded = self.pos_encoder(embedded)
        x = self.input_drop(encoded)
        
        for block in self.enc_blocks:
            x = block(x)
            self.attn_weights.append([block.attn_mechanism.stored_attn])
        
        pooled = x.mean(dim=1)
        return self.output_layer(pooled)
    
    def get_attention_weights(self):
        return self.attn_weights


SentimentTransformer = EmotionAnalysisModel
