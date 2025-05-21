# =============================================================================
# REFERENCES
#   [R1] Vaswani A. et al., “Attention Is All You Need”, 2017 – https://arxiv.org/abs/1706.03762                     # :contentReference[oaicite:0]{index=0}
#   [R2] Positional Encoding Deep Dive (blog), 2024 – https://medium.com/thedeephub/positional-encoding-explained-a-deep-dive-into-transformer-pe-65cfe8cfe10b   # :contentReference[oaicite:1]{index=1}
#   [R3] StackOverflow: Using positional encoding in PyTorch, 2023 – https://stackoverflow.com/questions/77444485   # :contentReference[oaicite:2]{index=2}
#   [R4] Mean Pooling with Sentence Transformers (blog), 2023 – https://medium.com/@joseph.nyirenda/mean-pooling-with-sentence-transformers-2e326f25b046  # :contentReference[oaicite:3]{index=3}
#   [R5] Feed-forward Network in the Transformer Model (tutorial), 2023 – https://medium.com/image-processing-with-python/the-feedforward-network-ffn-in-the-transformer-model-6bb6e0ff18db  # :contentReference[oaicite:4]{index=4}
#   [R6] HF Discussion: Attention masks ignore padding, 2022 – https://discuss.huggingface.co/t/do-automatically-generated-attention-masks-ignore-padding/15479  # :contentReference[oaicite:5]{index=5}
#   [R7] PyTorch docs: nn.LayerNorm, latest – https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html     # :contentReference[oaicite:6]{index=6}
#   [R8] PyTorch docs: nn.Embedding, latest – https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html      # :contentReference[oaicite:7]{index=7}
#   [R9] Kaur & Singh, “Residual Dropout in Transformers Improves Generalization”, 2024 – https://aclanthology.org/2024.sigul-1.35.pdf  # :contentReference[oaicite:8]{index=8}
#   [R10] PyTorch docs: nn.MultiheadAttention, latest – https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html     # :contentReference[oaicite:9]{index=9}
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SequencePositionEncoder(nn.Module):
    """
    Positional encoding for the transformer model.
    Adds information about token positions in the sequence.
    """
    def __init__(self, emb_dim, seq_length=512):
        super().__init__()
        
        # Create a tensor of shape (seq_length, emb_dim)
        pos_enc_matrix = torch.zeros(seq_length, emb_dim)
        
        # Create a tensor of shape (seq_length)
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        scaling = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        
        # Apply sine to even indices
        pos_enc_matrix[:, 0::2] = torch.sin(pos * scaling)
        
        # Apply cosine to odd indices
        pos_enc_matrix[:, 1::2] = torch.cos(pos * scaling)
        
        # Add batch dimension
        pos_enc_matrix = pos_enc_matrix.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pos_enc', pos_enc_matrix)
        
    def forward(self, inp):
        """
        Args:
            inp: Tensor of shape (batch_size, seq_len, emb_dim)
        """
        return inp + self.pos_enc[:, :inp.size(1)]


class ParallelAttentionBlock(nn.Module):
    """
    Multi-head attention module.
    Allows the model to attend to different parts of the sequence simultaneously.
    """
    def __init__(self, emb_dim, num_attn_heads, dropout_prob=0.1):
        super().__init__()
        
        # Ensure emb_dim is divisible by num_attn_heads
        if emb_dim % num_attn_heads != 0:
            raise ValueError(f"Embedding dimension {emb_dim} must be divisible by number of attention heads {num_attn_heads}")
            
        self.emb_dim = emb_dim
        self.num_attn_heads = num_attn_heads
        self.head_dim = emb_dim // num_attn_heads
        
        # Linear projections
        self.q_map = nn.Linear(emb_dim, emb_dim)
        self.k_map = nn.Linear(emb_dim, emb_dim)
        self.v_map = nn.Linear(emb_dim, emb_dim)
        self.output_map = nn.Linear(emb_dim, emb_dim)
        
        self.attn_drop = nn.Dropout(dropout_prob)
        
        # Store attention weights for visualization
        self.stored_attn = None
        
    def forward(self, q_vec, k_vec, v_vec, mask_input=None):
        batch = q_vec.size(0)
        
        # Linear projections and reshape
        q = self.q_map(q_vec).view(batch, -1, self.num_attn_heads, self.head_dim).transpose(1, 2)
        k = self.k_map(k_vec).view(batch, -1, self.num_attn_heads, self.head_dim).transpose(1, 2)
        v = self.v_map(v_vec).view(batch, -1, self.num_attn_heads, self.head_dim).transpose(1, 2)
        
        # Apply scaled dot-product attention
        energy = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask_input is not None:
            energy = energy.masked_fill(mask_input == 0, -1e9)
        
        attn = F.softmax(energy, dim=-1)
        attn = self.attn_drop(attn)
        
        # Store attention weights for later visualization
        self.stored_attn = attn
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        
        # Reshape and apply final linear projection
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.emb_dim)
        
        return self.output_map(context)


class DenseProjectionLayer(nn.Module):
    """
    Feed-forward network applied after attention.
    Consists of two linear layers with a ReLU activation in between.
    """
    def __init__(self, in_features, expansion_factor, drop_rate=0.1):
        super().__init__()
        
        hidden_dim = in_features * expansion_factor
        
        self.dense1 = nn.Linear(in_features, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, in_features)
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x):
        return self.dense2(self.drop(F.relu(self.dense1(x))))


class TransformerEncoderBlock(nn.Module):
    """
    Encoder layer of the transformer.
    Consists of multi-head attention and feed-forward network with residual connections.
    """
    def __init__(self, emb_dim, n_heads, ff_expansion, dropout=0.1):
        super().__init__()
        
        self.attn_mechanism = ParallelAttentionBlock(emb_dim, n_heads, dropout)
        self.projection_network = DenseProjectionLayer(emb_dim, ff_expansion, dropout)
        self.norm_attn = nn.LayerNorm(emb_dim)
        self.norm_ff = nn.LayerNorm(emb_dim)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection and layer normalization
        attn_out = self.attn_mechanism(x, x, x, mask)
        x = self.norm_attn(x + self.drop_attn(attn_out))
        
        # Feed-forward with residual connection and layer normalization
        ff_out = self.projection_network(x)
        x = self.norm_ff(x + self.drop_ff(ff_out))
        
        return x


class EmotionAnalysisModel(nn.Module):
    """
    Transformer model for sentiment classification.
    Consists of an embedding layer, positional encoding, and multiple encoder layers.
    """
    def __init__(self, vocab_size, emb_dim=256, stack_depth=6, attn_heads=8, ff_expansion=2, 
                 max_len=512, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoder = SequencePositionEncoder(emb_dim, max_len)
        self.input_drop = nn.Dropout(dropout)
        
        # Create multiple encoder layers
        self.enc_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, attn_heads, ff_expansion, dropout)
            for _ in range(stack_depth)
        ])
        
        # Final classification layer
        self.output_layer = nn.Linear(emb_dim, 2)  # Binary classification: negative/positive
        
        # Store attention weights for visualization
        self.attn_weights = []
        
    def forward(self, x):
        # Save attention weights if needed
        self.attn_weights = []
        
        # Apply embedding and positional encoding
        embedded = self.token_embedding(x)
        encoded = self.pos_encoder(embedded)
        x = self.input_drop(encoded)
        
        # Apply encoder layers
        for block in self.enc_blocks:
            x = block(x)
            # Store attention weights for visualization
            self.attn_weights.append([block.attn_mechanism.stored_attn])
        
        # Global average pooling
        pooled = x.mean(dim=1)
        
        # Classification
        return self.output_layer(pooled)
    
    def get_attention_weights(self):
        """
        Returns attention weights from all layers and heads.
        Used for visualization.
        """
        return self.attn_weights


SentimentTransformer = EmotionAnalysisModel