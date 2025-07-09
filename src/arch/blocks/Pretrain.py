
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    """Transformer encoder with multi-head attention and MLP"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.,
                 qkv_bias: bool = False, drop_rate: float = 0., 
                 attn_drop_rate: float = 0.):
        """
        Args:
            embed_dim: Dimension of embedding
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            qkv_bias: Whether to use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, N, E]
        Returns:
            Output tensor of shape [B, L, N, E]
        """
        B, L, N, E = x.shape
        x = x.reshape(B*L, N, E)
        x = x + self._sa_block(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(B, L, N, E)
        return x
        
    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention block"""
        x = x.transpose(0, 1)
        x = self.attn(x, x, x)[0]
        x = x.transpose(0, 1)
        return x