import math
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CAMSA(nn.Module):
    """
    Correlation Aware Multi-head Self Attention (CAMSA).

    This module implements a modified version of the Multi-head Self Attention mechanism
    that takes into account the correlation between channels through an additional
    embedding input. The attention is computed using both the input features and
    the channel correlation information, with separate normalization and learnable
    balancing parameters for each contribution.
    """

    def __init__(self, d_model, c_features: int, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0.,
                 proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.c_features = c_features
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        # Res attention and scale requirements
        self.res_attention = res_attention
        self.scale = nn.Parameter(torch.tensor(d_model // n_heads ** -0.5), requires_grad=lsa)
        self.lsa = lsa

        # Correlation layers for Q, K, V projections
        self.Wc_q = nn.Linear(c_features, d_k * n_heads, bias=qkv_bias)
        self.Wc_k = nn.Linear(c_features, d_k * n_heads, bias=qkv_bias)
        self.Wc_v = nn.Linear(c_features, d_v * n_heads, bias=qkv_bias)
        # Learnable norms of X and C
        self.q_norm = nn.LayerNorm(d_k * n_heads)
        self.q_norm_c = nn.LayerNorm(d_k * n_heads)
        self.k_norm = nn.LayerNorm(d_k * n_heads)
        self.k_norm_c = nn.LayerNorm(d_k * n_heads)
        self.v_norm = nn.LayerNorm(d_v * n_heads)
        self.v_norm_c = nn.LayerNorm(d_v * n_heads)

        # Output projection
        self.W_o = nn.Linear(n_heads * d_v, d_model)

        # Dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def split_heads(self, x: torch.Tensor):
        """
        Split the last dimension of the input tensor into (num_heads, d_k).

        :param x: Input tensor. Shape: [batch_size, seq_len, features]
        :return: Tensor with separated heads. Shape: [batch_size, num_heads, seq_len, d_k]
        """
        batch_size = x.size(0)
        # Reshape from (batch_size, seq_len, d_model) to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def forward(self, Q: Tensor, c: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None,
                prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, ):
        """
        :param Q: tensor of shape [Batch Size, Max_q_len, d_model]
        :param c: tensor of shape [Batch Size, Embedding Size]
        :param K: tensor of shape [Batch Size, q_len, d_model]
        :param V: tensor of shape [Batch Size, q_len, d_model]
        :param key_padding_mask: tensor of shape [q_len, q_len]
        :param attn_mask: tensor of shape [q_len, q_len]
        :return: tensor of shape [batch_size, q_len, d_model]
        """

        # If K ro Value not provided, use Query
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Repeat over 1st dimension the correlation embedding, so each token in the transformer has the same embedding
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]

        c = c.unsqueeze(1).expand(-1, seq_len, -1)
        # Q,K,V calculations
        Q = self.split_heads(
            self.q_norm(self.W_q(Q)) + self.q_norm_c(self.Wc_q(c)))
        K = self.split_heads(
            self.k_norm(self.W_k(K)) + self.k_norm_c(self.Wc_k(c)))
        V = self.split_heads(
            self.v_norm(self.W_v(V)) + self.v_norm_c(self.Wc_v(c)))

        # Scaled dot-product attention
        # (batch_size, num_heads, seq_len_q, d_k) @ (batch_size, num_heads, d_k, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale / math.sqrt(self.d_k)

        # Scores transformations
        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            scores = scores + prev

        # Apply mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # Key padding mask if provided
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # Attention weights computed w sofmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # Compute final value
        # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_v, d_k)
        context = torch.matmul(attention_weights, V)

        # Transpose and reshape back
        # (batch_size, seq_len_q, num_heads, d_k)
        context = context.transpose(1, 2).contiguous()
        # (batch_size, seq_len_q, d_model)
        context = context.view(batch_size, -1, self.n_heads * self.d_v)

        # Project in shape
        output = self.W_o(context)
        output = self.proj_dropout(output)

        if self.res_attention:
            return output, attention_weights, scores
        else:
            return output, attention_weights
