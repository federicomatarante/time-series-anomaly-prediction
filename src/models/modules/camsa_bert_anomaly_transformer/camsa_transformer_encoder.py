import copy
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np

import torch
import torch.nn as nn

from timm.layers import trunc_normal_


def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Main transformer encoder
class CAMSATransformerEncoder(nn.Module):
    def __init__(self, positional_encoding_layer, encoder_layer, n_layer):
        super(CAMSATransformerEncoder, self).__init__()
        self.encoder_layers = clone_layer(encoder_layer, n_layer)

        self.positional_encoding = True if positional_encoding_layer is not None else False
        if self.positional_encoding:
            self.positional_encoding_layer = positional_encoding_layer

    def forward(self, x, c):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        position_vector = None
        if self.positional_encoding:
            out = self.positional_encoding_layer(x)
        else:
            out = x

        for layer in self.encoder_layers:
            out = layer(out, c)

        return out


# Encoder layer
class CAMSAEncoderLayer(nn.Module):
    def __init__(self, attention_layer, feed_forward_layer, norm_layer, dropout=0.1):
        super(CAMSAEncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = clone_layer(norm_layer, 2)
        self.dropout_layer = nn.Dropout(p=dropout)

        for p in self.attention_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x, c):
        out1 = self.norm_layers[0](x)  # Layer norm first
        out1 = self.attention_layer(out1, c)
        out1 = self.dropout_layer(out1) + x

        out2 = self.norm_layers[1](out1)
        out2 = self.feed_forward_layer(out2)
        return self.dropout_layer(out2) + out1


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_embed, n_head, max_seq_len_c, max_seq_len=512, relative_position_embedding=True,
                 causal_mask=False):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_embed % n_head == 0  # Ckeck if d_model is divisible by n_head.

        self.d_embed = d_embed
        self.n_head = n_head
        self.d_k = d_embed // n_head
        self.scale = 1 / np.sqrt(self.d_k)

        self.x_query = nn.Linear(d_embed, d_embed)
        self.c_key = nn.Linear(d_embed, d_embed)
        self.c_value = nn.Linear(d_embed, d_embed)
        self.out_linear = nn.Linear(d_embed, d_embed)

        self.max_seq_len = max_seq_len
        self.max_seq_len_c = max_seq_len_c
        self.relative_position_embedding = relative_position_embedding

        self.causal_mask = causal_mask

        if relative_position_embedding:
            self.relative_position_embedding_table = nn.Parameter(
                torch.zeros((max_seq_len + max_seq_len_c - 1, n_head))
            )
            trunc_normal_(self.relative_position_embedding_table, std=.02)

            coords_x = torch.arange(max_seq_len)
            coords_c = torch.arange(max_seq_len_c)

            relative_coords = coords_x[:, None] - coords_c[None, :]

            relative_coords += max_seq_len_c - 1

            # Salviamo gli indici come buffer del modulo
            self.register_buffer('relative_position_index', relative_coords)
        # Casual mask
        if causal_mask:
            # Create a lower triangular matrix of ones
            mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            self.register_buffer('causal_attention_mask', mask)

    def forward(self, x, c):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        batch_size = x.shape[0]

        # x: (batch_size, x_len, d_model)
        # c: (batch_size, c_len, d_model)

        q = self.x_query(x)  # (batch_size, x_len, d_model)
        k = self.c_key(c)  # (batch_size, c_len, d_model)
        v = self.c_value(c)  # (batch_size, c_len, d_model)

        q = q.view(batch_size, -1, self.n_head, self.d_k)
        k = k.view(batch_size, -1, self.n_head, self.d_k)
        v = v.view(batch_size, -1, self.n_head, self.d_k)

        q = q.transpose(1, 2)  # (batch_size, n_heads, x_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, n_heads, c_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, c_len, head_dim)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        # (batch_size, n_heads, x_len, c_len)

        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        # Add relative position embedding
        if self.relative_position_embedding:
            relative_position_index = self.relative_position_index[:self.max_seq_len, :self.max_seq_len_c]

            relative_position_embeddings = self.relative_position_embedding_table[
                relative_position_index
            ]  # (seq_len_x, seq_len_c, n_head)

            relative_position_embeddings = relative_position_embeddings.permute(2, 0, 1)
            # (n_head, seq_len_x, seq_len_c)

            attention_scores = attention_scores + relative_position_embeddings.unsqueeze(0)
        if self.causal_mask is not None:
            attention_scores = attention_scores + self.causal_mask * (
                -1e9)  # Add very small negative number to padding columns.
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # (batch_size, n_heads, x_len, c_len)

        out = torch.matmul(attention_weights, v)
        # (batch_size, n_heads, x_len, head_dim)

        out = out.transpose(1, 2)  # (batch_size, x_len, n_heads, head_dim)
        out = out.contiguous().view(batch_size, -1, self.d_embed)
        # (batch_size, x_len, d_model)

        out = self.out_linear(out)  # (batch_size, x_len, d_model)
        return out


def get_camsa_transformer_encoder(
        max_seq_len_c, d_embed=512,
        positional_encoding=None,
        relative_position_embedding=True,
        n_layer=6,
        n_head=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        causal_mask=False
):
    if positional_encoding == 'Sinusoidal' or positional_encoding == 'sinusoidal' or positional_encoding == 'sin':
        positional_encoding_layer = SinusoidalPositionalEncoding(d_embed, max_seq_len, dropout)
    elif positional_encoding == 'Absolute' or positional_encoding == 'absolute' or positional_encoding == 'abs':
        positional_encoding_layer = AbsolutePositionEmbedding(d_embed, max_seq_len, dropout)
    elif positional_encoding == None or positional_encoding == 'None':
        positional_encoding_layer = None

    attention_layer = MultiHeadAttentionLayer(d_embed, n_head, max_seq_len_c, max_seq_len, relative_position_embedding,
                                              causal_mask)
    feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    encoder_layer = CAMSAEncoderLayer(attention_layer, feed_forward_layer, norm_layer, dropout)

    return CAMSATransformerEncoder(positional_encoding_layer, encoder_layer, n_layer)
