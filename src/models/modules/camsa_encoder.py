from typing import Optional

from torch import nn, Tensor

from src.models.modules.camsa import CAMSA
from src.patchtst.layers.PatchTST_layers import Transpose
from src.trainings.utils.config_enums_utils import get_activation_fn


class CAMSAEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False, c_features=128):
        super().__init__()

        self.layers = nn.ModuleList(
            [CAMSAEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                               attn_dropout=attn_dropout, dropout=dropout,
                               activation=activation, res_attention=res_attention,
                               pre_norm=pre_norm, store_attn=store_attn, c_features=c_features) for i in
             range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, c: Tensor, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        """
        :param src: Input tensor of shape [Batch * Channel, Patch Num, d_model]
        :param c: Correlation embeddings tensor of shape [Batch * Channels, Embedding Size]

        :return: Output tensor of shape [Batch * Channel, Patch Num, d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, c, prev=scores, key_padding_mask=key_padding_mask,
                                     attn_mask=attn_mask)  # NEW
            return output
        else:
            for mod in self.layers:
                output = mod(output, c, key_padding_mask=key_padding_mask, attn_mask=attn_mask)  # NEW
            return output


class CAMSAEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False, c_features=128):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention

        self.self_attn = CAMSA(d_model=d_model, n_heads=n_heads,
                               c_features=c_features,
                               d_k=d_k, d_v=d_v,
                               attn_dropout=attn_dropout,
                               proj_dropout=dropout,
                               res_attention=res_attention)
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, c: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        :param src: Input tensor of shape [Batch * Channel, Patch Num, d_model]
        :param c: Correlation embeddings tensor of shape [Batch * Channels, Embedding Size]

        :return: Output tensor of shape [Batch * Channel, Patch Num, d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(Q=src, K=src, V=src, c=c, prev=prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)  # NEW
        else:
            src2, attn = self.self_attn(Q=src, K=src, V=src, c=c, key_padding_mask=key_padding_mask,
                                        attn_mask=attn_mask)  # NEW
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src
