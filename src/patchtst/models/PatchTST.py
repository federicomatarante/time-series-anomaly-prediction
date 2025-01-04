from torch import nn

from src.patchtst.layers.PatchTST_backbone import PatchTST_backbone
from src.patchtst.layers.PatchTST_layers import series_decomp
from src.utils.config.config_reader import ConfigReader


class Model(nn.Module):
    """
    PatchTST Model for time series forecasting that supports series decomposition.

    This model implements the PatchTST (Patch Time Series Transformer) architecture with optional 
    series decomposition into trend and seasonal components.

    Parameters
    ----------
    config : object
        Configuration object that provides model parameters via get_param() method.
        Required parameters grouped by functionality:

        Core Model Dimensions:
        - data.enc_in : int
            Number of input features/variables
        - seq.len : int
            Input sequence length
        - pred.len : int
            Prediction sequence length
        - seq.max_len : int, default=1024
            Maximum sequence length

        Patch Configuration:
        - patch.len : int
            Length of patches
        - patch.stride : int
            Stride between patches
        - patch.padding : bool
            Whether to pad patches
        - patch.padding_var : int
            Padding variable

        Encoder Configuration:
        - encoder.layers : int
            Number of encoder layers
        - encoder.heads : int
            Number of attention heads
        - encoder.dim : int
            Model dimension
        - encoder.ff_dim : int
            Dimension of feed-forward network
        - encoder.individual : bool
            Whether to use individual encoders

        Attention Configuration:
        - attn.key_dim : int
            Key dimension
        - attn.value_dim : int
            Value dimension
        - attn.dropout : float, default=0.0
            Attention dropout rate
        - attn.residual : bool, default=True
            Whether to use residual attention
        - attn.store : bool, default=False
            Whether to store attention weights
        - attn.mask : str, default='auto'
            Key padding mask type

        Dropout Configuration:
        - dropout.main : float
            Main dropout rate
        - dropout.fc : float
            Fully connected dropout rate
        - dropout.head : float
            Head dropout rate

        Normalization Configuration:
        - norm.type : str
            Normalization type ('BatchNorm', etc.)
        - norm.revin : bool
            Whether to use RevIN normalization
        - norm.pre : bool, default=False
            Whether to use pre-normalization
        - norm.affine : bool
            Whether to use affine transformation
        - norm.subtract_last : bool
            Whether to subtract last value

        Decomposition:
        - decomp.use : bool
            Whether to use series decomposition
        - decomp.kernel_size : int
            Kernel size for decomposition

        Position Encoding:
        - pos.encoding : str, default='zeros'
            Positional encoding type
        - pos.learnable : bool, default=True
            Whether to learn positional encoding

        Head Configuration:
        - head.type : str, default='flatten'
            Head type
        - head.pretrain : bool, default=False
            Whether to use pretrained head

        Activation:
        - act.type : str, default='gelu'
            Activation function type

    **kwargs : dict, optional
        Additional arguments passed to PatchTST backbone:
        - attn_mask : Tensor, optional
            Attention mask
        - verbose : bool, default=False
            Whether to print verbose info
    """

    def __init__(self, config: ConfigReader, **kwargs):
        super().__init__()

        # Load model configuration
        self.decomposition = config.get_param('decomp.use', v_type=bool)
        if self.decomposition:
            self.decomp_module = series_decomp(config.get_param('decomp.kernel_size', v_type=int))
            self.model_trend = PatchTST_backbone(
                **self._get_backbone_params(config, **kwargs)
            )
            self.model_res = PatchTST_backbone(
                **self._get_backbone_params(config, **kwargs)
            )
        else:
            self.model = PatchTST_backbone(
                **self._get_backbone_params(config, **kwargs)
            )

    @staticmethod
    def _get_backbone_params(config_reader, **kwargs):
        return {
            # Core Model Dimensions
            'c_in': config_reader.get_param('data.enc_in', v_type=int),
            'context_window': config_reader.get_param('seq.len', v_type=int),
            'target_window': config_reader.get_param('pred.len', v_type=int),
            'max_seq_len': config_reader.get_param('seq.max_len', v_type=int, default=1024),

            # Patch Configuration
            'patch_len': config_reader.get_param('patch.len', v_type=int),
            'stride': config_reader.get_param('patch.stride', v_type=int),
            'padding_patch': config_reader.get_param('patch.padding', v_type=bool),
            'padding_var': config_reader.get_param('patch.padding_var', v_type=int),

            # Encoder Configuration
            'n_layers': config_reader.get_param('encoder.layers', v_type=int),
            'n_heads': config_reader.get_param('encoder.heads', v_type=int),
            'd_model': config_reader.get_param('encoder.dim', v_type=int),
            'd_ff': config_reader.get_param('encoder.ff_dim', v_type=int),
            'individual': config_reader.get_param('encoder.individual', v_type=bool),

            # Attention Configuration
            'd_k': config_reader.get_param('attn.key_dim', v_type=int),
            'd_v': config_reader.get_param('attn.value_dim', v_type=int),
            'attn_dropout': config_reader.get_param('attn.dropout', v_type=float, default=0.0),
            'res_attention': config_reader.get_param('attn.residual', v_type=bool, default=True),
            'store_attn': config_reader.get_param('attn.store', v_type=bool, default=False),
            'key_padding_mask': config_reader.get_param('attn.mask', v_type=str, default='auto'),

            # Dropout Configuration
            'dropout': config_reader.get_param('dropout.main', v_type=float),
            'fc_dropout': config_reader.get_param('dropout.fc', v_type=float),
            'head_dropout': config_reader.get_param('dropout.head', v_type=float),

            # Normalization Configuration
            'norm': config_reader.get_param('norm.type', v_type=str),
            'revin': config_reader.get_param('norm.revin', v_type=bool),
            'pre_norm': config_reader.get_param('norm.pre', v_type=bool, default=False),
            'affine': config_reader.get_param('norm.affine', v_type=bool),
            'subtract_last': config_reader.get_param('norm.subtract_last', v_type=bool),

            # Position Encoding
            'pe': config_reader.get_param('pos.encoding', v_type=str, default='zeros'),
            'learn_pe': config_reader.get_param('pos.learnable', v_type=bool, default=True),

            # Head Configuration
            'head_type': config_reader.get_param('head.type', v_type=str, default='flatten'),
            'pretrain_head': config_reader.get_param('head.pretrain', v_type=bool, default=False),

            # Activation
            'act': config_reader.get_param('act.type', v_type=str, default='gelu'),

            **kwargs
        }

    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x
