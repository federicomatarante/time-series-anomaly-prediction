import torch
from torch import nn, Tensor

from src.models.modules.camsa_tst_backbone import CAMSATSTBackbone
from src.patchtst.layers.PatchTST_layers import series_decomp
from src.utils.config.config_reader import ConfigReader
from src.utils.config.ini_config_reader import INIConfigReader


class CAMSAPatchTST(nn.Module):
    """
    CAMSA PatchTST Model for time series forecasting that supports series decomposition.
    PatchTST model improved by using the CAMSA mechanism with the correlation embeddings of the channels.

    This model implements the PatchTST (Patch Time Series Transformer) architecture with optional
    series decomposition into trend and seasonal components.

    :param config: Configuration reader object containing model parameters
    """

    def __init__(self, config: ConfigReader, **kwargs):
        super().__init__()

        # Load model configuration
        self.decomposition = config.get_param('decomp.use', v_type=bool)
        if self.decomposition:
            self.decomp_module = series_decomp(config.get_param('decomp.kernel_size', v_type=int))
            self.model_trend = CAMSATSTBackbone(
                **self._get_backbone_params(config, **kwargs)
            )
            self.model_res = CAMSATSTBackbone(
                **self._get_backbone_params(config, **kwargs)
            )
        else:
            self.model = CAMSATSTBackbone(
                **self._get_backbone_params(config, **kwargs)
            )

    @staticmethod
    def _get_backbone_params(config_reader, **kwargs):
        """
         Extract parameters for the CAMSA TST backbone model from configuration.
        """
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

            # Class correlation tokens
            'c_features': config_reader.get_param('graph.projector.embedding_size', v_type=int),
            **kwargs
        }

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        :param x: Input tensor of shape [Batch, Seq Len, Channel]
        :param c: Correlation embeddings tensor of shape [Batch, Channels, Embedding Size]

        :return: Output tensor of shape [Batch, Pred Len, Channel]
        """
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init, c)  # NEW
            trend = self.model_trend(trend_init, c)  # NEW
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x, c)  # NEW
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x
