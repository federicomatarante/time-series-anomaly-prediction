from torch import Tensor, nn

from src.models.modules.graph_encoder import GraphCorrelationEncoder


class CAMSAModule(nn.Module):
    """
    Class combining the graph encoder and CAMSAPatchTST in a single module.

    :param graph_encoder: graph encoder to use in the module.
    :param encoder: path tst to use in the module.
    """
    def __init__(self, graph_encoder: GraphCorrelationEncoder, encoder):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.camsa_patch_tst = encoder

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: tensor of shape: # [batch_size, channels, seq_len]
        :return: tensor of [batch_size, channels, pred_len]
        """
        c = self.graph_encoder(x)  # [batch_size, channels, correlation_features]
        x = x.transpose(1, 2)  # [batch_size, seq_len, channels]
        y = self.camsa_patch_tst(x, c)  # [batch_size, pred_len, channels]
        y = y.transpose(1, 2)  # [batch_size, channels, pred_len]
        return y

