from torch import nn, Tensor

from src.models.anomaly_prediction_module import AnomalyDetectionModule
from src.models.modules.camsa_patch_tst import CAMSAPatchTST
from src.models.modules.graph_encoder import GraphCorrelationEncoder
from src.models.utils import init_transformer_encoder_weights, init_gcn_weights, ActivationFunction
from src.trainings.utils.config_enums_utils import get_activation_fn
from src.utils.config.config_reader import ConfigReader


class CAMSAModule(nn.Module):
    """
    Class combining the graph encoder and CAMSAPatchTST in a single module.

    :param graph_encoder: graph encoder to use in the module.
    :param camsa_patch_tst: path tst to use in the module.
    """

    def __init__(self, graph_encoder: GraphCorrelationEncoder, camsa_patch_tst: CAMSAPatchTST):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.camsa_patch_tst = camsa_patch_tst

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


class CPatchTSTLightning(AnomalyDetectionModule):
    """
    Implementation of the AnomalyPredictionModule.
    A Correlation-aware Patch Time Series Transformer model for time series prediction.
    Encoder:
     This model combines a graph encoder for capturing channel correlations with a
     CAMSA-PatchTST transformer architecture for sequence modeling. It processes multivariate
     time series data to generate future predictions.

     Model Steps:
         1. Graph Encoding:
            - Input time series is processed by the graph encoder
            - Extracts correlation features between different channels/variables
            - Output: correlation features tensor [batch_size, channels, correlation_features]

         2. CAMSA-PatchTST Processing:
            - Time series is divided into patches and embedded
            - Transformer processes the sequence by combining the channels correlations with the CAMSA mechanism.
            - Generates predictions for future timesteps

    Classifier: Fully Connected Model
    :param model_config: Configuration file with model's hyperparameters.
    :param training_config: Configuration file with training's hyperparameters.
     """

    def __init__(self, model_config: ConfigReader, training_config: ConfigReader):
        # Encoder
        self.model_config = model_config

        # Data characteristics
        self.channels = model_config.get_param('data.enc_in', v_type=int)
        self.pred_len = model_config.get_param('pred.len', v_type=int)
        self.seq_len = model_config.get_param('seq.len', v_type=int)

        self.output_act = model_config.get_param('head.output_act', v_type=str)

        super().__init__(training_config, self.channels, self.pred_len, self.seq_len)

    def _setup_model(self) -> nn.Module:
        """
        PatchTST encoder with Graph Correlation Encoder and CAMSA attention..
        """
        graph_encoder = GraphCorrelationEncoder(self.model_config)
        init_gcn_weights(graph_encoder)
        camsa_patch_tst = CAMSAPatchTST(self.model_config)
        init_transformer_encoder_weights(camsa_patch_tst)
        output_act = ActivationFunction(self.output_act)
        return CAMSAModule(graph_encoder, nn.Sequential([camsa_patch_tst, output_act]))
