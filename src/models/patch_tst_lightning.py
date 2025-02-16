from torch import nn

from src.models.anomaly_prediction_module import AnomalyDetectionModule
from src.models.utils import Permute, init_transformer_encoder_weights, ActivationFunction
from src.patchtst.models.PatchTST import Model as PatchTST
from src.utils.config.config_reader import ConfigReader


class PatchTSTLightning(AnomalyDetectionModule):
    """
    Base Implementation of the AnomalyPredictionModule.
    Encoder: Base PatchTST
    Classifier: Fully Connected Layer
    :param model_config: Configuration file with model's hyperparameters.
    :param training_config: Configuration file with training's hyperparameters.
    """

    def __init__(self, model_config: ConfigReader, training_config: ConfigReader):
        # Encoder
        self.model_config_reader = model_config

        # Data characteristics
        self.channels = model_config.get_param('data.enc_in', v_type=int)
        self.pred_len = model_config.get_param('pred.len', v_type=int)
        self.seq_len = model_config.get_param('seq.len', v_type=int)

        # Output at
        self.output_act = model_config.get_param('head.output_act', v_type=str)
        super().__init__(training_config, self.channels, self.pred_len, self.seq_len)

    def _setup_model(self) -> nn.Module:
        """
        PatchTST standard encoder.
        """
        encoder = PatchTST(self.model_config_reader)
        init_transformer_encoder_weights(encoder)
        permute = Permute(0, 2, 1)
        output_act = ActivationFunction(self.output_act)
        return nn.Sequential(
            permute, encoder, permute, output_act
        )
