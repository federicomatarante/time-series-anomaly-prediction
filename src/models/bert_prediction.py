from torch import nn

from src.models.anomaly_prediction_module import AnomalyPredictionModule
from src.models.modules.bert_anomaly_transformer.anomaly_prediction_transformer import get_anomaly_prediction_bert
from src.models.utils import Permute, init_transformer_encoder_weights, init_mlp_classifier_weights
from src.trainings.utils.config_enums_utils import get_activation_fn
from src.utils.config.config_reader import ConfigReader

from src.patchtst.models.PatchTST import Model as PatchTST


class AnomalyPredictionBertLightning(AnomalyPredictionModule):
    """
    Base Implementation of the AnomalyPredictionBert.
    :param model_config: Configuration file with model's hyperparameters.
    :param training_config: Configuration file with training's hyperparameters.
    """

    def __init__(self, model_config: ConfigReader, training_config: ConfigReader):
        # Encoder
        self.model_config_reader = model_config
        # Classifier parameters
        self.seq_len = model_config.get_param('data.input_length', v_type=int)
        self.channels = model_config.get_param('data.channels', v_type=int)
        self.pred_len = model_config.get_param('data.output_length', v_type=int)
        self.patch_size = model_config.get_param('data.patch_size', v_type=int)

        # Next, get the core model dimension parameters
        # These control the transformer's embedding space and positional encoding
        self.d_embed = model_config.get_param('embedding.d_embed', v_type=int)
        self.positional_encoding = model_config.get_param('embedding.positional_encoding', v_type=str,nullable=True)
        self.relative_position_embedding = model_config.get_param('embedding.relative_position_embedding',
                                                                  v_type=bool)

        # Get transformer block specific parameters
        self.causal_mask = model_config.get_param('transformer_block.causal_mask', v_type=bool)
        self.transformer_n_layer = model_config.get_param('transformer_block.transformer_n_layer', v_type=int)
        self.transformer_n_head = model_config.get_param('transformer_block.transformer_n_head', v_type=int)
        self.hidden_dim_rate = model_config.get_param('transformer_block.hidden_dim_rate', v_type=float)

        # Finally, get training-related parameters
        self.dropout = model_config.get_param('training.dropout', v_type=float)
        super().__init__(training_config, self.channels, self.pred_len, self.seq_len)

    def _setup_encoder(self) -> nn.Module:
        """
        PatchTST standard encoder.
        """
        return get_anomaly_prediction_bert(
            input_channels=self.channels,
            output_channels=self.channels,
            patch_size=self.patch_size,
            d_embed=self.d_embed,
            hidden_dim_rate=self.hidden_dim_rate,
            input_length=self.seq_len,
            output_length=self.pred_len,
            positional_encoding=self.positional_encoding,
            relative_position_embedding=self.relative_position_embedding,
            transformer_n_layer=self.transformer_n_layer,
            transformer_n_head=self.transformer_n_head,
            dropout=self.dropout,
            causal_mask=self.causal_mask
        )

    def _setup_classifier(self) -> nn.Module:
        """
        Fully Connected Network classifier.
        """
        # Initialize activations
        return nn.Identity()