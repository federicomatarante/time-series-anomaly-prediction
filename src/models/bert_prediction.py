from torch import nn

from src.models.anomaly_prediction_module import AnomalyPredictionModule
from src.models.modules.bert_anomaly_transformer.anomaly_prediction_transformer import get_anomaly_prediction_bert
from src.utils.config.config_reader import ConfigReader


class OutputProjection(nn.Module):
    def __init__(self, num_patches, d_embed, hidden_dim, pred_len, channels):
        super(OutputProjection, self).__init__()
        self.pred_len = pred_len
        self.channels = channels

        self.output_projection = nn.Sequential(
            nn.Linear(num_patches * d_embed, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len * channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply output_projection
        output = self.output_projection(x)

        # Reshape and transpose
        n_batch = x.shape[0]
        output = output.view(n_batch, self.pred_len, -1)  # (n_batch, pred_len, output_d_data)

        return output


class AnomalyPredictionBertLightning(AnomalyPredictionModule):
    """
    The following key features are:
        - Input (channels, seq_len) is divided into patches and reshaped in (num_patches, channels * patch_size )
        - The patches are projected in a new embedding space (num_patches, embed_dim)
        - The patches are processed in a transformer encoder
        - The output is processed by a MLP and classified with a classification layer
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
        self.positional_encoding = model_config.get_param('embedding.positional_encoding', v_type=str, nullable=True)
        self.relative_position_embedding = model_config.get_param('embedding.relative_position_embedding',
                                                                  v_type=bool)

        # Get transformer block specific parameters
        self.causal_mask = model_config.get_param('transformer_block.causal_mask', v_type=bool)
        self.transformer_n_layer = model_config.get_param('transformer_block.transformer_n_layer', v_type=int)
        self.transformer_n_head = model_config.get_param('transformer_block.transformer_n_head', v_type=int)
        self.hidden_dim_rate = model_config.get_param('transformer_block.hidden_dim_rate', v_type=float)

        # Finally, get training-related parameters
        self.dropout = model_config.get_param('training.dropout', v_type=float)
        super().__init__(training_config, self.channels, self.pred_len, self.seq_len, flatten=False)

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
        return nn.Identity()
        """hidden_dim = int(self.hidden_dim_rate * self.d_embed)
        num_patches = self.seq_len // self.patch_size
        return OutputProjection(num_patches, self.d_embed, hidden_dim, self.pred_len, self.channels)"""
