import os
import sys

from src.models.modules.bert_anomaly_transformer.transformer_encoder import get_transformer_encoder

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn


class AnomalyPredictionBert(nn.Module):
    def __init__(self, linear_embedding, transformer_encoder, output_projection, d_embed, patch_size, max_seq_len,
                 target_len):
        """
        Args come prima, ma ora output_projection è progettato per generare sequenze
        di lunghezza variabile a partire dall'embedding completo
        """
        super(AnomalyPredictionBert, self).__init__()
        self.linear_embedding = linear_embedding
        self.transformer_encoder = transformer_encoder
        self.output_projection = output_projection

        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.d_embed = d_embed
        self.target_len = target_len

    def forward(self, x):
        """
        Args:
            x: (n_batch,  d_data,n_token) = (_, max_seq_len*patch_size, _)
        """
        n_batch = x.shape[0]
        x = x.transpose(-1, -2)  # B,C,S

        # Embedding dell'input come prima
        embedded_out = x.contiguous().view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch,
                                                                                                self.max_seq_len, -1)
        # x: [batch_size,  num_patches, channels * patch_size]
        embedded_out = self.linear_embedding(embedded_out)
        # x: [batch_size,  num_patches, d_embed]

        transformer_out = self.transformer_encoder(embedded_out)  # (n_batch, num_patches, d_embed)

        full_context = transformer_out.reshape(n_batch, -1)  # (n_batch, num_patches * d_embed)

        # Proiezione nell'output di lunghezza desiderata
        output = self.output_projection(full_context)  # (n_batch, target_len * output_d_data)
        output = output.view(n_batch, self.target_len, -1)  # (n_batch, target_len, output_d_data)
        output = output.transpose(-1, -2)

        return output


def get_anomaly_prediction_bert(
        input_channels,
        output_channels,
        patch_size,
        d_embed=512,
        hidden_dim_rate=4.,
        input_length=512,
        output_length=1024,
        positional_encoding=None,
        relative_position_embedding=True,
        transformer_n_layer=12,
        transformer_n_head=8,
        dropout=0.1,
        causal_mask=False
):
    """
    Versione modificata che può generare output di lunghezza variabile
    utilizzando l'intero contesto dell'input
    """
    hidden_dim = int(hidden_dim_rate * d_embed)
    num_patches = input_length // patch_size
    linear_embedding = nn.Linear(input_channels * patch_size, d_embed)
    transformer_encoder = get_transformer_encoder(
        d_embed=d_embed,
        positional_encoding=positional_encoding,
        relative_position_embedding=relative_position_embedding,
        n_layer=transformer_n_layer,
        n_head=transformer_n_head,
        d_ff=hidden_dim,
        max_seq_len=num_patches,
        dropout=dropout,
        causal_mask=causal_mask
    )

    # Output projection che prende l'intero contesto e genera una sequenza
    output_projection = nn.Sequential(
        nn.Linear(num_patches * d_embed, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_length * output_channels),
        nn.Sigmoid()
    )

    # Inizializzazione dei pesi
    nn.init.xavier_uniform_(linear_embedding.weight)
    nn.init.zeros_(linear_embedding.bias)
    for layer in output_projection:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    return AnomalyPredictionBert(
        linear_embedding,
        transformer_encoder,
        output_projection,
        d_embed,
        patch_size,
        num_patches,
        output_length
    )
