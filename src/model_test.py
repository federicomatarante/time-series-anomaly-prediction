import torch

from src.patchtst.models.PatchTST import Model
from src.utils.config.config_reader import ConfigReader

# Define the configuration dictionary
config_data = {
    'model': {
        # Model architecture
        'enc_in': 7,  # number of input features
        'seq_len': 96,  # input sequence length
        'pred_len': 24,  # prediction sequence length

        # Transformer parameters
        'e_layers': 2,  # number of encoder layers
        'n_heads': 4,  # number of attention heads
        'd_model': 128,  # dimension of model
        'd_ff': 256,  # dimension of feed forward network

        # Dropout parameters
        'dropout': 0.2,  # dropout rate
        'fc_dropout': 0.2,  # fully connected dropout rate
        'head_dropout': 0.0,  # head dropout rate

        # Patch parameters
        'patch_len': 16,  # length of patch
        'stride': 8,  # stride of patch
        'padding_patch': True,  # whether to pad patches

        # Other parameters
        'individual': False,  # whether to use individual features
        'revin': True,  # whether to use reversible instance normalization
        'affine': True,  # whether to use affine transformation
        'subtract_last': False,  # whether to subtract last value

        # Decomposition parameters
        'decomposition': True,  # whether to use decomposition
        'kernel_size': 25  # kernel size for decomposition
    }
}

# Create ConfigReader instance
config_reader = ConfigReader(config_data)

# Initialize the model
model = Model(
    config_reader=config_reader,
    max_seq_len=1024,  # optional parameters can still be passed directly
    d_k=None,
    d_v=None,
    norm='BatchNorm',
    attn_dropout=0.0,
    act="gelu",
    pre_norm=False,
    store_attn=False,
    pe='zeros',
    learn_pe=True,
    head_type='flatten',
    verbose=True
)

# Example usage with dummy data
batch_size = 32
seq_length = config_reader.get_param('model.seq_len', v_type=int)
n_features = config_reader.get_param('model.enc_in', v_type=int)

# Create dummy input data [Batch, Input length, Channel]
x = torch.randn(batch_size, seq_length, n_features)

# Forward pass
output = model(x)
print(output)