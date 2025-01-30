from typing import Tuple

import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset

from src.patchtst.models.PatchTST import Model as PatchTST
from src.trainings.utils.anomaly_prediction_metrics import ExistenceOfAnomaly, DensityOfAnomalies, LeadTime, DiceScore
from src.trainings.utils.config_enums_utils import get_activation_fn
from src.trainings.utils.losses import WassersteinLoss
from src.trainings.utils.optimizer_factory import OptimizerFactory
from src.trainings.utils.scheduler_factory import SchedulerFactory
from src.utils.config.config_reader import ConfigReader


class PatchTSTLightning(pl.LightningModule):
    """
    PyTorch Lightning implementation of PatchTST model for time series anomaly prediction.

    :param model_config_reader: Configuration reader containing model architecture parameters
    :param training_config_reader: Configuration reader containing training parameters

    Example:
         model_config = INIConfigReader("model_config.yaml")
         training_config = INIConfigReader("training_config.yaml")
         model = PatchTSTLightning(model_config, training_config)
    """

    def __init__(self,
                 model_config_reader: ConfigReader,
                 training_config_reader: ConfigReader):
        super().__init__()
        self.encoder = PatchTST(model_config_reader)
        self.loss_fn = WassersteinLoss(
            apply_scaling_factor=training_config_reader.get_param('training.loss_scaling_factor', v_type=bool))
        # Scheduler parameters
        self.scheduler_monitor = training_config_reader.get_param('scheduler.monitor', v_type=str)
        self.scheduler_frequency = training_config_reader.get_param('scheduler.frequency', v_type=int)
        self.scheduler_type = training_config_reader.get_param('scheduler.type', v_type=str)
        self.scheduler_config = ConfigReader(
            {'scheduler': training_config_reader.get_section('scheduler'),
             'scheduler.ReduceLROnPlateau': training_config_reader.get_section('scheduler.ReduceLROnPlateau'),
             'scheduler.StepLR': training_config_reader.get_section('scheduler.StepLR'),
             'scheduler.CosineAnnealingLR': training_config_reader.get_section('scheduler.CosineAnnealingLR'),
             'scheduler.ExponentialLR': training_config_reader.get_section('scheduler.ExponentialLR'),
             'scheduler.CosineAnnealingWarmRestarts': training_config_reader.get_section(
                 'scheduler.CosineAnnealingWarmRestarts'),
             'scheduler.OneCycleLR': training_config_reader.get_section('scheduler.OneCycleLR'),

             }
        )
        # Optimizer parameters
        self.optimizer_type = training_config_reader.get_param('optimizer.type', v_type=str)
        self.optimizer_config = ConfigReader(
            {
                'optimizer': training_config_reader.get_section('optimizer'),
                'optimizer.Adam': training_config_reader.get_section('optimizer.Adam'),
                'optimizer.Adamw': training_config_reader.get_section('optimizer.Adamw'),
                'optimizer.SGD': training_config_reader.get_section('optimizer.SGD'),
                'optimizer.RMSprop': training_config_reader.get_section('optimizer.RMSprop'),
                'optimizer.Adadelta': training_config_reader.get_section('optimizer.Adadelta'),
                'optimizer.Adagrad': training_config_reader.get_section('optimizer.Adagrad'),
                'optimizer.RAdam': training_config_reader.get_section('optimizer.RAdam'),
            }
        )

        # Validation metrics with configured threshold
        val_threshold = training_config_reader.get_param('metrics.val_metrics_threshold', v_type=float)
        self.val_existence = ExistenceOfAnomaly(threshold=val_threshold)
        self.val_density = DensityOfAnomalies()
        self.val_leadtime = LeadTime(threshold=val_threshold)
        self.val_dice = DiceScore(threshold=val_threshold)

        # Test metrics with the configured threshold4
        test_threshold = training_config_reader.get_param('metrics.evaluate_metrics_threshold', v_type=float)
        self.test_existence = ExistenceOfAnomaly(threshold=test_threshold)
        self.test_density = DensityOfAnomalies()
        self.test_leadtime = LeadTime(threshold=test_threshold)
        self.test_dice = DiceScore(threshold=test_threshold)

        # Setup classifier
        self.context_window = model_config_reader.get_param('seq.len', v_type=int)
        self.layers_sizes = model_config_reader.get_collection('classifier.layers_sizes', v_type=int,
                                                               collection_type=tuple)
        self.hidden_act = model_config_reader.get_param('classifier.activation.hidden', v_type=str)
        self.output_act = model_config_reader.get_param('classifier.activation.output', v_type=str)
        self.dropout = model_config_reader.get_param('dropout.classifier', v_type=float)
        self.channels = model_config_reader.get_param('data.enc_in', v_type=int)
        self.pred_len = model_config_reader.get_param('pred.len', v_type=int)
        self._setup_classifier(self.layers_sizes, self.hidden_act, self.output_act, self.dropout)
        # Save hyperparameters

    def _setup_classifier(self,
                          layers_sizes: Tuple[int, ...],
                          hidden_act: str,
                          output_act: str,
                          dropout_rate: int,
                          ):
        """
        Sets up the classifier network for anomaly prediction.

        :param layers_sizes: Sizes of hidden layers in the classifier
        :param hidden_act: Activation function name for hidden layers
        :param output_act: Activation function name for output layer
        :param dropout_rate: Dropout probability
        """
        # Initialize activations
        hidden_activation = get_activation_fn(hidden_act)
        output_activation = get_activation_fn(output_act)
        dropout = nn.Dropout(p=dropout_rate)
        layers = []
        encoder_output_size = self.pred_len * self.channels
        for size in layers_sizes:
            layers.extend([
                nn.Linear(encoder_output_size, size),
                hidden_activation,
                dropout
            ])
            encoder_output_size = size

        layers.extend([
            nn.Linear(encoder_output_size, self.pred_len * self.channels),
            output_activation
        ])

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input tensor of shape (batch_size, channels, seq_len)
        :return: Predicted anomaly scores of shape (batch_size, channels, pred_len)

        Example:
             x = torch.randn(32, 100, 10)  # batch_size=32, seq_len=100, features=10
             output = model(x)
             print(output.shape)  # torch.Size([32, window_size])
        """
        # x: (batch_size, channels, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, channels)
        x = self.encoder(x)  # (batch_size, pred_len, channels)
        x = x.transpose(1,2) # (batch_size, channels, pred_len)
        x = x.flatten(-2, -1) # (batch_size, pred_len * channels)
        x = self.classifier(x)  # (batch_size, pred_len * channels)
        x = x.view(-1, self.channels, self.pred_len,)  # (batch_size, channels, pred_len)
        return x

    def check_compatibility(self, dataset: Dataset):
        """
        Checks the compatibility with the model and the dataset given.
        The dataset return type must contain:
            - Signal: tensor of shape (batch_size, channels, context_window)
            - Labels: tensor of shape (batch_size, channels, pred_len)
        :param dataset:
        :return:
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")
        signal, output = dataset[0]
        signal_shape, output_shape = signal.shape, output.shape
        if len(signal_shape) != 2 or signal_shape[0] != self.channels or signal_shape[1] != self.context_window:
            raise ValueError(
                f"Wrong input shape! Expected: (channels, context_window) = ({self.channels},{self.context_window}). Found: ({signal_shape})")
        if len(output_shape) != 2 or output_shape[0] != self.channels or output_shape[1] != self.pred_len:
            raise ValueError(
                f"Wrong output shape! Expected: (channels,window_size) = ({self.channels},{self.pred_len}). Found: ({output_shape})")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log metrics
        self.val_existence(y_hat, y)
        self.val_density(y_hat, y)
        self.val_leadtime(y_hat, y)
        self.val_dice(y_hat, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_existence', self.val_existence, on_epoch=True)
        self.log('val_density', self.val_density, on_epoch=True)
        self.log('val_leadtime', self.val_leadtime, on_epoch=True)
        self.log('val_dice', self.val_dice, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Use test metrics for testing
        self.test_existence(y_hat, y)
        self.test_density(y_hat, y)
        self.test_leadtime(y_hat, y)
        self.test_dice(y_hat, y)

        self.log('test_loss', loss, on_epoch=True)
        self.log('test_existence', self.test_existence, on_epoch=True)
        self.log('test_density', self.test_density, on_epoch=True)
        self.log('test_leadtime', self.test_leadtime, on_epoch=True)
        self.log('test_dice', self.test_dice, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Get optimizer configuration
        optimizer_factory = OptimizerFactory(self.optimizer_config)
        optimizer = optimizer_factory.get_optimizer(self.optimizer_type, self.parameters())

        # Get scheduler configuration
        scheduler_factory = SchedulerFactory(self.scheduler_config)
        scheduler_config = {
            "scheduler": scheduler_factory.get_scheduler(self.scheduler_type, optimizer),
            "monitor": self.scheduler_monitor,
            "frequency": self.scheduler_frequency
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }

    def on_save_checkpoint(self, checkpoint: dict):
        """
        Saves additional parameters and configurations in the checkpoint during model saving.
        This method is automatically called by PyTorch Lightning during saving.

        :param checkpoint: Checkpoint dictionary where additional configurations will be saved.
                          Already contains standard PyTorch Lightning parameters

        :Note:
            - Parameters saved here will be available during loading via on_load_checkpoint
            - The method extends the checkpoint dictionary with custom model configurations
            - Useful for saving parameters that cannot be handled by save_hyperparameters()
        """
        # Save model configurations
        checkpoint['model_config'] = {
            # Architecture parameters
            'layers_sizes': self.layers_sizes,
            'context_window': self.context_window,
            'hidden_act': self.hidden_act,
            'output_act': self.output_act,
            'dropout': self.dropout,
            'channels': self.channels,
            'pred_len': self.pred_len,

            # Other model-specific parameters
            # 'threshold': self.train_existence.threshold,
            'val_threshold': self.val_existence.threshold,
            'test_threshold': self.test_existence.threshold
        }

        # Save training configurations
        checkpoint['training_config'] = {
            # Optimizer configuration
            'optimizer_type': self.optimizer_type,
            'optimizer_config': {
                'optimizer' + name: self.optimizer_config.get_section(f'optimizer{name}')
                for name in ['', '.Adam', '.Adamw', '.SGD', '.RMSprop', '.Adadelta', '.Adagrad', '.RAdam']
            },

            # Scheduler configuration
            'scheduler_type': self.scheduler_type,
            'scheduler_monitor': self.scheduler_monitor,
            'scheduler_frequency': self.scheduler_frequency,
            'scheduler_config': {
                "scheduler" + name: self.scheduler_config.get_section(f'scheduler{name}')
                for name in ['', '.ReduceLROnPlateau', '.StepLR', '.CosineAnnealingLR',
                             '.ExponentialLR', '.CosineAnnealingWarmRestarts', '.OneCycleLR']
            }
        }

    def on_load_checkpoint(self, checkpoint: dict):
        """
        Loads and restores parameters and configurations from the checkpoint during model loading.
        This method is automatically called by PyTorch Lightning during loading.

        :param checkpoint: Checkpoint dictionary containing configurations to be loaded.
                          Includes both standard PyTorch Lightning parameters and custom ones
                          saved in on_save_checkpoint
        :raises KeyError: If required keys are missing in the checkpoint
        :raises ValueError: If loaded values are invalid

        :Note:
            - Restores model state using parameters saved in on_save_checkpoint
            - Verifies the presence of necessary keys to avoid errors
            - Useful for loading parameters that cannot be handled by save_hyperparameters()
        """
        try:
            # Restore model configuration
            model_config = checkpoint.get('model_config')
            if model_config:
                # Update encoder parameters
                for key, value in model_config.items():
                    if hasattr(self.encoder, key):
                        setattr(self.encoder, key, value)

                # Update metrics thresholds
                if 'threshold' in model_config:
                    self.train_existence.threshold = model_config['threshold']
                if 'val_threshold' in model_config:
                    self.val_existence.threshold = model_config['val_threshold']
                if 'test_threshold' in model_config:
                    self.test_existence.threshold = model_config['test_threshold']

            # Restore training configuration
            training_config = checkpoint.get('training_config')
            if training_config:
                # Restore optimizer configurations
                self.optimizer_type = training_config.get('optimizer_type', self.optimizer_type)
                optimizer_config = training_config.get('optimizer_config', {})
                self.optimizer_config = ConfigReader(optimizer_config)

                # Restore scheduler configurations
                self.scheduler_type = training_config.get('scheduler_type', self.scheduler_type)
                self.scheduler_monitor = training_config.get('scheduler_monitor', self.scheduler_monitor)
                self.scheduler_frequency = training_config.get('scheduler_frequency', self.scheduler_frequency)
                scheduler_config = training_config.get('scheduler_config', {})
                self.scheduler_config = ConfigReader(scheduler_config)

        except Exception as e:
            raise ValueError(f"Error during checkpoint loading: {str(e)}")
