from abc import abstractmethod, ABC
from typing import final

import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset

from src.trainings.utils.anomaly_prediction_metrics import ExistenceOfAnomaly, DensityOfAnomalies, LeadTime, DiceScore
from src.trainings.utils.losses import WassersteinLoss
from src.trainings.utils.optimizer_factory import OptimizerFactory
from src.trainings.utils.scheduler_factory import SchedulerFactory
from src.utils.config.config_reader import ConfigReader


class AnomalyPredictionModule(pl.LightningModule, ABC):
    """
    A PyTorch Lightning module for anomaly prediction in time series data.

    This abstract base class implements the core functionality for training, validating,
    and testing models that predict anomalies in multivariate time series. The module
    follows an encoder-classifier architecture where the encoder processes the input
    time series and the classifier predicts anomaly scores.

    Key Features:
        - Configurable loss function with optional scaling
        - Customizable optimizer and learning rate scheduler
        - Multiple evaluation metrics including existence, density, lead time, and Dice score
        - Separate thresholds for validation and test metrics
        - Automatic model checkpoint handling


    :param config_reader (ConfigReader): Configuration object containing model parameters
    :param channels (int): Number of input/output channels in the time series
    :param pred_len (int): Length of the prediction window
    :param seq_len (int): Length of the input sequence


    Note:
        Subclasses must implement:
            - _setup_encoder(): Creates the time series encoder
            - _setup_classifier(): Creates the anomaly classifier
        Optional:
            - on_save(): handles the checkpoint saving of the model.
            - on_load(): handles the checkpoint loading of the model

    Input Shape:
        - Input: (batch_size, channels, seq_len)
        - Output: (batch_size, channels, pred_len)
    """

    def __init__(self, config_reader: ConfigReader, channels: int, pred_len: int, seq_len: int):
        super().__init__()
        self.scheduler_state, self.optimizer_state = None, None
        self.save_hyperparameters()
        # Loss Function
        loss_scaling_factor = config_reader.get_param('training.loss_scaling_factor', v_type=bool)
        self.loss_fn = WassersteinLoss(apply_scaling_factor=loss_scaling_factor)
        # Scheduler and optimizer parameters
        self.scheduler_config = config_reader.sub_reader({'scheduler', 'scheduler.StepLR'})
        self.optimizer_config = config_reader.sub_reader({'optimizer', 'optimizer.Adam'})

        # Model modules
        self.encoder = self._setup_encoder()
        self.classifier = self._setup_classifier()

        # Input/Output parameters
        self.channels = channels
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Validation metrics with configured threshold
        val_threshold = config_reader.get_param('metrics.val_metrics_threshold', v_type=float)
        self.val_existence = ExistenceOfAnomaly(threshold=val_threshold)
        self.val_density = DensityOfAnomalies()
        self.val_leadtime = LeadTime(threshold=val_threshold)
        self.val_dice = DiceScore(threshold=val_threshold)

        # Test metrics with the configured threshold
        test_threshold = config_reader.get_param('metrics.evaluate_metrics_threshold', v_type=float)
        self.test_existence = ExistenceOfAnomaly(threshold=test_threshold)
        self.test_density = DensityOfAnomalies()
        self.test_leadtime = LeadTime(threshold=test_threshold)
        self.test_dice = DiceScore(threshold=test_threshold)

    @abstractmethod
    def _setup_encoder(self) -> nn.Module:
        """
        Returns an encoder of the time series, which accepts the following constraints:
            - Input Shape: [batch_size, channels, seq_len]
            - Output Shape: [batch_size, channels, pred_len]
        :return: the encoder of the time series.
        """
        pass

    @abstractmethod
    def _setup_classifier(self) -> nn.Module:
        """
        Returns a classifier of the time series, which accepts the following constraints:
            - Input Shape: [batch_size, channels * pred_len]
            - Output Shape: [batch_size, channels * pred_len]
        :return: the classifier of the time series.
        """
        pass

    @final
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
        x = self.encoder(x)  # (batch_size, channels, pred_len )
        x = x.flatten(-2, -1)  # (batch_size, channels * pred_len)
        x = self.classifier(x)  # (batch_size, pred_len * channels)
        x = x.view(-1, self.channels, self.pred_len, )  # (batch_size, channels, pred_len)
        return x

    @final
    def check_compatibility(self, dataset: Dataset):
        """
        Checks the compatibility with the model and the dataset given.
        The dataset return type must contain:
            - Signal: tensor of shape (batch_size, channels, seq_len)
            - Labels: tensor of shape (batch_size, channels, pred_len)
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")
        signal, output = dataset[0]
        signal_shape, output_shape = signal.shape, output.shape
        if len(signal_shape) != 2 or signal_shape[0] != self.channels or signal_shape[1] != self.seq_len:
            raise ValueError(
                f"Wrong input shape! Expected: (channels, seq_len) = ({self.channels},{self.seq_len}). Found: ({signal_shape})")
        if len(output_shape) != 2 or output_shape[0] != self.channels or output_shape[1] != self.pred_len:
            raise ValueError(
                f"Wrong output shape! Expected: (channels,pred_len) = ({self.channels},{self.pred_len}). Found: ({output_shape})")

    @final
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @final
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

    @final
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

    @final
    def configure_optimizers(self):
        """
        Configures the optimizers and the LR scheduler of the class, reading parameters from the config reader.
        """
        # Get optimizer configuration
        optimizer_factory = OptimizerFactory(self.optimizer_config)
        optimizer = optimizer_factory.get_optimizer('adam', self.parameters())
        if self.optimizer_state:
            optimizer.load_state_dict(self.optimizer_state)
        # Get scheduler configuration
        scheduler_factory = SchedulerFactory(self.scheduler_config)
        scheduler = scheduler_factory.get_scheduler('steplr', optimizer)
        if self.scheduler_state:
            scheduler.load_state_dict(self.scheduler_state)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": self.scheduler_config.get_param("scheduler.frequency", v_type=int),
                "interval": self.scheduler_config.get_param("scheduler.interval", v_type=str, domain={"step", "epoch"})
            }
        }

    def on_save(self, checkpoint: dict):
        """
        Saves additional parameters and configurations in the checkpoint during model saving.
        This method is automatically called by PyTorch Lightning during saving.

        :param checkpoint: Checkpoint dictionary where additional configurations will be saved.
                          Already contains standard PyTorch Lightning parameters and optimizers/scheduler states.
        """
        pass

    @final
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

        checkpoint['optimizer_config'] = self.optimizer_config.config_data
        checkpoint['optimizer_state'] = self.optimizers().state_dict()
        checkpoint['scheduler_state'] = self.lr_schedulers().state_dict()
        self.on_save(checkpoint)

    def on_load(self, checkpoint: dict):
        """
        Loads and restores parameters and configurations from the checkpoint during model loading.
        This method is automatically called by PyTorch Lightning during loading.

        :param checkpoint: Checkpoint dictionary containing configurations to be loaded.
                          Includes both standard PyTorch Lightning parameters and custom ones
                          saved in on_save_checkpoint.
                          Model, scheduler and optimizer parameters are already loaded.
        """

    @final
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
        if 'optimizer_state' in checkpoint:
            self.optimizer_state = checkpoint['optimizer_state']

        if 'scheduler_state' in checkpoint:
            self.scheduler_state = checkpoint['scheduler_state']
        self.on_load(checkpoint)
