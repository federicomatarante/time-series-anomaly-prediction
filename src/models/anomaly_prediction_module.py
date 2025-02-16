from abc import abstractmethod, ABC
from typing import final

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset

from src.trainings.utils.anomaly_prediction_metrics import ExistenceOfAnomaly, DensityOfAnomalies, LeadTime, DiceScore, \
    ROCAUC
from src.trainings.utils.losses import WassersteinLoss, HuberLoss
from src.trainings.utils.optimizer_factory import OptimizerFactory
from src.trainings.utils.scheduler_factory import SchedulerFactory
from src.utils.config.config_reader import ConfigReader


class AnomalyDetectionModule(pl.LightningModule, ABC):
    """
    A PyTorch Lightning module for anomaly detection in time series data.

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
        delta = config_reader.get_param('training.loss_delta', v_type=float)
        reduction = config_reader.get_param("training.loss_reduction", v_type=str, nullable=True, domain={'mean', 'sum'})
        self.loss_fn = HuberLoss(delta=delta, reduction=reduction)
        # Scheduler and optimizer parameters
        self.scheduler_config = config_reader.sub_reader({'scheduler', 'scheduler.StepLR'})
        self.optimizer_config = config_reader.sub_reader({'optimizer', 'optimizer.Adam'})

        # Model modules
        self.model = self._setup_model()

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
        self.val_roc_auc = ROCAUC()  # This would use y_hat_scores instead of y_hat

        # Test metrics with the configured threshold
        test_threshold = config_reader.get_param('metrics.evaluate_metrics_threshold', v_type=float)
        self.test_existence = ExistenceOfAnomaly(threshold=test_threshold)
        self.test_density = DensityOfAnomalies()
        self.test_leadtime = LeadTime(threshold=test_threshold)
        self.test_dice = DiceScore(threshold=test_threshold)
        self.test_roc_auc = ROCAUC()  # This would use y_hat_scores instead of y_hat

    @abstractmethod
    def _setup_model(self) -> nn.Module:
        """
        Returns a forecaster of the time series, which accepts the following constraints:
            - Input Shape: [batch_size, channels, seq_len]
            - Output Shape: [batch_size, channels, pred_len]
        :return: the encoder of the time series.
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
        x = self.model(x)  # (batch_size, channels, pred_len )
        return x

    def _find_anomalies(self, prediction, actual_values):
        anomaly_scores = torch.abs(prediction - actual_values)

        # Calculate threshold using mean + std of errors # TODO remove stupid comments
        # Using reduction=None in mean/std to preserve dimensions for broadcasting
        threshold = (
                anomaly_scores.mean(dim=-1, keepdim=True) +
                2.0 * anomaly_scores.std(dim=-1, keepdim=True)
        )

        # Create boolean mask for anomalies
        anomaly_mask = (anomaly_scores > threshold).float()
        # TODO cimprove or customize?
        #TODO use anomaly score in metrics instead of mask'

        return anomaly_mask, anomaly_scores  # TODO document and change documentation

    def detect_anomalies(self, starting_window: torch.Tensor, prediction_window: torch.Tensor):
        """
        Detects anomalies by comparing the model's predictions with actual values.

        :param starting_window: Input tensor of shape (batch_size, channels, seq_len)
        :param prediction_window: Ground truth tensor of shape (batch_size, channels, pred_len)
        :return: Tuple containing (anomaly_scores, anomaly_mask, prediction)
            - anomaly_scores (torch.Tensor): Raw anomaly scores
            - anomaly_mask (torch.Tensor): Boolean mask indicating anomalies
            - prediction (torch.Tensor): Model's predictions
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """

        if len(starting_window.shape) != len(prediction_window.shape) != 3:
            raise ValueError(
                f"starting_window and prediction_window must be 3D (batch_size, channels, seq_len), got shape {starting_window.shape} and {prediction_window.shape}")

        # Check specific dimensions
        if starting_window.shape[1] != self.channels:
            raise ValueError(f"starting_window must have {self.channels} channels, got {starting_window.shape[1]}")
        if starting_window.shape[2] != self.seq_len:
            raise ValueError(
                f"starting_window must have sequence length {self.seq_len}, got {starting_window.shape[2]}")
        if prediction_window.shape[1] != self.channels:
            raise ValueError(f"prediction_window must have {self.channels} channels, got {prediction_window.shape[1]}")
        if prediction_window.shape[2] != self.pred_len:
            raise ValueError(
                f"prediction_window must have prediction length {self.pred_len}, got {prediction_window.shape[2]}")

        # Ensure batch sizes match
        if starting_window.shape[0] != prediction_window.shape[0]:
            raise ValueError(
                f"Batch sizes must match: starting_window has {starting_window.shape[0]}, prediction_window has {prediction_window.shape[0]}")

        with torch.no_grad():
            prediction = self(starting_window)  # Shape: (batch_size, channels, pred_len)

        anomaly_mask, anomaly_scores = self._find_anomalies(prediction, prediction_window)

        return anomaly_scores, anomaly_mask, prediction

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
        signal_in, signal_out, labels = dataset[0]
        signal_in_shape, signal_out_shape, labels_shape = signal_in.shape, signal_out.shape, labels.shape
        if len(signal_in_shape) != 2 or signal_in_shape[0] != self.channels or signal_in_shape[
            1] != self.seq_len:
            raise ValueError(
                f"Wrong input shape! Expected: (channels, seq_len) = ({self.channels},{self.seq_len}). Found: ({signal_in_shape})")

        if len(signal_out_shape) != 2 or signal_out_shape[0] != self.channels or signal_out_shape[
            1] != self.pred_len:
            raise ValueError(
                f"Wrong input shape! Expected: (channels, pred_len) = ({self.channels},{self.seq_len}). Found: ({signal_out_shape})")
        if len(labels_shape) != 2 or labels_shape[0] != self.channels or labels_shape[
            1] != self.pred_len:
            raise ValueError(
                f"Wrong input shape! Expected: (channels, pred_len) = ({self.channels},{self.seq_len}). Found: ({labels_shape})")

    @final
    def training_step(self, batch, batch_idx):
        x_in, x_out, _ = batch
        x_hat = self(x_in)
        loss = self.loss_fn(x_hat, x_out)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @final
    def validation_step(self, batch, batch_idx):
        x_in, x_out, y = batch
        x_hat = self(x_in)
        loss = self.loss_fn(x_hat, x_out)
        y_hat, y_hat_scores = self._find_anomalies(x_hat, x_out)

        # Log metrics
        self.val_existence(y_hat, y)
        self.val_density(y_hat, y)
        self.val_leadtime(y_hat, y)
        self.val_dice(y_hat, y)
        self.val_roc_auc(y_hat_scores, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_existence', self.val_existence, on_epoch=True)
        self.log('val_density', self.val_density, on_epoch=True)
        self.log('val_leadtime', self.val_leadtime, on_epoch=True)
        self.log('val_dice', self.val_dice, on_epoch=True)
        self.log('val_roc_auc', self.val_roc_auc, on_epoch=True)

        return loss

    @final
    def test_step(self, batch, batch_idx):
        x_in, x_out, y = batch
        x_hat = self(x_in)
        loss = self.loss_fn(x_hat, x_out)
        y_hat, y_hat_scores = self._find_anomalies(x_hat, x_out)

        # Use test metrics for testing
        self.test_existence(y_hat, y)
        self.test_density(y_hat, y)
        self.test_leadtime(y_hat, y)
        self.test_dice(y_hat, y)
        self.test_roc_auc(y_hat_scores, y)

        self.log('test_loss', loss, on_epoch=True)
        self.log('test_existence', self.test_existence, on_epoch=True)
        self.log('test_density', self.test_density, on_epoch=True)
        self.log('test_leadtime', self.test_leadtime, on_epoch=True)
        self.log('test_dice', self.test_dice, on_epoch=True)
        self.log('val_roc_auc', self.test_roc_auc, on_epoch=True)

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
