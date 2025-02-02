from abc import ABC, abstractmethod

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from src.utils.config.config_reader import ConfigReader


class SchedulerBuilder(ABC):
    @abstractmethod
    def build(self, optimizer: Optimizer) -> LRScheduler:
        pass


class ReduceLROnPlateauBuilder(SchedulerBuilder):
    """
    Reduces learning rate when a metric has stopped improving.

    Config Parameters:
    - scheduler.ReduceLROnPlateau.mode: str - 'min' for loss/error metrics, 'max' for accuracy metrics
    - scheduler.ReduceLROnPlateau.factor: float - Factor to decrease learning rate by (e.g. 0.1 = reduce by 10x)
    - scheduler.ReduceLROnPlateau.patience: int - Number of epochs with no improvement after which LR will be reduced
    - scheduler.ReduceLROnPlateau.verbose: bool - If True, prints message when LR is reduced
    - scheduler.ReduceLROnPlateau.threshold: float - Minimum change in monitored quantity to be considered as improvement
    - scheduler.ReduceLROnPlateau.threshold_mode: str - 'rel' for relative threshold, 'abs' for absolute threshold
    - scheduler.ReduceLROnPlateau.cooldown: int - Number of epochs to wait before resuming normal operation after LR has been reduced
    - scheduler.ReduceLROnPlateau.min_lr: float - Lower bound on the learning rate
    - scheduler.ReduceLROnPlateau.eps: float - Minimal decay applied to lr
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, optimizer: Optimizer) -> ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=self.config.get_param('scheduler.ReduceLROnPlateau.mode', v_type=str),
            factor=self.config.get_param('scheduler.ReduceLROnPlateau.factor', v_type=float),
            patience=self.config.get_param('scheduler.ReduceLROnPlateau.patience', v_type=int),
            verbose=self.config.get_param('scheduler.ReduceLROnPlateau.verbose', v_type=bool),
            threshold=self.config.get_param('scheduler.ReduceLROnPlateau.threshold', v_type=float),
            threshold_mode=self.config.get_param('scheduler.ReduceLROnPlateau.threshold_mode', v_type=str),
            cooldown=self.config.get_param('scheduler.ReduceLROnPlateau.cooldown', v_type=int),
            min_lr=self.config.get_param('scheduler.ReduceLROnPlateau.min_lr', v_type=float),
            eps=self.config.get_param('scheduler.ReduceLROnPlateau.eps', v_type=float)
        )


class StepLRBuilder(SchedulerBuilder):
    """
    Decays learning rate by gamma every step_size epochs.

    Config Parameters:
    - scheduler.StepLR.step_size: int - Period of learning rate decay (epochs)
    - scheduler.StepLR.gamma: float - Multiplicative factor of learning rate decay
    - scheduler.StepLR.verbose: bool - If True, prints message when LR is updated
    - scheduler.StepLR.last_epoch: int - The index of last epoch, -1 for start of training
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, optimizer: Optimizer) -> LRScheduler:
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.config.get_param('scheduler.StepLR.step_size', v_type=int),
            gamma=self.config.get_param('scheduler.StepLR.gamma', v_type=float),
            verbose=self.config.get_param('scheduler.StepLR.verbose', v_type=bool),
            last_epoch=self.config.get_param('scheduler.StepLR.last_epoch', v_type=int)
        )


class CosineAnnealingLRBuilder(SchedulerBuilder):
    """
    Decays learning rate with cosine annealing schedule.

    Config Parameters:
    - scheduler.CosineAnnealingLR.max_iter: int - Maximum number of iterations
    - scheduler.CosineAnnealingLR.min_lr: float - Minimum learning rate
    - scheduler.CosineAnnealingLR.last_epoch: int - The index of last epoch
    - scheduler.CosineAnnealingLR.verbose: bool - If True, prints message when LR is updated
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, optimizer: Optimizer) -> LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.config.get_param('scheduler.CosineAnnealingLR.max_iter', v_type=int),
            eta_min=self.config.get_param('scheduler.CosineAnnealingLR.min_lr', v_type=float),
            last_epoch=self.config.get_param('scheduler.CosineAnnealingLR.last_epoch', v_type=int),
            verbose=self.config.get_param('scheduler.CosineAnnealingLR.verbose', v_type=bool)
        )


class ExponentialLRBuilder(SchedulerBuilder):
    """
    Exponentially decays learning rate by gamma every epoch.

    Config Parameters:
    - scheduler.ExponentialLR.gamma: float - Multiplicative factor of learning rate decay
    - scheduler.ExponentialLR.last_epoch: int - The index of last epoch
    - scheduler.ExponentialLR.verbose: bool - If True, prints message when LR is updated
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, optimizer: Optimizer) -> LRScheduler:
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.config.get_param('scheduler.ExponentialLR.gamma', v_type=float),
            last_epoch=self.config.get_param('scheduler.ExponentialLR.last_epoch', v_type=int),
            verbose=self.config.get_param('scheduler.ExponentialLR.verbose', v_type=bool)
        )


class CosineAnnealingWarmRestartsBuilder(SchedulerBuilder):
    """
    Cosine annealing with warm restarts - learning rate cycles with increasing period.

    Config Parameters:
    - scheduler.CosineAnnealingWarmRestarts.T_0: int - Number of iterations for the first restart
    - scheduler.CosineAnnealingWarmRestarts.T_mult: int - Factor increasing T_i after a restart
    - scheduler.CosineAnnealingWarmRestarts.min_lr: float - Minimum learning rate
    - scheduler.CosineAnnealingWarmRestarts.last_epoch: int - The index of last epoch
    - scheduler.CosineAnnealingWarmRestarts.verbose: bool - If True, prints message when LR is updated
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, optimizer: Optimizer) -> LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.config.get_param('scheduler.CosineAnnealingWarmRestarts.T_0', v_type=int),
            T_mult=self.config.get_param('scheduler.CosineAnnealingWarmRestarts.T_mult', v_type=int),
            eta_min=self.config.get_param('scheduler.CosineAnnealingWarmRestarts.min_lr', v_type=float),
            last_epoch=self.config.get_param('scheduler.CosineAnnealingWarmRestarts.last_epoch', v_type=int),
            verbose=self.config.get_param('scheduler.CosineAnnealingWarmRestarts.verbose', v_type=bool)
        )


class OneCycleLRBuilder(SchedulerBuilder):
    """
    One cycle learning rate policy with initial warmup and final cooldown.

    Config Parameters:
        - scheduler.OneCycleLR.max_lr: float - Upper learning rate boundaries in the cycle
        - scheduler.OneCycleLR.total_steps: int - Total number of training steps
        - scheduler.OneCycleLR.pct_start: float - Percentage of cycle spent increasing LR (0.0-1.0)
        - scheduler.OneCycleLR.anneal_strategy: str - 'cos' for cosine annealing, 'linear' for linear annealing
        - scheduler.OneCycleLR.cycle_momentum: bool - If True, momentum cycling is enabled
        - scheduler.OneCycleLR.base_momentum: float - Lower momentum boundaries
        - scheduler.OneCycleLR.max_momentum: float - Upper momentum boundaries
        - scheduler.OneCycleLR.div_factor: float - Initial LR division factor
        - scheduler.OneCycleLR.final_div_factor: float - Final LR division factor
        - scheduler.OneCycleLR.last_epoch: int - The index of last epoch
        - scheduler.OneCycleLR.verbose: bool - If True, prints message when LR is updated
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, optimizer: Optimizer) -> LRScheduler:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.config.get_param('scheduler.OneCycleLR.max_lr', v_type=float),
            total_steps=self.config.get_param('scheduler.OneCycleLR.total_steps', v_type=int),
            pct_start=self.config.get_param('scheduler.OneCycleLR.pct_start', v_type=float),
            anneal_strategy=self.config.get_param('scheduler.OneCycleLR.anneal_strategy', v_type=str),
            cycle_momentum=self.config.get_param('scheduler.OneCycleLR.cycle_momentum', v_type=bool),
            base_momentum=self.config.get_param('scheduler.OneCycleLR.base_momentum', v_type=float),
            max_momentum=self.config.get_param('scheduler.OneCycleLR.max_momentum', v_type=float),
            div_factor=self.config.get_param('scheduler.OneCycleLR.div_factor', v_type=float),
            final_div_factor=self.config.get_param('scheduler.OneCycleLR.final_div_factor', v_type=float),
            last_epoch=self.config.get_param('scheduler.OneCycleLR.last_epoch', v_type=int),
            verbose=self.config.get_param('scheduler.OneCycleLR.verbose', v_type=bool)
        )


class SchedulerFactory:
    """
    Factory class for creating PyTorch learning rate schedulers based on configuration.

    Supports multiple scheduler types including:
    - ReduceLROnPlateau
    - StepLR
    - CosineAnnealingLR
    - ExponentialLR
    - CosineAnnealingWarmRestarts
    - OneCycleLR
    Required configuration parameters for each scheduler type:

    ReduceLROnPlateau:
        - scheduler.ReduceLROnPlateau.mode: str - 'min' or 'max'
        - scheduler.ReduceLROnPlateau.factor: float - Factor to decrease learning rate
        - scheduler.ReduceLROnPlateau.patience: int - Number of epochs with no improvement
        - scheduler.ReduceLROnPlateau.verbose: bool - Print message on updates
        - scheduler.ReduceLROnPlateau.threshold: float - Improvement threshold
        - scheduler.ReduceLROnPlateau.threshold_mode: str - 'rel' or 'abs'
        - scheduler.ReduceLROnPlateau.cooldown: int - Epochs before resuming normal operation
        - scheduler.ReduceLROnPlateau.min_lr: float - Lower bound on learning rate
        - scheduler.ReduceLROnPlateau.eps: float - Minimal decay applied

    StepLR:
        - scheduler.StepLR.step_size: int - Period of learning rate decay
        - scheduler.StepLR.gamma: float - Multiplicative factor of learning rate decay
        - scheduler.StepLR.verbose: bool - Print message on updates
        - scheduler.StepLR.last_epoch: int - Index of last epoch

    CosineAnnealingLR:
        - scheduler.CosineAnnealingLR.max_iter: int - Maximum iterations
        - scheduler.CosineAnnealingLR.min_lr: float - Minimum learning rate
        - scheduler.CosineAnnealingLR.last_epoch: int - Index of last epoch
        - scheduler.CosineAnnealingLR.verbose: bool - Print message on updates

    ExponentialLR:
        - scheduler.ExponentialLR.gamma: float - Multiplicative factor of learning rate decay
        - scheduler.ExponentialLR.last_epoch: int - Index of last epoch
        - scheduler.ExponentialLR.verbose: bool - Print message on updates

    CosineAnnealingWarmRestarts:
        - scheduler.CosineAnnealingWarmRestarts.T_0: int - First restart iterations
        - scheduler.CosineAnnealingWarmRestarts.T_mult: int - Factor to increase T_i after restart
        - scheduler.CosineAnnealingWarmRestarts.min_lr: float - Minimum learning rate
        - scheduler.CosineAnnealingWarmRestarts.last_epoch: int - Index of last epoch
        - scheduler.CosineAnnealingWarmRestarts.verbose: bool - Print message on updates

    OneCycleLR:
        - scheduler.OneCycleLR.max_lr: float - Upper learning rate boundaries
        - scheduler.OneCycleLR.total_steps: int - Total number of training steps
        - scheduler.OneCycleLR.pct_start: float - Percentage of cycle for increasing LR
        - scheduler.OneCycleLR.anneal_strategy: str - 'cos' or 'linear'
        - scheduler.OneCycleLR.cycle_momentum: bool - Enable momentum cycling
        - scheduler.OneCycleLR.base_momentum: float - Lower momentum boundaries
        - scheduler.OneCycleLR.max_momentum: float - Upper momentum boundaries
        - scheduler.OneCycleLR.div_factor: float - Initial LR division factor
        - scheduler.OneCycleLR.final_div_factor: float - Final LR division factor
        - scheduler.OneCycleLR.last_epoch: int - Index of last epoch
        - scheduler.OneCycleLR.verbose: bool - Print message on updates

    :param config: Configuration reader containing scheduler parameters

    Example:
         # Create configuration
         config = ConfigReader({
             'scheduler.ReduceLROnPlateau.mode': 'min',
             'scheduler.ReduceLROnPlateau.factor': 0.1,
             'scheduler.ReduceLROnPlateau.patience': 10,
             'scheduler.ReduceLROnPlateau.verbose': True
         })
         
         # Initialize optimizer
         model = YourModel()
         optimizer = torch.optim.Adam(model.parameters())
         
         # Create scheduler factory and get scheduler
         factory = SchedulerFactory(config)
         scheduler = factory.get_scheduler('reducelronplateau', optimizer)
    """

    def __init__(self, config: ConfigReader):
        self.config = config
        self.builders = {
            'reducelronplateau': ReduceLROnPlateauBuilder,
            'steplr': StepLRBuilder,
            'cosineannealinglr': CosineAnnealingLRBuilder,
            'exponentiallr': ExponentialLRBuilder,
            'cosineannealingwarmrestarts': CosineAnnealingWarmRestartsBuilder,
            'onecyclelr': OneCycleLRBuilder
        }

    def get_scheduler(self, scheduler_type: str, optimizer: Optimizer) -> LRScheduler:
        """
        Create and return a learning rate scheduler based on the specified type.

        :param scheduler_type: Type of scheduler to create (case-insensitive).
                             Valid options: 'reducelronplateau', 'steplr', 'cosineannealinglr',
                             'exponentiallr', 'cosineannealingwarmrestarts', 'onecyclelr'
        :param optimizer: PyTorch optimizer to attach the scheduler to
        :return: Configured learning rate scheduler
        :raises ValueError: If the specified scheduler type is not supported

        Example:
        # Create StepLR scheduler
        config = ConfigReader({
                        'scheduler.StepLR.step_size': 30,
                        'scheduler.StepLR.gamma': 0.1,
                        'scheduler.StepLR.verbose': True
                    })
        factory = SchedulerFactory(config)
        step_scheduler = factory.get_scheduler('steplr', optimizer)
        
        # Create OneCycleLR scheduler
        config = ConfigReader({
                 'scheduler.OneCycleLR.max_lr': 0.1,
                 'scheduler.OneCycleLR.total_steps': 1000,
                 'scheduler.OneCycleLR.pct_start': 0.3
             })
        factory = SchedulerFactory(config)
        cycle_scheduler = factory.get_scheduler('onecyclelr', optimizer)
        """
        builder_class = self.builders.get(scheduler_type.lower())
        if not builder_class:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return builder_class(self.config).build(optimizer)
