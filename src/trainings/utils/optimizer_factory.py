from abc import ABC, abstractmethod
from typing import Optional, Iterator

import torch
from torch.nn import Parameter
from torch.optim import Optimizer

from src.utils.config.config_reader import ConfigReader


class OptimizerBuilder(ABC):
    @abstractmethod
    def build(self, params: Iterator[Parameter]) -> Optimizer:
        pass


class AdamBuilder(OptimizerBuilder):
    """
    Builds Adam optimizer with configurable parameters.

    Config Parameters:
    - optimizer.Adam.lr: float - Learning rate (default: 1e-3)
    - optimizer.Adam.betas: tuple - Coefficients for computing running averages (default: (0.9, 0.999))
    - optimizer.Adam.eps: float - Term added for numerical stability (default: 1e-8)
    - optimizer.Adam.weight_decay: float - Weight decay coefficient (default: 0)
    - optimizer.Adam.amsgrad: bool - Whether to use AMSGrad variant (default: False)
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, params: Iterator[Parameter]) -> Optimizer:
        return torch.optim.Adam(
            params=params,
            lr=self.config.get_param('optimizer.Adam.lr', v_type=float),
            betas=self.config.get_collection('optimizer.Adam.betas', v_type=float, collection_type=tuple, num_elems=2),
            eps=self.config.get_param('optimizer.Adam.eps', v_type=float),
            weight_decay=self.config.get_param('optimizer.Adam.weight_decay', v_type=float),
            amsgrad=self.config.get_param('optimizer.Adam.amsgrad', v_type=bool)
        )


class AdamWBuilder(OptimizerBuilder):
    """
    Builds AdamW optimizer (Adam with decoupled weight decay).

    Config Parameters:
    - optimizer.AdamW.lr: float - Learning rate (default: 1e-3)
    - optimizer.AdamW.betas: tuple - Coefficients for computing running averages (default: (0.9, 0.999))
    - optimizer.AdamW.eps: float - Term added for numerical stability (default: 1e-8)
    - optimizer.AdamW.weight_decay: float - Weight decay coefficient (default: 0.01)
    - optimizer.AdamW.amsgrad: bool - Whether to use AMSGrad variant (default: False)
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, params: Iterator[Parameter]) -> Optimizer:
        return torch.optim.AdamW(
            params=params,
            lr=self.config.get_param('optimizer.AdamW.lr', v_type=float),
            betas=self.config.get_collection('optimizer.AdamW.betas', v_type=float, collection_type=tuple, num_elems=2),
            eps=self.config.get_param('optimizer.AdamW.eps', v_type=float),
            weight_decay=self.config.get_param('optimizer.AdamW.weight_decay', v_type=float),
            amsgrad=self.config.get_param('optimizer.AdamW.amsgrad', v_type=bool)
        )


class SGDBuilder(OptimizerBuilder):
    """
    Builds SGD optimizer with configurable parameters.

    Config Parameters:
    - optimizer.SGD.lr: float - Learning rate (default: 1e-1)
    - optimizer.SGD.momentum: float - Momentum factor (default: 0)
    - optimizer.SGD.dampening: float - Dampening for momentum (default: 0)
    - optimizer.SGD.weight_decay: float - Weight decay coefficient (default: 0)
    - optimizer.SGD.nesterov: bool - Enables Nesterov momentum (default: False)
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, params: Iterator[Parameter]) -> Optimizer:
        return torch.optim.SGD(
            params=params,
            lr=self.config.get_param('optimizer.SGD.lr', v_type=float),
            momentum=self.config.get_param('optimizer.SGD.momentum', v_type=float),
            dampening=self.config.get_param('optimizer.SGD.dampening', v_type=float),
            weight_decay=self.config.get_param('optimizer.SGD.weight_decay', v_type=float),
            nesterov=self.config.get_param('optimizer.SGD.nesterov', v_type=bool)
        )


class RMSpropBuilder(OptimizerBuilder):
    """
    Builds RMSprop optimizer with configurable parameters.

    Config Parameters:
    - optimizer.RMSprop.lr: float - Learning rate (default: 1e-2)
    - optimizer.RMSprop.alpha: float - Smoothing constant (default: 0.99)
    - optimizer.RMSprop.eps: float - Term added for numerical stability (default: 1e-8)
    - optimizer.RMSprop.weight_decay: float - Weight decay coefficient (default: 0)
    - optimizer.RMSprop.momentum: float - Momentum factor (default: 0)
    - optimizer.RMSprop.centered: bool - If True, compute the centered RMSprop (default: False)
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, params: Iterator[Parameter]) -> Optimizer:
        return torch.optim.RMSprop(
            params=params,
            lr=self.config.get_param('optimizer.RMSprop.lr', v_type=float),
            alpha=self.config.get_param('optimizer.RMSprop.alpha', v_type=float),
            eps=self.config.get_param('optimizer.RMSprop.eps', v_type=float),
            weight_decay=self.config.get_param('optimizer.RMSprop.weight_decay', v_type=float),
            momentum=self.config.get_param('optimizer.RMSprop.momentum', v_type=float),
            centered=self.config.get_param('optimizer.RMSprop.centered', v_type=bool)
        )


class AdadeltaBuilder(OptimizerBuilder):
    """
    Builds Adadelta optimizer with configurable parameters.

    Config Parameters:
    - optimizer.Adadelta.lr: float - Learning rate (default: 1.0)
    - optimizer.Adadelta.rho: float - Coefficient for computing running average (default: 0.9)
    - optimizer.Adadelta.eps: float - Term added for numerical stability (default: 1e-6)
    - optimizer.Adadelta.weight_decay: float - Weight decay coefficient (default: 0)
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, params: Iterator[Parameter]) -> Optimizer:
        return torch.optim.Adadelta(
            params=params,
            lr=self.config.get_param('optimizer.Adadelta.lr', v_type=float),
            rho=self.config.get_param('optimizer.Adadelta.rho', v_type=float),
            eps=self.config.get_param('optimizer.Adadelta.eps', v_type=float),
            weight_decay=self.config.get_param('optimizer.Adadelta.weight_decay', v_type=float)
        )


class AdagradBuilder(OptimizerBuilder):
    """
    Builds Adagrad optimizer with configurable parameters.

    Config Parameters:
    - optimizer.Adagrad.lr: float - Learning rate (default: 1e-2)
    - optimizer.Adagrad.lr_decay: float - Learning rate decay (default: 0)
    - optimizer.Adagrad.weight_decay: float - Weight decay coefficient (default: 0)
    - optimizer.Adagrad.initial_accumulator_value: float - Initial accumulator value (default: 0)
    - optimizer.Adagrad.eps: float - Term added for numerical stability (default: 1e-10)
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, params: Iterator[Parameter]) -> Optimizer:
        return torch.optim.Adagrad(
            params=params,
            lr=self.config.get_param('optimizer.Adagrad.lr', v_type=float),
            lr_decay=self.config.get_param('optimizer.Adagrad.lr_decay', v_type=float),
            weight_decay=self.config.get_param('optimizer.Adagrad.weight_decay', v_type=float),
            initial_accumulator_value=self.config.get_param('optimizer.Adagrad.initial_accumulator_value',
                                                            v_type=float),
            eps=self.config.get_param('optimizer.Adagrad.eps', v_type=float)
        )


class RAdamBuilder(OptimizerBuilder):
    """
    Builds RAdam optimizer (Rectified Adam) with configurable parameters.

    Config Parameters:
    - optimizer.RAdam.lr: float - Learning rate (default: 1e-3)
    - optimizer.RAdam.betas: tuple - Coefficients for computing running averages (default: (0.9, 0.999))
    - optimizer.RAdam.eps: float - Term added for numerical stability (default: 1e-8)
    - optimizer.RAdam.weight_decay: float - Weight decay coefficient (default: 0)
    """

    def __init__(self, config: ConfigReader):
        self.config = config

    def build(self, params: Iterator[Parameter]) -> Optimizer:
        return torch.optim.RAdam(
            params=params,
            lr=self.config.get_param('optimizer.RAdam.lr', v_type=float),
            betas=self.config.get_collection('optimizer.RAdam.betas', v_type=float, collection_type=tuple, num_elems=2),
            eps=self.config.get_param('optimizer.RAdam.eps', v_type=float),
            weight_decay=self.config.get_param('optimizer.RAdam.weight_decay', v_type=float)
        )


class OptimizerFactory:
    """
    Factory class for creating PyTorch optimizers based on configuration.

    Supports multiple optimizer types including:
    - Adam
    - AdamW
    - SGD
    - RMSprop
    - Adadelta
    - Adagrad
    - RAdam

    Required configuration parameters for each optimizer type:

    Adam:
        - optimizer.Adam.lr: float - Learning rate
        - optimizer.Adam.betas: tuple - Running average coefficients
        - optimizer.Adam.eps: float - Term for numerical stability
        - optimizer.Adam.weight_decay: float - Weight decay
        - optimizer.Adam.amsgrad: bool - Whether to use AMSGrad

    AdamW:
        - optimizer.AdamW.lr: float - Learning rate
        - optimizer.AdamW.betas: tuple - Running average coefficients
        - optimizer.AdamW.eps: float - Term for numerical stability
        - optimizer.AdamW.weight_decay: float - Weight decay
        - optimizer.AdamW.amsgrad: bool - Whether to use AMSGrad

    SGD:
        - optimizer.SGD.lr: float - Learning rate
        - optimizer.SGD.momentum: float - Momentum factor
        - optimizer.SGD.dampening: float - Dampening for momentum
        - optimizer.SGD.weight_decay: float - Weight decay
        - optimizer.SGD.nesterov: bool - Enables Nesterov momentum

    RMSprop:
        - optimizer.RMSprop.lr: float - Learning rate
        - optimizer.RMSprop.alpha: float - Smoothing constant
        - optimizer.RMSprop.eps: float - Term for numerical stability
        - optimizer.RMSprop.weight_decay: float - Weight decay
        - optimizer.RMSprop.momentum: float - Momentum factor
        - optimizer.RMSprop.centered: bool - If True, compute centered RMSprop

    Adadelta:
        - optimizer.Adadelta.lr: float - Learning rate
        - optimizer.Adadelta.rho: float - Coefficient for running average
        - optimizer.Adadelta.eps: float - Term for numerical stability
        - optimizer.Adadelta.weight_decay: float - Weight decay

    Adagrad:
        - optimizer.Adagrad.lr: float - Learning rate
        - optimizer.Adagrad.lr_decay: float - Learning rate decay
        - optimizer.Adagrad.weight_decay: float - Weight decay
        - optimizer.Adagrad.initial_accumulator_value: float - Initial accumulator value
        - optimizer.Adagrad.eps: float - Term for numerical stability

    RAdam:
        - optimizer.RAdam.lr: float - Learning rate
        - optimizer.RAdam.betas: tuple - Running average coefficients
        - optimizer.RAdam.eps: float - Term for numerical stability
        - optimizer.RAdam.weight_decay: float - Weight decay

    :param config: Configuration reader containing optimizer parameters

    Example:
        # Create configuration
        config = ConfigReader({
            'optimizer.Adam.lr': 0.001,
            'optimizer.Adam.betas': (0.9, 0.999),
            'optimizer.Adam.eps': 1e-8,
            'optimizer.Adam.weight_decay': 0,
            'optimizer.Adam.amsgrad': False
        })

        # Initialize model
        model = YourModel()

        # Create optimizer factory and get optimizer
        factory = OptimizerFactory(config)
        optimizer = factory.get_optimizer('adam', model.parameters())
    """

    def __init__(self, config: ConfigReader):
        self.config = config
        self.builders = {
            'adam': AdamBuilder,
            'adamw': AdamWBuilder,
            'sgd': SGDBuilder,
            'rmsprop': RMSpropBuilder,
            'adadelta': AdadeltaBuilder,
            'adagrad': AdagradBuilder,
            'radam': RAdamBuilder
        }

    def get_optimizer(self, optimizer_type: str, params: Iterator[Parameter]) -> Optional[Optimizer]:
        """
        Create and return an optimizer based on the specified type.

        :param optimizer_type: Type of optimizer to create (case-insensitive).
                             Valid options: 'adam', 'adamw', 'sgd', 'rmsprop',
                             'adadelta', 'adagrad', 'radam'
        :param params: Model parameters to optimize
        :return: Configured optimizer
        :raises ValueError: If the specified optimizer type is not supported

        Example:
            # Create Adam optimizer
            config = ConfigReader({
                'optimizer.Adam.lr': 0.001,
                'optimizer.Adam.weight_decay': 1e-5
            })
            factory = OptimizerFactory(config)
            adam_optimizer = factory.get_optimizer('adam', model.parameters())

            # Create SGD optimizer
            config = ConfigReader({
                'optimizer.SGD.lr': 0.1,
                'optimizer.SGD.momentum': 0.9
            })
            factory = OptimizerFactory(config)
            sgd_optimizer = factory.get_optimizer('sgd', model.parameters())
        """
        builder_class = self.builders.get(optimizer_type.lower())
        if not builder_class:
            supported_optimizers = "', '".join(self.builders.keys())
            raise ValueError(
                f"Unsupported optimizer type: {optimizer_type}. Supported types are: '{supported_optimizers}'")
        return builder_class(self.config).build(params)
