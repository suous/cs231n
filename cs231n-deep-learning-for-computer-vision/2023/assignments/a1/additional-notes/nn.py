from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np
from functional import (
    cross_entropy,
    cross_entropy_derivative,
    linear,
    log_softmax,
    log_softmax_derivative,
    nll_loss,
    nll_loss_derivative,
    relu,
    softmax,
    softmax_derivative,
)
from utils import generate_batches


@dataclass
class Parameter:
    """
    Variable that is used in a computation graph of the backpropagation algorithm.

    Parameters
    ----------
    value : np.ndarray
        The value of the parameter.
    grad: np.ndarray, optional
        The gradient of the parameter.
    """

    value: np.ndarray
    grad: np.ndarray = None


class Base(ABC):
    """
    Base class for all layers and models.

    Properties
    ----------
    parameters : List[Parameter], default []
    """

    @property
    def parameters(self) -> List[Parameter]:
        return []

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, *args, **kwargs):
        raise NotImplementedError()


@dataclass
class Linear(Base):
    """
    A fully connected layer of a neural network.

    Parameters
    ----------
    in_features : int
        The number of input features.
    out_features : int
        The number of output features.
    """

    in_features: int
    out_features: int
    regularization: float = 0.0
    _weight: Parameter = field(init=False, repr=False)
    _bias: Parameter = field(init=False, repr=False)
    _cache: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # Xavier initialization
        std = np.sqrt(2.0 / (self.in_features + self.out_features))
        init_bound = np.sqrt(3.0) * std
        self._weight = Parameter(np.random.uniform(-init_bound, init_bound, (self.in_features, self.out_features)))
        self._bias = Parameter(np.random.uniform(-init_bound, init_bound, (1, self.out_features)))

    @property
    def parameters(self) -> List[Parameter]:
        return [self._weight, self._bias]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return linear(x, self._weight.value, self._bias.value) + self.regularization * np.sum(self._weight.value**2)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self._weight.grad = self._cache.T @ grad + 2 * self.regularization * self._weight.value
        self._bias.grad = np.sum(grad, axis=0, keepdims=True)
        return grad @ self._weight.value.T


@dataclass
class ReLU(Base):
    """Rectified linear unit activation function."""

    _cache: np.ndarray = field(init=False, repr=False, default=None)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return relu(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self._cache > 0).astype(int)


@dataclass
class Softmax(Base):
    """Softmax activation function."""

    _cache: np.ndarray = field(init=False, repr=False, default=None)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = softmax(x)
        self._cache = z
        return z

    def backward(self, grad: np.ndarray) -> np.ndarray:
        s = softmax_derivative(self._cache)
        return (np.expand_dims(grad, 1) @ s).squeeze(axis=1)


@dataclass
class LogSoftmax(Base):
    """Log-softmax activation function."""

    _cache: np.ndarray = field(init=False, repr=False, default=None)

    def forward(self, x: np.ndarray) -> np.ndarray:
        s, z = log_softmax(x, with_softmax=True)
        self._cache = z
        return s

    def backward(self, grad: np.ndarray) -> np.ndarray:
        s = log_softmax_derivative(self._cache)
        return (np.expand_dims(grad, 1) @ s).squeeze(axis=1)


class CrossEntropy(Base):
    """Categorical cross entropy loss function."""

    reduction: str = "mean"
    with_softmax: bool = True

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return cross_entropy(x, y, reduction=self.reduction, with_softmax=self.with_softmax)

    def backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return cross_entropy_derivative(x, y, reduction=self.reduction, with_softmax=self.with_softmax)


@dataclass
class NLLLoss(Base):
    """Negative log-likelihood loss function."""

    reduction: str = "mean"

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return nll_loss(x, y, reduction=self.reduction)

    def backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return nll_loss_derivative(x, y, reduction=self.reduction)


################################################################################
# Optimizers
################################################################################


@dataclass
class Optimizer(ABC):
    """
    Base class for all optimizers.

    Parameters
    ----------
    params : List[Parameter], default []
        The parameters to optimize.
    learning_rate : float, default 0.01
        The learning rate of the optimizer.
    """

    params: List[Parameter] = field(default_factory=list, repr=False)
    learning_rate: float = 0.01

    def reset_grad(self):
        for param in self.params:
            param.grad = None

    @abstractmethod
    def update(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def update(self):
        for param in self.params:
            param.value -= self.learning_rate * param.grad


@dataclass
class Sequential:
    """
    Sequential model, a linear stack of layers.

    Parameters
    ----------
    layers : List[Base], default []
        The layers of the model.
    optimizer : Optimizer, default None
        The optimizer of the model.
    loss : Base, default None
        The loss function of the model.
    """

    layers: List[Base] = field(default_factory=list)
    optimizer: Optimizer = field(init=False, default=None)
    loss: Base = field(init=False, default=None)

    @property
    def parameters(self) -> List[Parameter]:
        return [p for layer in self.layers for p in layer.parameters]

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def _train_on_batch(self, x, y):
        y_hat = self._forward(x)
        loss = self.loss.forward(y_hat, y)
        grad = self.loss.backward(y_hat, y)
        self._backward(grad=grad)
        return loss

    def _test_on_batch(self, x, y):
        y_hat = self._forward(x)
        loss = self.loss.forward(y_hat, y)
        return loss

    # ruff: noqa: PLR0913
    def fit(self, x, y, epochs=100, batch_size=200, validation_data=None, verbose=True, store_weights=False):
        x_val, y_val = None, None
        if validation_data is not None:
            x_val, y_val = validation_data
            n_val_samples = x_val.shape[0]
        n_samples = x.shape[0]
        history = defaultdict(list)
        for epoch in range(epochs):
            train_loss = np.mean([self._train_on_batch(x[i], y[i]) for i in generate_batches(n_samples, batch_size)])
            history["train_loss"].append(train_loss)
            if x_val is not None and y_val is not None:
                valid_loss = np.mean(
                    [self._test_on_batch(x_val[i], y_val[i]) for i in generate_batches(n_val_samples, batch_size)]
                )
                history["valid_loss"].append(valid_loss)
            if verbose:
                if validation_data is not None:
                    print(f"Epoch: {epoch}/{epochs}, Cost: {train_loss}, Val Cost: {valid_loss}")
                else:
                    print(f"Epoch: {epoch}/{epochs}, Cost: {train_loss}")
            if store_weights:
                history["weights"].append(self.parameters[0].value.copy())
        return history

    def _forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _backward(self, grad: np.ndarray) -> np.ndarray:
        self.optimizer.reset_grad()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        self.optimizer.update()
        return grad

    def predict(self, x):
        return self._forward(x)
