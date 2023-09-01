from collections import defaultdict

import numpy as np
from utils import seed_everything


# ruff: noqa: PLR0913
def train(
    x: np.ndarray,
    y: np.ndarray,
    reg: float,
    learning_rate: float = 1e-3,
    num_iters: int = 1000,
    batch_size: int = 128,
    verbose: bool = False,
) -> tuple:
    """
    Train a linear classifier using stochastic gradient descent.

    Parameters
    ----------
    x : ndarray, shape (N, H, W, C)
        The input images.
    y : ndarray, shape (N,)
        The labels of the input images.
    reg : float
        The regularization strength.
    learning_rate : float
        The learning rate.
    num_iters : int
        The number of iterations to train for.
    batch_size : int
        The number of images to use in each batch.
    verbose : bool
        Whether to print the loss during training.

    Returns
    -------
    history : dict
        The training history with keys "loss" and "weights".
    weights : ndarray, shape (M, H, W, C)
        The weights of the softmax treated as images for visualization.
    bias : ndarray, shape (M,)
        The bias of the softmax.
    """
    seed_everything()
    num_train, height, width, channels = x.shape
    num_classes = np.max(y) + 1
    weights = np.random.normal(size=(num_classes, height, width, channels), scale=1e-3)
    bias = np.zeros(num_classes)

    history = defaultdict(list)
    for i in range(num_iters):
        batch_indices = np.random.choice(num_train, size=batch_size)
        x_batch, y_batch = x[batch_indices], y[batch_indices]
        loss, dw, db = softmax_loss_original_with_grads(x_batch, y_batch, weights, bias, reg)
        weights -= learning_rate * dw
        bias -= learning_rate * db
        if verbose and i % 100 == 0:
            print(f"Iteration {i} / {num_iters}: loss = {loss}")
        history["loss"].append(loss)
        history["weights"].append(np.copy(weights))
        history["bias"].append(np.copy(bias))
    return history, weights, bias


def predict(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Predict the labels of the input images using the trained softmax.

    Parameters
    ----------
    x : ndarray, shape (N, H, W, C)
        The input images.
    weights : ndarray, shape (M, H, W, C)
        The weights of the softmax treated as images for visualization.
    bias : ndarray, shape (M,)
        The bias of the softmax.

    Returns
    -------
    ndarray, shape (N,)
        The predicted labels of the input images.
    """
    scores = np.einsum("nhwc,mhwc->nm", x, weights) + bias
    return np.argmax(scores, axis=1)


def softmax_loss_original(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray, reg: float) -> float:
    """
    Calculate the loss of the softmax using the original image shape.
    This implementation is slower but more intuitive as it uses the original image to learn the weights as templates.

    Parameters
    ----------
    x : ndarray, shape (N, H, W, C)
        The input images.
    y : ndarray, shape (N,)
        The labels of the input images.
    weights : ndarray, shape (M, H, W, C)
        The weights of the softmax treated as images for visualization.
    bias : ndarray, shape (M,)
        The bias of the softmax.
    reg : float
        The regularization strength.

    Returns
    -------
    float
        The loss of the softmax.
    """
    num_train, *_ = x.shape
    s = np.einsum("nhwc,mhwc->nm", x, weights) + bias
    e = np.exp(s - np.max(s, axis=1, keepdims=True))
    p = e / np.sum(e, axis=1, keepdims=True)
    loss = -np.log(p[np.arange(num_train), y])
    return np.sum(loss) / num_train + reg * np.sum(weights * weights)


def softmax_loss_reshaped(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray, reg: float) -> float:
    """
    Calculate the loss of the softmax using the flattened image shape.

    Parameters
    ----------
    x : ndarray, shape (N, H*W*C)
        The input images reshaped as a 2d array.
    y : ndarray, shape (N,)
        The labels of the input images.
    weights : ndarray, shape (H*W*C, M)
        The weights of the softmax reshaped as a 2d array.
    bias : ndarray, shape (M,)
        The bias of the softmax.
    reg : float
        The regularization strength.

    Returns
    -------
    float
        The loss of the softmax.
    """
    num_train, *_ = x.shape
    s = x @ weights + bias
    e = np.exp(s - np.max(s, axis=1, keepdims=True))
    p = e / np.sum(e, axis=1, keepdims=True)
    loss = -np.log(p[np.arange(num_train), y])
    return np.sum(loss) / num_train + reg * np.sum(weights * weights)


def softmax_loss_original_with_grads(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray, reg: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate the loss and gradients of the softmax using the original image shape.
    This implementation is slower but more intuitive as it uses the original image to learn the weights as templates.

    Parameters
    ----------
    x : ndarray, shape (N, H, W, C)
        The input images.
    y : ndarray, shape (N,)
        The labels of the input images.
    weights : ndarray, shape (M, H, W, C)
        The weights of the softmax treated as images for visualization.
    bias : ndarray, shape (M,)
        The bias of the softmax.
    reg : float
        The regularization strength.

    Returns
    -------
    tuple(float, ndarray, ndarray)
        The loss of the softmax, the gradient of the weights, and the gradient of the bias.
    """
    num_train, *_ = x.shape
    s = np.einsum("nhwc,mhwc->nm", x, weights) + bias
    e = np.exp(s - np.max(s, axis=1, keepdims=True))
    p = e / np.sum(e, axis=1, keepdims=True)
    loss = -np.log(p[np.arange(num_train), y])
    loss = np.sum(loss) / num_train + reg * np.sum(weights * weights)

    # Compute the gradient of the weights and bias.
    p[np.arange(num_train), y] -= 1
    dw = np.einsum("nm,nhwc->mhwc", p, x) / num_train + 2 * reg * weights
    db = np.sum(p, axis=0)
    return loss, dw, db


def softmax_loss_reshaped_with_grads(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray, reg: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate the loss and gradients of the softmax using the flattened image shape.

    Parameters
    ----------
    x : ndarray, shape (N, H*W*C)
        The input images reshaped as a 2d array.
    y : ndarray, shape (N,)
        The labels of the input images.
    weights : ndarray, shape (H*W*C, M)
        The weights of the softmax reshaped as a 2d array.
    bias : ndarray, shape (M,)
        The bias of the softmax.
    reg : float
        The regularization strength.

    Returns
    -------
    tuple(float, ndarray, ndarray)
        The loss of the softmax, the gradient of the weights, and the gradient of the bias.
    """
    num_train, *_ = x.shape
    s = x @ weights + bias
    e = np.exp(s - np.max(s, axis=1, keepdims=True))
    p = e / np.sum(e, axis=1, keepdims=True)
    loss = -np.log(p[np.arange(num_train), y])
    loss = np.sum(loss) / num_train + reg * np.sum(weights * weights)

    # Compute the gradient of the weights and bias.
    p[np.arange(num_train), y] -= 1
    dw = x.T @ p / num_train + 2 * reg * weights
    db = np.sum(p, axis=0)
    return loss, dw, db
