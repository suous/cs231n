from typing import Tuple, Union

import numpy as np
from utils import matrix_to_diagonals


def relu(x: np.ndarray) -> np.ndarray:
    """
    Calculate the ReLU function for a given input x.

    Parameters
    ----------
    x : ndarray
        The input to the ReLU function.

    Returns
    -------
    ndarray
        The output of the ReLU function applied element-wise to the input array.
    """
    return np.clip(x, 0, np.finfo(x.dtype).max)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Calculate the softmax function for a given input x.

    Parameters
    ----------
    x : ndarray
        The input to the softmax function.

    Returns
    -------
    ndarray
        The output of the softmax function applied element-wise to the input array.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def log_softmax(x: np.ndarray, with_softmax: bool = False) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Calculate the log softmax function for a given input x.

    Parameters
    ----------
    x : ndarray
        The input to the softmax function.
    with_softmax : bool, optional, default=False
        Whether to return the softmax function as well.

    Returns
    -------
    tuple
        The output of the log softmax function applied element-wise to the input array.
        If `with_softmax` is True, the softmax function is also returned.
    """
    t = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(t)
    s = np.sum(e, axis=-1, keepdims=True)
    if with_softmax:
        return t - np.log(s), e / s
    return t - np.log(s), None


def softmax_derivative(x: np.ndarray) -> np.ndarray:
    """
    Calculate the derivative of the softmax function for a given input x.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        The input to the softmax function.

    Returns
    -------
    ndarray, shape (n_samples, n_features, n_features)
        The matrix of derivatives of the softmax function.
    """

    """
    diag: (n_classes, n_classes)
    |  x_i[0]           0                0                |
    |  0                x_i[1]           0                |
    |  0                0                x_i[2]           |
    outer: (n_classes, n_classes)
    |  x_i[0] * x_j[0]  x_i[0] * x_j[1]  x_i[0] * x_j[2]  |
    |  x_i[1] * x_j[0]  x_i[1] * x_j[1]  x_i[1] * x_j[2]  |
    |  x_i[2] * x_j[0]  x_i[2] * x_j[1]  x_i[2] * x_j[2]  |
    diag - outer: (n_classes, n_classes)
    |  x_i[0] - x_i[0] * x_j[0]  -x_i[0] * x_j[1]          -x_i[0] * x_j[2]           |
    |  -x_i[1] * x_j[0]          x_i[1] - x_i[1] * x_j[1]  -x_i[1] * x_j[2]           |
    |  -x_i[2] * x_j[0]          -x_i[2] * x_j[1]           x_i[2] - x_i[2] * x_j[2]  |
    """
    return matrix_to_diagonals(x) - np.einsum("bi,bj->bij", x, x)


def log_softmax_derivative(x: np.ndarray) -> np.ndarray:
    """
    Calculate the derivative of the log softmax function for a given input x.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        The input to the softmax function.

    Returns
    -------
    ndarray, shape (n_samples, n_features, n_features)
        The matrix of derivatives of the softmax function.
    """
    n, c = x.shape
    return np.repeat(np.eye(c)[None, :, :], n, axis=0) - np.repeat(x[:, None, :], c, axis=1)


def nll_loss(x: np.ndarray, y: np.ndarray, reduction: str = "mean") -> np.ndarray:
    """
    Calculate the negative log-likelihood loss for a given input x and target y.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        The input to the softmax function.
    y : ndarray, shape (n_samples,)
        The target values.
    reduction : str, optional, default="mean"
        The reduction method to use.

    Returns
    -------
    ndarray
        The negative log-likelihood loss.
    """
    assert reduction in ["mean", "sum", "none"], "Invalid reduction method."
    loss = -x[np.arange(x.shape[0]), y]
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    return loss


def nll_loss_derivative(x: np.ndarray, y: np.ndarray, reduction: str = "mean") -> np.ndarray:
    """
    Calculate the derivative of the negative log-likelihood loss for a given input x and target y.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        The input to the softmax function.
    y : ndarray, shape (n_samples,)
        The target values.
    reduction : str, optional, default="mean"
        The reduction method to use.

    Returns
    -------
    ndarray
        The derivative of the negative log-likelihood loss.
    """
    assert reduction in ["mean", "sum", "none"], "Invalid reduction method."
    n, c = x.shape
    d = np.zeros((n, c))
    d[np.arange(n), y] = -1
    if reduction == "mean":
        return d / n
    elif reduction == "sum":
        return d
    return d


def cross_entropy(x: np.ndarray, y: np.ndarray, reduction: str = "mean", with_softmax=True) -> np.ndarray:
    """
    Calculate the cross-entropy loss for a given input x and target y.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        The input to the softmax function.
    y : ndarray, shape (n_samples,)
        The target values.
    reduction : str, optional, default="mean"
        The reduction method to use.
    with_softmax : bool, optional, default=True
        Whether to apply the softmax function to the input.

    Returns
    -------
    ndarray
        The cross-entropy loss.
    """
    assert reduction in ["mean", "sum", "none"], "Invalid reduction method."
    if with_softmax:
        log_soft, _ = log_softmax(x, with_softmax=False)
        loss = -log_soft[range(len(y)), y]
    else:
        loss = -np.log(x[range(len(y)), y])
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    return loss


def cross_entropy_derivative(x: np.ndarray, y: np.ndarray, reduction: str = "mean", with_softmax=True) -> np.ndarray:
    """
    Calculate the derivative of the cross-entropy loss for a given input x and target y.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        The input to the softmax function.
    y : ndarray, shape (n_samples,)
        The target values.
    reduction : str, optional, default="mean"
        The reduction method to use.
    with_softmax : bool, optional, default=True
        Whether to apply the softmax function to the input.

    Returns
    -------
    ndarray
        The derivative of the cross-entropy loss.
    """
    assert reduction in ["mean", "sum", "none"], "Invalid reduction method."
    n, c = x.shape
    d = np.zeros((n, c))
    if with_softmax:
        d = softmax(x)
        d[np.arange(n), y] -= 1
    else:
        d[np.arange(n), y] = -1 / x[np.arange(n), y]
    if reduction == "mean":
        return d / n
    elif reduction == "sum":
        return d
    return d


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate the linear function for a given input x.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        The input to the linear function.
    w : ndarray, shape (n_features, n_classes)
        The weight matrix.
    b : ndarray, shape (n_classes,)
        The bias vector.

    Returns
    -------
    ndarray, shape (n_samples, n_classes)
        The output of the linear function.
    """
    return x @ w + b
