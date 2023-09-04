import os
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def seed_everything(seed: int = 21):
    """
    Seed all random number generators for reproducibility.

    Parameters
    ----------
    seed : int, optional, default=21
        The seed to use.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def moving_average(x, win=10):
    """
    Calculate the moving average of a 1D array.

    Parameters
    ----------
    x : ndarray
        The input array.
    win : int, optional, default=10
        The window size.

    Returns
    -------
    ndarray
        The moving average of the input array.
    """
    return np.convolve(x, np.ones(win), "valid") / win


def matrix_to_diagonals(x: np.ndarray) -> np.ndarray:
    """
    Convert a 2d matrix to a matrix of diagonals.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        The input matrix.

    Returns
    -------
    ndarray, shape (n_samples, n_features, n_features)
        The matrix of diagonals.

    Example
    -------
    >>> matrix_to_diagonals(np.array([[1, 2], [3, 4]]))
    array([[[1, 0],
            [0, 2]],
           [[3, 0],
            [0, 4]]])
    """
    # create a zero matrix with the shape of (n_samples, n_features, n_features)
    o = np.zeros(x.shape + x.shape[-1:], dtype=x.dtype)
    # extract the diagonals according to the axis
    d = np.diagonal(o, axis1=1, axis2=2)
    # set the values of the diagonals to the values of the input matrix
    d.setflags(write=True)
    d[:] = x
    return o


def load_cifar_batch(filename):
    """
    Load a single batch of the CIFAR-10 dataset from disk.

    Parameters
    ----------
    filename : str
        The path to the batch file.

    Returns
    -------
    x : ndarray, shape (n_samples, 32, 32, 3)
        The images in the batch.
    y : ndarray, shape (n_samples,)
        The labels in the batch.
    """
    pack = np.load(file=filename, allow_pickle=True, encoding="latin1")
    x = pack["data"].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    y = np.asarray(pack["labels"])
    return x, y


def load_cifar10(path):
    """
    Load the CIFAR-10 dataset from disk.

    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).
    The classes are:
    | Label | Description |
    |:-----:|-------------|
    |   0   | airplane    |
    |   1   | automobile  |
    |   2   | bird        |
    |   3   | cat         |
    |   4   | deer        |
    |   5   | dog         |
    |   6   | frog        |
    |   7   | horse       |
    |   8   | ship        |
    |   9   | truck       |

    Parameters
    ----------
    path : str
        The path to the CIFAR-10 dataset.

    Returns
    -------
    (x_train, y_train), (x_test, y_test) : tuple
        A tuple containing the training and test data.

    Examples
    --------
    >>> (x_train, y_train), (x_test, y_test) = load_cifar10('data/cifar-10-batches-py')
    >>> assert x_train.shape == (50000, 32, 32, 3)
    >>> assert y_train.shape == (50000,)
    >>> assert x_test.shape == (10000, 32, 32, 3)
    >>> assert y_test.shape == (10000,)
    """
    path = Path(path)

    def load_all(patten):
        xs, ys = [], []
        for f in sorted(path.glob(patten)):
            x, y = load_cifar_batch(f)
            xs.append(x)
            ys.append(y)
        return np.concatenate(xs), np.concatenate(ys)

    x_train, y_train = load_all("data_batch*")
    x_test, y_test = load_all("test_batch*")
    return (x_train, y_train), (x_test, y_test)


def one_hot_encode(x):
    """
    One-hot encode a 1D array.

    This function takes a 1D array of integers and returns a 2D array of
    one-hot encoded values. The input array is assumed to contain integers
    in the range 0 to k-1, where k is the maximum value in the array.

    Parameters
    ----------
    x : ndarray
        The input array to encode.

    Returns
    -------
    ndarray
        The one-hot encoded array.

    Example
    -------
    >>> one_hot_encode(np.array([1, 0, 2, 3]))
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    """

    return np.eye(x.max() + 1)[x]


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of a classification model.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.

    Returns
    -------
    float
        The accuracy of the model, as a percentage.
    """

    return 100 * np.mean(y_true == y_pred)


def value_to_rgba(value, cmap=mpl.cm.gray_r, vmin=0, vmax=1):
    """Convert a value to an RGBA tuple using a Matplotlib colormap.

    Parameters
    ----------
    value : float
        The value to be converted to an RGBA tuple.
    cmap : Matplotlib colormap, optional
        The colormap to use for the conversion. The default is `matplitlib.cm.gray_r`,
        which uses a reversed grayscale colormap.
    vmin : float, optional
        The minimum value for the colormap normalization. The default is 0.
    vmax : float, optional
        The maximum value for the colormap normalization. The default is 1.

    Returns
    -------
    rgba : tuple
        An RGBA tuple representing the color corresponding to the input value.

    """

    # Create a Normalize object to scale the value to the range 0 to 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # Apply the colormap to the scaled value to get an RGBA tuple
    rgba = cmap(norm(abs(value)))
    return rgba


def weights_to_images(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights to a 4D array of images.

    Parameters
    ----------
    weights : ndarray, shape (batch_size, n_weights, n_features, channels)
        The weights to be converted to images.

    Returns
    -------
    images : ndarray, shape (n_weights, n_rows, n_columns, n_channels)
        The weights as images.

    """
    weights_min = np.min(weights, axis=(1, 2, 3), keepdims=True)
    weights_max = np.max(weights, axis=(1, 2, 3), keepdims=True)
    return ((weights - weights_min) * 255 / (weights_max - weights_min)).astype("uint8")


# ruff: noqa: PLR0913
def batch_plot(
    xs,
    ys=None,
    imgsize=4,
    with_border=True,
    tight_layout=True,
    flatten_layout=False,
    flatten_columns=False,
    wspace=0.1,
    hspace=0.1,
    save_path=None,
    **kwargs,
):
    """
    Plot a batch of images or color swatches in a grid layout using `matplotlib`.

    Parameters
    ----------
    xs : list or array
        A list or array of images or color swatch values to plot.
    ys : list or array, optional
        A list or array of labels for each image or color swatch. If not provided, no labels will be displayed.
    imgsize : int, optional, (default: 4)
        The size (in inches) of each image or color swatch.
    with_border : bool, optional, (default: True)
        A flag indicating whether to draw borders around the images or color swatches.
    tight_layout : bool, optional (default: True)
        A flag indicating whether to use `matplotlib`'s `tight_layout` to adjust the layout of the figure.
    flatten_layout : bool, optional, (default: False)
        A flag indicating whether to flatten the layout of the images or color swatches into a single row.
    flatten_columns : bool, optional, (default: False)
        A flag indicating whether to flatten the layout of the images or color swatches into a single column.
    wspace : float, optional, (default: 0.1)
        The amount of horizontal space (in inches) between subplots.
    hspace : float, optional, (default: 0.1)
        The amount of vertical space (in inches) between subplots.
    save_path : str, optional
        The path to save the figure to. If not provided, the figure will not be saved.
    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to Matplotlib's `imshow` function when plotting images.

    Returns
    -------
    None
    """

    # Calculate the number of rows and columns needed to lay out the images or color swatches in a grid
    rows = cols = int(np.ceil(np.sqrt(len(xs))))
    # Flatten the layout into a single row if specified
    if flatten_layout is True:
        if flatten_columns is True:
            rows = rows * cols
            cols = 1
        else:
            cols = rows * cols
            rows = 1
    if rows * cols > len(xs):
        rows = 1
    # Set default labels to None if not provided
    if ys is None:
        ys = np.full(len(xs), None)
    # Calculate the figure size through imgsize, rows and cols
    figure_size = (imgsize * cols, imgsize * rows)
    # Create the figure and axis in a grid layout
    fig, axs = plt.subplots(rows, cols, figsize=figure_size, tight_layout=tight_layout, squeeze=False)
    axs = np.array(axs)
    # Calculate the minimum and maximum values of the color swatches
    xs_min, xs_max = np.min(xs), np.max(xs)

    # Plot the images or color swatches
    for x, y, ax in zip(xs, ys, axs.flatten()):
        # ruff: noqa: PLR2004
        if x.ndim >= 2:
            # Plot images if the dimension of the x is larger than 1
            ax.imshow(x, **kwargs)
        else:
            # Plot color swatch
            ax.set_facecolor(value_to_rgba(x, vmin=xs_min, vmax=xs_max, **kwargs))
        # Set label as title if provided
        if y is not None:
            ax.set_title(y)
    # Hide the x- and y-axes and borders of the subplots
    for ax in axs.flatten():
        # ruff: noqa: PLR2004
        if with_border is False and x.ndim >= 2:
            ax.axis("off")
            continue
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # Adjust the spacing between the subplots
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, format="png")
        plt.close()
        return
    plt.show()


def generate_batches(n, batch_size, *, min_batch_size=0):
    """
    Generate batches of indices.

    Parameters
    ----------
    n : int
        The number of indices to generate batches for.
    batch_size : int
        The size of each batch.
    min_batch_size : int, optional
        The minimum size of each batch. The default is 0.

    Yields
    ------
    slice
        A slice object representing the indices of the current batch.
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)
