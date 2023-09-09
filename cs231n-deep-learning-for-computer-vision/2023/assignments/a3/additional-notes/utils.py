import os
import random

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
