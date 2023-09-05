from typing import Optional, Union

import numpy as np
from scipy import signal


def generate_output_size(image_size, kernel_size, stride, padding):
    """
    Calculates the output width of a convolutional layer.

    Parameters
    ----------
    image_size : int
        The width (or height) of the input image.
    kernel_size : int
        The width (or height) of the convolutional kernel.
    padding : int
        The amount of zero padding applied to the input.
    stride : int
        The stride of the convolution.

    Returns
    -------
    int
        The output width (or height) of the convolutional layer.
    """
    return (image_size - kernel_size + 2 * padding) // stride + 1


def window_images(
    images: np.ndarray,
    kernel_size: Union[int, tuple[int]] = 3,
    stride: Optional[Union[int, tuple[int]]] = None,
    padding: Union[int, tuple[int]] = 0,
    channels_first: bool = True,
) -> np.ndarray:
    """
    Convert images to sliding windowed view of chunks for convolution.

    Parameters
    ----------
    images: np.ndarray
        The input images of shape (batch_size, channels, height, width) or (batch_size, height, width, channels).
    kernel_size: int or tuple of ints, (kernel_height, kernel_width)
        The size of the convolution kernel.
    stride: int or tuple of ints, (stride_height, stride_width)
        The stride of the convolution, usually stride_height=string_width
    padding: int or tuple of ints, (padding_height, padding_width)
        The padding value of the convolution, usually padding_height=padding_width
    channels_first: bool
        Whether the channels are the first dimension of the input images.

    Returns
    -------
    np.ndarray
        The sliding windowed view of the input images.
    """
    # ruff: noqa: PLR2004
    assert images.ndim == 4, f"images must be 4-dimensional, got {images.ndim} dimensions"
    assert isinstance(kernel_size, (int, tuple)), f"kernel_size must be int or tuple, got {type(kernel_size)}"
    assert stride is None or isinstance(
        stride, (int, tuple)
    ), f"stride must be None or int or tuple, got {type(stride)}"
    assert padding is None or isinstance(
        padding, (int, tuple)
    ), f"padding must be None or int or tuple, got {type(padding)}"

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    square_dim = 2
    if len(kernel_size) != square_dim:
        raise ValueError(f"kernel_size must be int or tuple of length 2, got {kernel_size}")

    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if len(stride) != square_dim:
        raise ValueError(f"stride must be None or int or tuple of length 2, got {stride}")

    if isinstance(padding, int):
        padding = (padding, padding)
    if len(padding) != square_dim:
        raise ValueError(f"padding must be None or int or tuple of length 2, got {padding}")

    if channels_first:
        return _window_images_channel_first(images=images, kernel_size=kernel_size, stride=stride, padding=padding)
    return _window_images_channel_last(images=images, kernel_size=kernel_size, stride=stride, padding=padding)


def _window_images_channel_first(
    images: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> np.ndarray:
    """
    Convert images to sliding windowed view of chunks for convolution.

    Parameters
    ----------
    images: np.ndarray, shape=(batch_size, channels, height, width)
        The input images.
    kernel_size: tuple of ints, (kernel_height, kernel_width)
        The size of the convolution kernel.
    stride: tuple of ints, (stride_height, stride_width)
        The stride of the convolution, usually stride_height=string_width
    padding: tuple of ints, (padding_height, padding_width)
        The padding value of the convolution, usually padding_height=padding_width

    Returns
    -------
    np.ndarray, shape=(batch_size, channels, height, width, kernel_height, kernel_width)
        The sliding windowed view of the input images.
    """
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    padding_height, padding_width = padding

    batch_size, image_channels, image_height, image_width = images.shape

    if padding_height > 0 or padding_width > 0:
        images = np.pad(
            array=images,
            pad_width=((0, 0), (0, 0), (padding_height, padding_width), (padding_height, padding_width)),
        )

    stride_batch_size, stride_image_channels, stride_image_height, stride_image_width = images.strides

    windowed_height = generate_output_size(
        image_size=image_height,
        kernel_size=kernel_height,
        stride=stride_height,
        padding=padding_height,
    )
    windowed_width = generate_output_size(
        image_size=image_width,
        kernel_size=kernel_width,
        stride=stride_width,
        padding=padding_width,
    )
    stride_block_height, stride_block_width = (
        stride_height * stride_image_height,
        stride_width * stride_image_width,
    )

    return np.lib.stride_tricks.as_strided(
        x=images,
        shape=(batch_size, image_channels, windowed_height, windowed_width, kernel_height, kernel_width),
        strides=(
            stride_batch_size,
            stride_image_channels,
            stride_block_height,
            stride_block_width,
            stride_image_height,
            stride_image_width,
        ),
    )


def _window_images_channel_last(
    images: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> np.ndarray:
    """
    Convert images to sliding windowed view of chunks for convolution.

    Parameters
    ----------
    images: np.ndarray, shape=(batch_size, height, width, channels)
        The input images.
    kernel_size: tuple of ints, (kernel_height, kernel_width)
        The size of the convolution kernel.
    stride: tuple of ints, (stride_height, stride_width)
        The stride of the convolution, usually stride_height=string_width
    padding: tuple of ints, (padding_height, padding_width)
        The padding value of the convolution, usually padding_height=padding_width

    Returns
    -------
    np.ndarray, shape=(batch_size, height, width, kernel_height, kernel_width, channels)
        The sliding windowed view of the input images.
    """
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    padding_height, padding_width = padding

    batch_size, image_height, image_width, image_channels = images.shape

    if padding_height > 0 or padding_width > 0:
        images = np.pad(
            array=images,
            pad_width=((0, 0), (padding_height, padding_width), (padding_height, padding_width), (0, 0)),
        )

    stride_batch_size, stride_image_height, stride_image_width, stride_image_channels = images.strides

    windowed_height = generate_output_size(
        image_size=image_height,
        kernel_size=kernel_height,
        stride=stride_height,
        padding=padding_height,
    )
    windowed_width = generate_output_size(
        image_size=image_width,
        kernel_size=kernel_width,
        stride=stride_width,
        padding=padding_width,
    )
    stride_block_height, stride_block_width = (
        stride_height * stride_image_height,
        stride_width * stride_image_width,
    )

    return np.lib.stride_tricks.as_strided(
        x=images,
        shape=(batch_size, windowed_height, windowed_width, kernel_height, kernel_width, image_channels),
        strides=(
            stride_batch_size,
            stride_block_height,
            stride_block_width,
            stride_image_height,
            stride_image_width,
            stride_image_channels,
        ),
    )


def img2col(
    images: np.ndarray,
    kernel_size: Union[int, tuple[int]] = 3,
    stride: Union[int, tuple[int]] = 1,
    padding: Union[int, tuple[int]] = 0,
    channel_first: bool = True,
) -> np.ndarray:
    """
    Convert images to columns for convolution.

    Parameters
    ----------
    images: np.ndarray, shape=(batch_size, height, width, channels)
        The input images.
    kernel_size: int or tuple of ints, (kernel_height, kernel_width)
        The size of the convolution kernel.
    stride: int or tuple of ints, (stride_height, stride_width)
        The stride of the convolution, usually stride_height=string_width
    padding: int or tuple of ints, (padding_height, padding_width)
        The padding value of the convolution, usually padding_height=padding_width
    channel_first: bool
        Whether the images are in channel first format.

    Returns
    -------
    np.ndarray, shape=(batch_size, kernel_height * kernel_width * channels, output_height * output_width) # noqa: E501
        The columns of the input images.
    """

    windowed_images = window_images(
        images=images, kernel_size=kernel_size, stride=stride, padding=padding, channels_first=channel_first
    )
    if channel_first:
        batch_size, image_channel, windowed_height, windowed_width, kernel_height, kernel_width = windowed_images.shape
    else:
        batch_size, windowed_height, windowed_width, kernel_height, kernel_width, image_channel = windowed_images.shape
    return windowed_images.reshape(
        (batch_size * windowed_height * windowed_width, kernel_height * kernel_width * image_channel)
    )


# ruff: noqa: PLR0913
def pooling(
    images: np.ndarray,
    kernel_size: Union[int, tuple[int]] = 2,
    stride: Union[int, tuple[int]] = 2,
    padding: Union[int, tuple[int]] = 0,
    channel_first: bool = True,
    mode: str = "max",
) -> np.ndarray:
    """
    Pooling operation for images.

    Parameters
    ----------
    images: np.ndarray
        The input images.
    kernel_size: int or tuple of ints, (kernel_height, kernel_width)
        The size of the pooling kernel.
    stride: int or tuple of ints, (stride_height, stride_width)
        The stride of the pooling, usually stride_height=string_width
    padding: int or tuple of ints, (padding_height, padding_width)
        The padding value of the pooling, usually padding_height=padding_width
    channel_first: bool
        Whether the images are in channel first format.
    mode: str
        The pooling mode, either "max" or "mean".

    Returns
    -------
    np.ndarray
        The output of the pooling layer.
    """
    assert mode in {"max", "mean"}, "Pooling mode must be either 'max' or 'mean'."
    windowed_images = window_images(
        images=images,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        channels_first=channel_first,
    )

    if channel_first:
        if mode == "mean":
            return np.nanmean(windowed_images, axis=(4, 5))
        return np.nanmax(windowed_images, axis=(4, 5))

    if mode == "mean":
        return np.nanmean(windowed_images, axis=(3, 4))
    return np.nanmax(windowed_images, axis=(3, 4))


def relu(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, np.finfo(x.dtype).max)


def sobel(size=3, axis=0):
    """
    Generate a sobel filter kernel.
    https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size

    Parameters
    ----------
    size: int
        The size of the kernel.
    axis: int
        The axis of the kernel, 0 for vertical, 1 for horizontal.

    Returns
    -------
    np.ndarray, shape=(size, size)
        The sobel filter kernel.
    """
    k = np.zeros((size, size))
    x = np.arange(size)

    t = (size - 1) // 2
    for j, i in np.array(np.meshgrid(x, x)).T.reshape(-1, 2):
        i_, j_ = i - t, j - t
        if i_ == 0 and j_ == 0:
            continue
        k[j, i] = (i_ if axis == 0 else j_) / (i_ * i_ + j_ * j_)
    return k


def load_sample_filters(size, channel=1, sigma=1):
    """
    Load sample filters, including vertical, horizontal, diagonal and gaussian filters.

    Parameters
    ----------
    size: int
        The size of the filters.
    channel: int
        The number of channels of the filters.
    sigma: float
        The sigma of the gaussian filter.

    Returns
    -------
    dict
        vertical: np.ndarray, shape=(size, size, channel)
            The vertical filter.
        horizontal: np.ndarray, shape=(size, size, channel)
            The horizontal filter.
        diagonal: np.ndarray, shape=(size, size, channel)
            The diagonal filter.
        gaussian: np.ndarray, shape=(size, size, channel)
            The gaussian filter.
    """

    def expand_channels(f):
        # expand the filter to multiple channels
        return np.repeat(np.expand_dims(f, 2), channel, axis=2) if channel > 1 else f

    vertical_filter = sobel(size=size, axis=0)
    vertical_filter = expand_channels(vertical_filter)

    horizontal_filter = sobel(size=size, axis=1)
    horizontal_filter = expand_channels(horizontal_filter)

    flattop_filter = signal.windows.flattop(size)
    flattop_filter = np.outer(flattop_filter, flattop_filter)
    flattop_filter = expand_channels(flattop_filter)

    gaussian_filter = signal.windows.gaussian(size, sigma)
    gaussian_filter = np.outer(gaussian_filter, gaussian_filter)
    gaussian_filter = expand_channels(gaussian_filter)

    return {
        "vertical": vertical_filter,
        "horizontal": horizontal_filter,
        "flattop": flattop_filter,
        "gaussian": gaussian_filter,
    }


def max_pool2d_with_grad(
    x: np.ndarray,
    kernel_size: Union[int, tuple[int]] = 2,
    stride: Union[int, tuple[int]] = 2,
    padding: Union[int, tuple[int]] = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Max pooling operation for images.

    Parameters
    ----------
    x: np.ndarray
        The input images of shape (batch_size, channels, height, width).
    kernel_size: int or tuple of ints, (kernel_height, kernel_width)
        The size of the pooling kernel.
    stride: int or tuple of ints, (stride_height, stride_width)
        The stride of the pooling, usually stride_height=string_width
    padding: int or tuple of ints, (padding_height, padding_width)
        The padding value of the pooling, usually padding_height=padding_width

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The output and gradient of the pooling layer.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    windowed_x = window_images(x, kernel_size=kernel_size, stride=stride, padding=padding)
    pooling_x = np.nanmax(windowed_x, axis=(4, 5))

    batch_size, channels, height_blocks, width_blocks, kernel_height, kernel_width = windowed_x.shape
    block_size, kernel_size = height_blocks * width_blocks, kernel_height * kernel_width
    # backward
    reshaped_windowed_x = windowed_x.reshape((batch_size, channels, block_size, kernel_size))
    # reshaped_windowed_x_argmax: (batch_size, channels, block_size)
    reshaped_windowed_x_argmax = np.argmax(reshaped_windowed_x, axis=3)

    # indices: (batch_size, channels, block_size, 3)
    indices = np.stack(np.indices((batch_size, channels, block_size)), axis=-1)
    reshaped_maximum_locations = np.zeros_like(reshaped_windowed_x)

    reshaped_maximum_locations[indices[..., 0], indices[..., 1], indices[..., 2], reshaped_windowed_x_argmax] = 1
    max_windowed_x_index = reshaped_maximum_locations.reshape(windowed_x.shape)

    stride_height, stride_width = stride
    grad_height, grad_width = height_blocks * stride_height, width_blocks * stride_width
    grad = max_windowed_x_index.transpose((0, 1, 2, 4, 3, 5)).reshape((batch_size, channels, grad_height, grad_width))

    # check padding
    batch_size, channels, height, width = x.shape
    padding_height, padding_width = height - grad_height, width - grad_width
    if padding_height > 0 or padding_width > 0:
        grad = np.pad(grad, ((0, 0), (0, 0), (0, padding_height), (0, padding_width)))
    return pooling_x, grad


def avg_pool2d_with_grad(
    x: np.ndarray,
    kernel_size: Union[int, tuple[int]] = 2,
    stride: Union[int, tuple[int]] = 2,
    padding: Union[int, tuple[int]] = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average pooling operation for images.

    Parameters
    ----------
    x: np.ndarray
        The input images of shape (batch_size, channels, height, width).
    kernel_size: int or tuple of ints, (kernel_height, kernel_width)
        The size of the pooling kernel.
    stride: int or tuple of ints, (stride_height, stride_width)
        The stride of the pooling, usually stride_height=string_width
    padding: int or tuple of ints, (padding_height, padding_width)
        The padding value of the pooling, usually padding_height=padding_width

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The output and gradient of the pooling layer.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    windowed_x = window_images(x, kernel_size=kernel_size, stride=stride, padding=padding)
    pooling_x = np.nanmean(windowed_x, axis=(4, 5))

    batch_size, channels, height_blocks, width_blocks, kernel_height, kernel_width = windowed_x.shape

    stride_height, stride_width = stride
    grad_height, grad_width = height_blocks * stride_height, width_blocks * stride_width
    grad = np.ones((batch_size, channels, grad_height, grad_width)) / (kernel_height * kernel_width)
    # check padding
    *_, height, width = x.shape
    padding_height, padding_width = height - grad_height, width - grad_width
    if padding_height > 0 or padding_width > 0:
        grad = np.pad(grad, ((0, 0), (0, 0), (0, padding_height), (0, padding_width)))
    return pooling_x, grad
