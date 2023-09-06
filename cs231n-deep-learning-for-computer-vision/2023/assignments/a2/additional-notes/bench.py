# ruff: noqa: PLR2004
import numpy as np
from scipy import signal


def generate_same_padding(kernel_size: int) -> int:
    """
    Generate padding for same convolution with given kernel size when stride is 1.

    Parameters
    ----------
    kernel_size : int
        The size of the kernel.

    Returns
    -------
    int
        The padding to use for same convolution.
    """
    return (kernel_size - 1) // 2


def _pad_same_crgb(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    *_, kernel_height, kernel_width = w.shape
    padding_height = generate_same_padding(kernel_height)
    padding_width = generate_same_padding(kernel_width)

    if x.ndim == 4:
        return np.pad(x, ((0, 0), (0, 0), (padding_height, padding_height), (padding_width, padding_width)))
    if x.ndim == 3:
        return np.pad(x, ((0, 0), (padding_height, padding_height), (padding_width, padding_width)))
    if x.ndim == 2:
        return np.pad(x, ((padding_height, padding_height), (padding_width, padding_width)))
    return x


def _pad_same_rgbc(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    kernel_height, kernel_width = None, None
    if w.ndim == 4:
        _, kernel_height, kernel_width, _ = w.shape
    if w.ndim == 3:
        kernel_height, kernel_width, _ = w.shape

    assert kernel_height is not None and kernel_width is not None, "Kernel must be 3D or 4D."
    padding_height = generate_same_padding(kernel_height)
    padding_width = generate_same_padding(kernel_width)

    if x.ndim == 4:
        return np.pad(x, ((0, 0), (padding_height, padding_height), (padding_width, padding_width), (0, 0)))
    if x.ndim == 3:
        return np.pad(x, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)))
    return x


class Convolution:
    """
    Convolution class for different methods of convolution (naive, fft, tensordot, einsum, img2col).
    Just for illustration CNN purposes, not for production use.
    """

    def __init__(self, padding: str = "same", method: str = "img2col", channel: tuple[None, str] = None):
        """
        Initialize the convolution class.

        Parameters
        ----------
        padding : str, optional, default="same"
            The padding of the convolution.
        method : str, optional, default="img2col"
            The method to use for convolution.
        channel : tuple[None, str], optional, default=None
            The channel to use for convolution.
        """
        assert padding in {"same", "valid"}, "Padding must be either 'same' or 'valid'."
        assert method in {
            "naive",
            "fft",
            "tensordot",
            "einsum",
            "img2col",
        }, "Method must be one of {'naive', 'fft', 'tensordot', 'einsum', 'img2col'}."
        assert channel in {None, "first", "last"}, "Channel must be one of {None, 'first', 'last'}."
        self.padding = padding
        self.method = method
        self.channel = channel
        self._is_rgb = channel is not None

    def __str__(self):
        return f"{self.__class__.__name__}(padding={self.padding}, method={self.method}, channel={self.channel})"

    def __repr__(self):
        return self.__str__()

    def __call__(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Perform convolution on the given input.

        Parameters
        ----------
        x : np.ndarray
            The input to convolve.
        w : np.ndarray
            The kernel to convolve with.

        Returns
        -------
        np.ndarray
            The convolved output.
        """
        if self.padding == "same":
            if self.channel == "last":
                x = _pad_same_rgbc(x, w)
            else:
                x = _pad_same_crgb(x, w)

        if self._is_rgb:
            return self._conv_rgb(x, w)
        return self._conv_gray(x, w)

    def _pad_same_rgb(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        if self.channel == "first":
            return _pad_same_crgb(x, w)
        return _pad_same_rgbc(x, w)

    def _conv_gray(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        functions = {}
        if x.ndim == 3 and w.ndim == 3:
            functions = {
                "naive": _conv_gray_full_naive,
                "fft": _conv_gray_full_fft,
                "tensordot": _conv_gray_full_tensordot,
                "einsum": _conv_gray_full_einsum,
                "img2col": _conv_gray_full_img2col,
            }
        if x.ndim == 3 and w.ndim == 2:
            functions = {
                "naive": _conv_gray_multi_naive,
                "fft": _conv_gray_multi_fft,
                "tensordot": _conv_gray_multi_tensordot,
                "einsum": _conv_gray_multi_einsum,
                "img2col": _conv_gray_multi_img2col,
            }
        if x.ndim == 2 and w.ndim == 2:
            functions = {
                "naive": _conv_gray_single_naive,
                "fft": _conv_gray_single_fft,
                "tensordot": _conv_gray_single_tensordot,
                "einsum": _conv_gray_single_einsum,
                "img2col": _conv_gray_single_img2col,
            }
        assert self.method in functions, "Method not implemented for given input and kernel dimensions."
        return functions[self.method](x, w)

    def _conv_rgb(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        if self.channel == "first":
            return self._conv_crgb(x, w)
        return self._conv_rgbc(x, w)

    def _conv_rgbc(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        functions = {}
        if x.ndim == 4 and w.ndim == 4:
            functions = {
                "naive": _conv_rgbc_full_naive,
                "fft": _conv_rgbc_full_fft,
                "tensordot": _conv_rgbc_full_tensordot,
                "einsum": _conv_rgbc_full_einsum,
                "img2col": _conv_rgbc_full_img2col,
            }
        if x.ndim == 4 and w.ndim == 3:
            functions = {
                "naive": _conv_rgbc_multi_naive,
                "fft": _conv_rgbc_multi_fft,
                "tensordot": _conv_rgbc_multi_tensordot,
                "einsum": _conv_rgbc_multi_einsum,
                "img2col": _conv_rgbc_multi_img2col,
            }
        if x.ndim == 3 and w.ndim == 3:
            functions = {
                "naive": _conv_rgbc_single_naive,
                "fft": _conv_rgbc_single_fft,
                "tensordot": _conv_rgbc_single_tensordot,
                "einsum": _conv_rgbc_single_einsum,
                "img2col": _conv_rgbc_single_img2col,
            }
        assert self.method in functions, "Method not implemented for given input and kernel dimensions."
        return functions[self.method](x, w)

    def _conv_crgb(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        functions = {}
        if x.ndim == 4 and w.ndim == 4:
            functions = {
                "naive": _conv_crgb_full_naive,
                "fft": _conv_crgb_full_fft,
                "tensordot": _conv_crgb_full_tensordot,
                "einsum": _conv_crgb_full_einsum,
                "img2col": _conv_crgb_full_img2col,
            }
        if x.ndim == 4 and w.ndim == 3:
            functions = {
                "naive": _conv_crgb_multi_naive,
                "fft": _conv_crgb_multi_fft,
                "tensordot": _conv_crgb_multi_tensordot,
                "einsum": _conv_crgb_multi_einsum,
                "img2col": _conv_crgb_multi_img2col,
            }
        if x.ndim == 3 and w.ndim == 3:
            functions = {
                "naive": _conv_crgb_single_naive,
                "fft": _conv_crgb_single_fft,
                "tensordot": _conv_crgb_single_tensordot,
                "einsum": _conv_crgb_single_einsum,
                "img2col": _conv_crgb_single_img2col,
            }
        assert self.method in functions, "Method not implemented for given input and kernel dimensions."
        return functions[self.method](x, w)


"""
====================================================================================================
gray scale image convolutions
====================================================================================================
"""


"""
----------------------------------------------------------------------------------------------------
single instance single kernel convolutions
----------------------------------------------------------------------------------------------------
"""


def _conv_gray_single_naive(x, w):
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=w.shape)
    return (windows * np.expand_dims(w, axis=(0, 1))).sum(axis=(2, 3))


def _conv_gray_single_fft(x, w, mode="valid"):
    return signal.convolve(x, np.flip(w), mode=mode)


def _conv_gray_single_tensordot(x, w):
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=w.shape)
    return np.tensordot(windows, w, axes=((2, 3), (0, 1)))


def _conv_gray_single_einsum(x, w):
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=w.shape)
    return np.einsum("ijkl,kl->ij", windows, w, optimize="greedy")


def _conv_gray_single_img2col(x, w):
    image_height, image_width = x.shape
    kernel_height, kernel_width = w.shape

    windowed_height, windowed_width = image_height - kernel_height + 1, image_width - kernel_width + 1
    windows = np.lib.stride_tricks.as_strided(
        x=x, shape=(*w.shape, windowed_height, windowed_width), strides=(*x.strides, *x.strides)
    )

    kernel_size = kernel_height * kernel_width
    return (w.reshape((-1, kernel_size)) @ windows.reshape((kernel_size, -1))).reshape(
        (windowed_height, windowed_width)
    )


"""
----------------------------------------------------------------------------------------------------
multi instance single kernel convolutions
----------------------------------------------------------------------------------------------------
"""


def _conv_gray_multi_naive(x, w):
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=w.shape, axis=(1, 2))
    return (windows * np.expand_dims(w, axis=(0, 1, 2))).sum(axis=(3, 4))


def _conv_gray_multi_fft(x, w, mode="valid"):
    return signal.convolve(x, np.expand_dims(np.flip(w), axis=0), mode=mode)


def _conv_gray_multi_tensordot(x, w):
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=w.shape, axis=(1, 2))
    return np.tensordot(windows, w, axes=((3, 4), (0, 1)))


def _conv_gray_multi_einsum(x, w):
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=w.shape, axis=(1, 2))
    return np.einsum("nijkl,kl->nij", windows, w, optimize="greedy")


def _conv_gray_multi_img2col(x, w):
    batch_size, image_height, image_width = x.shape
    kernel_height, kernel_width = w.shape

    windowed_height, windowed_width = image_height - kernel_height + 1, image_width - kernel_width + 1
    stride_batch_size, stride_image_height, stride_image_width = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(*w.shape, batch_size, windowed_height, windowed_width),
        strides=(stride_image_height, stride_image_width, *x.strides),
    )

    kernel_size = kernel_height * kernel_width
    return (w.reshape((-1, kernel_size)) @ windows.reshape((kernel_size, -1))).reshape(
        (batch_size, windowed_height, windowed_width)
    )


"""
----------------------------------------------------------------------------------------------------
multi instance multi kernel convolutions
----------------------------------------------------------------------------------------------------
"""


def _conv_gray_full_naive(x, w):
    _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return (np.expand_dims(windows, axis=1) * np.expand_dims(w, axis=(0, 2, 3))).sum(axis=(4, 5))


def _conv_gray_full_fft(x, w, mode="valid"):
    w = np.flip(w, axis=(1, 2))
    w = np.expand_dims(w, axis=0)  # (1 , ci, hw, ww)
    x = np.expand_dims(x, axis=1)  # (n , 1 , hi, wi)
    return signal.fftconvolve(x, w, axes=(2, 3), mode=mode)


def _conv_gray_full_tensordot(x, w):
    _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return np.tensordot(windows, w, axes=((3, 4), (1, 2))).transpose((0, 3, 1, 2))


def _conv_gray_full_einsum(x, w):
    _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return np.einsum("nijkl,mkl->nmij", windows, w, optimize="greedy")


def _conv_gray_full_img2col(x, w):
    batch_size, image_height, image_width = x.shape
    _, kernel_height, kernel_width = w.shape

    windowed_height, windowed_width = image_height - kernel_height + 1, image_width - kernel_width + 1
    stride_batch_size, stride_image_height, stride_image_width = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(kernel_height, kernel_width, batch_size, windowed_height, windowed_width),
        strides=(stride_image_height, stride_image_width, *x.strides),
    )

    kernel_size = kernel_height * kernel_width
    return (
        (w.reshape((-1, kernel_size)) @ windows.reshape((kernel_size, -1)))
        .reshape((-1, batch_size, windowed_height, windowed_width))
        .transpose((1, 0, 2, 3))
    )


"""
====================================================================================================
rgb (channel first) scale image convolutions
====================================================================================================
"""

"""
----------------------------------------------------------------------------------------------------
single instance single kernel convolutions
----------------------------------------------------------------------------------------------------
"""


def _conv_crgb_single_naive(x, w):
    _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return (windows * np.expand_dims(w, axis=(1, 2))).sum(axis=(0, 3, 4))


def _conv_crgb_single_fft(x, w, mode="valid"):
    w = np.flip(w, axis=(0, 1, 2))
    return signal.convolve(x, w, mode=mode).squeeze(axis=0)


def _conv_crgb_single_tensordot(x, w):
    _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return np.tensordot(windows, w, axes=((0, 3, 4), (0, 1, 2)))


def _conv_crgb_single_einsum(x, w):
    _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return np.einsum("cijkl,ckl->ij", windows, w, optimize="greedy")


def _conv_crgb_single_img2col(x, w):
    in_channels, image_height, image_width = x.shape
    _, kernel_height, kernel_width = w.shape

    windowed_height, windowed_width = image_height - kernel_height + 1, image_width - kernel_width + 1

    stride_image_channels, stride_image_height, stride_image_width = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(*w.shape, windowed_height, windowed_width),
        strides=(*x.strides, stride_image_height, stride_image_width),
    )

    kernel_size = in_channels * kernel_height * kernel_width
    return (w.reshape((-1, kernel_size)) @ windows.reshape((kernel_size, -1))).reshape(
        (windowed_height, windowed_width)
    )


"""
----------------------------------------------------------------------------------------------------
multi instance single kernel convolutions
----------------------------------------------------------------------------------------------------
"""


def _conv_crgb_multi_naive(x, w):
    _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(2, 3))
    return (windows * np.expand_dims(w, axis=(0, 2, 3))).sum(axis=(1, 4, 5))


def _conv_crgb_multi_fft(x, w, mode="valid"):
    w = np.flip(w, axis=(0, 1, 2))
    return signal.fftconvolve(x, np.expand_dims(w, axis=0), axes=(1, 2, 3), mode=mode).squeeze(axis=1)


def _conv_crgb_multi_tensordot(x, w):
    _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(2, 3))
    return np.tensordot(windows, w, axes=((1, 4, 5), (0, 1, 2)))


def _conv_crgb_multi_einsum(x, w):
    _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(2, 3))
    return np.einsum("ncijkl,ckl->nij", windows, w, optimize="greedy")


def _conv_crgb_multi_img2col(x, w):
    batch_size, in_channels, image_height, image_width = x.shape
    _, kernel_height, kernel_width = w.shape

    windowed_height, windowed_width = image_height - kernel_height + 1, image_width - kernel_width + 1

    stride_batch_size, stride_image_channels, stride_image_height, stride_image_width = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(in_channels, kernel_height, kernel_width, batch_size, windowed_height, windowed_width),
        strides=(
            stride_image_channels,
            stride_image_height,
            stride_image_width,
            stride_batch_size,
            stride_image_height,
            stride_image_width,
        ),
    )

    kernel_size = in_channels * kernel_height * kernel_width
    return (w.reshape((-1, kernel_size)) @ windows.reshape((kernel_size, -1))).reshape(
        (batch_size, windowed_height, windowed_width)
    )


"""
----------------------------------------------------------------------------------------------------
multi instance multi kernel convolutions
----------------------------------------------------------------------------------------------------
"""


def _conv_crgb_full_naive(x, w):
    _, _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(2, 3))
    return (np.expand_dims(windows, axis=1) * np.expand_dims(w, axis=(0, 3, 4))).sum(axis=(2, 5, 6))


def _conv_crgb_full_fft(x, w, mode="valid"):
    w = np.flip(w, axis=(1, 2, 3))
    w = np.expand_dims(w, axis=0)  # (1 , co, ci, hw, ww)
    x = np.expand_dims(x, axis=1)  # (n , 1 , ci, hi, wi)
    return signal.fftconvolve(x, w, axes=(2, 3, 4), mode=mode).squeeze(axis=2)


def _conv_crgb_full_tensordot(x, w):
    _, _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(2, 3))
    return np.tensordot(windows, w, axes=((1, 4, 5), (1, 2, 3))).transpose((0, 3, 1, 2))


def _conv_crgb_full_einsum(x, w):
    _, _, kernel_height, kernel_width = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(2, 3))
    return np.einsum("ncijkl,mckl->nmij", windows, w, optimize="greedy")


def _conv_crgb_full_img2col(x, w):
    batch_size, in_channels, image_height, image_width = x.shape
    _, _, kernel_height, kernel_width = w.shape

    windowed_height, windowed_width = image_height - kernel_height + 1, image_width - kernel_width + 1

    stride_batch_size, stride_image_channels, stride_image_height, stride_image_width = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(in_channels, kernel_height, kernel_width, batch_size, windowed_height, windowed_width),
        strides=(
            stride_image_channels,
            stride_image_height,
            stride_image_width,
            stride_batch_size,
            stride_image_height,
            stride_image_width,
        ),
    )

    kernel_size = in_channels * kernel_height * kernel_width
    return (
        (w.reshape((-1, kernel_size)) @ windows.reshape((kernel_size, -1)))
        .reshape((-1, batch_size, windowed_height, windowed_width))
        .transpose((1, 0, 2, 3))
    )


"""
====================================================================================================
rgb (channel last) scale image convolutions
====================================================================================================
"""

"""
----------------------------------------------------------------------------------------------------
single instance single kernel convolutions
----------------------------------------------------------------------------------------------------
"""


def _conv_rgbc_single_naive(x, w):
    kernel_height, kernel_width, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(0, 1))
    return (windows * np.expand_dims(w.transpose((2, 0, 1)), axis=(0, 1))).sum(axis=(2, 3, 4))


def _conv_rgbc_single_fft(x, w, mode="valid"):
    w = np.flip(w, axis=(0, 1, 2))
    return signal.convolve(x, w, mode=mode).squeeze(axis=2)


def _conv_rgbc_single_tensordot(x, w):
    kernel_height, kernel_width, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(0, 1))
    return np.tensordot(windows, w, axes=((3, 4, 2), (0, 1, 2)))


def _conv_rgbc_single_einsum(x, w):
    kernel_height, kernel_width, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(0, 1))
    return np.einsum("ijckl,klc->ij", windows, w, optimize="greedy")


def _conv_rgbc_single_img2col(x, w):
    image_height, image_width, in_channels = x.shape
    kernel_height, kernel_width, _ = w.shape

    windowed_height, windowed_width = image_height - kernel_height + 1, image_width - kernel_width + 1

    stride_image_height, stride_image_width, stride_image_channels = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(*w.shape, windowed_height, windowed_width),
        strides=(*x.strides, stride_image_height, stride_image_width),
    )

    kernel_size = in_channels * kernel_height * kernel_width
    return (w.reshape((-1, kernel_size)) @ windows.reshape((kernel_size, -1))).reshape(
        (windowed_height, windowed_width)
    )


"""
----------------------------------------------------------------------------------------------------
multi instance single kernel convolutions
----------------------------------------------------------------------------------------------------
"""


def _conv_rgbc_multi_naive(x, w):
    kernel_height, kernel_width, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return (windows * np.expand_dims(w.transpose((2, 0, 1)), axis=(0, 1, 2))).sum(axis=(3, 4, 5))


def _conv_rgbc_multi_fft(x, w, mode="valid"):
    w = np.flip(w, axis=(0, 1, 2))
    return signal.fftconvolve(x, np.expand_dims(w, axis=0), axes=(1, 2, 3), mode=mode).squeeze(axis=3)


def _conv_rgbc_multi_tensordot(x, w):
    kernel_height, kernel_width, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return np.tensordot(windows, w, axes=((3, 4, 5), (2, 0, 1)))


def _conv_rgbc_multi_einsum(x, w):
    kernel_height, kernel_width, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return np.einsum("nijckl,klc->nij", windows, w, optimize="greedy")


def _conv_rgbc_multi_img2col(x, w):
    batch_size, image_height, image_width, in_channels = x.shape
    kernel_height, kernel_width, _ = w.shape

    windowed_height, windowed_width = image_height - kernel_height + 1, image_width - kernel_width + 1

    stride_batch_size, stride_image_height, stride_image_width, stride_image_channels = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(kernel_height, kernel_width, in_channels, batch_size, windowed_height, windowed_width),
        strides=(
            stride_image_height,
            stride_image_width,
            stride_image_channels,
            stride_batch_size,
            stride_image_height,
            stride_image_width,
        ),
    )

    kernel_size = in_channels * kernel_height * kernel_width
    return (w.reshape((-1, kernel_size)) @ windows.reshape((kernel_size, -1))).reshape(
        (batch_size, windowed_height, windowed_width)
    )


"""
----------------------------------------------------------------------------------------------------
multi instance multi kernel convolutions
----------------------------------------------------------------------------------------------------
"""


def _conv_rgbc_full_naive(x, w):
    _, kernel_height, kernel_width, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return (np.expand_dims(windows, axis=1) * np.expand_dims(w.transpose((0, 3, 1, 2)), axis=(0, 2, 3))).sum(
        axis=(4, 5, 6)
    )


def _conv_rgbc_full_fft(x, w, mode="valid"):
    w = np.flip(w, axis=(1, 2, 3))
    w = np.expand_dims(w, axis=0)  # (1 , co, ci, hw, ww)
    x = np.expand_dims(x, axis=1)  # (n , 1 , ci, hi, wi)
    return signal.fftconvolve(x, w, axes=(2, 3, 4), mode=mode).squeeze(axis=4)


def _conv_rgbc_full_tensordot(x, w):
    _, kernel_height, kernel_width, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return np.tensordot(windows, w, axes=((3, 4, 5), (3, 1, 2))).transpose((0, 3, 1, 2))


def _conv_rgbc_full_einsum(x, w):
    _, kernel_height, kernel_width, _ = w.shape
    windows = np.lib.stride_tricks.sliding_window_view(x, window_shape=(kernel_height, kernel_width), axis=(1, 2))
    return np.einsum("nijckl,mklc->nmij", windows, w, optimize="greedy")


def _conv_rgbc_full_img2col(x, w):
    batch_size, image_height, image_width, in_channels = x.shape
    _, kernel_height, kernel_width, _ = w.shape

    windowed_height, windowed_width = image_height - kernel_height + 1, image_width - kernel_width + 1

    stride_batch_size, stride_image_height, stride_image_width, stride_image_channels = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(kernel_height, kernel_width, in_channels, batch_size, windowed_height, windowed_width),
        strides=(
            stride_image_height,
            stride_image_width,
            stride_image_channels,
            stride_batch_size,
            stride_image_height,
            stride_image_width,
        ),
    )

    kernel_size = in_channels * kernel_height * kernel_width
    return (
        (w.reshape((-1, kernel_size)) @ windows.reshape((kernel_size, -1)))
        .reshape((-1, batch_size, windowed_height, windowed_width))
        .transpose((1, 0, 2, 3))
    )
