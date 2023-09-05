import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.reshape(x.shape[0], -1) @ w + b  # (N, D) @ (D, M) + (M,) = (N, M)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout @ w.T                          # (N, M) @ (M, D) = (N, D)
    dx = dx.reshape(x.shape)                 # (N, D) -> (N, d1, ..., d_k)
    dw = x.reshape(x.shape[0], -1).T @ dout  # (D, N) @ (N, M) = (D, M)
    db = np.sum(dout, axis=0)                # (M,)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = (x > 0).astype(int) * dout  # (N, D) * (N, D) = (N, D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = len(x)
    x = x - np.max(x, axis=1, keepdims=True)  # (N, C), for numerical stability
    x = np.exp(x)                             # (N, C)
    x /= np.sum(x, axis=1, keepdims=True)     # (N, C)
    loss = -np.sum(np.log(x[np.arange(num_train), y])) / num_train

    x[np.arange(num_train), y] -= 1
    dx = x / num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mean = np.mean(x, axis=0, keepdims=True)  # (1, D)
        var = np.var(x, axis=0, keepdims=True)    # (1, D)
        std = np.sqrt(var + eps)                  # (1, D)
        x_hat = (x - mean) / std                  # (N, D)
        out = gamma * x_hat + beta                # (N, D)

        running_mean = momentum * running_mean + (1 - momentum) * mean   # (1, D)
        running_var = momentum * running_var + (1 - momentum) * var      # (1, D)

        cache = (x, x_hat, mean, std, gamma)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_hat = (x - running_mean) / np.sqrt(running_var + eps)  # (N, D)
        out = gamma * x_hat + beta                               # (N, D)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_hat, mean, std, gamma = cache
    m = x.shape[0]

    dgamma = np.sum(dout * x_hat, axis=0)  # (D,)
    dbeta = np.sum(dout, axis=0)           # (D,)
    dx_hat = dout * gamma                  # (N, D)

    dx_hat_dvar = (x - mean) * -0.5 * std**(-3)     # (N, D)
    dx_hat_dmean = -1 / std                         # (N, D)
    dmean_dx = 1 / m                                # scalar
    dvar_dx = 2 / m * (x - mean)                    # (N, D)
    dx_hat_dx = 1 / std                             # (N, D)
    dx = dx_hat * dx_hat_dx + np.sum(dx_hat * dx_hat_dvar, axis=0) * dvar_dx + np.sum(dx_hat * dx_hat_dmean, axis=0) * dmean_dx  # (N, D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_hat, mean, std, gamma = cache
    m = x.shape[0]

    dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)  # (D, 1)
    dbeta = np.sum(dout, axis=0, keepdims=True)           # (D, 1)
    dx_hat = dout * gamma                                 # (N, D)

    # dvar = -0.5 * np.sum(dx_hat * x_hat / std**2, axis=0)      # (D,)
    # dmean = - np.sum(dx_hat / std, axis=0)                     # (D,)
    # dx = dx_hat / std + dvar * 2 * (x - mean) / m + dmean / m  # (N, D)

    dx = 1 / (m * std) * (m * dx_hat - x_hat * np.sum(dx_hat*x_hat, axis=0, keepdims=True) - np.sum(dx_hat, axis=0, keepdims=True))  # (N, D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    axis = ln_param.get("axis", 0)
    shape = ln_param.get("shape", x.shape)

    mean = np.mean(x, axis=1, keepdims=True)  # (N, 1)
    var = np.var(x, axis=1, keepdims=True)    # (N, 1)
    std = np.sqrt(var + eps)                  # (N, 1)
    x_hat = (x - mean) / std                  # (N, D)
    out = gamma * x_hat + beta                # (N, D)
    cache = (x_hat, std, gamma, axis, shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_hat, std, gamma, axis, shape = cache
    m = x_hat.shape[1]
    dgamma = np.sum((dout * x_hat).reshape(shape), axis=axis, keepdims=True)
    dbeta = np.sum(dout.reshape(shape), axis=axis, keepdims=True)
    dx_hat = dout * gamma

    dx = 1 / (m * std) * (m * dx_hat - x_hat * np.sum(dx_hat*x_hat, axis=1, keepdims=True) - np.sum(dx_hat, axis=1, keepdims=True))  # (N, D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p  # (N, D)
        out = x * mask                             # (N, D)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def generate_output_size(image_size, kernel_size, stride, padding):
        return (image_size - kernel_size + 2 * padding) // stride + 1

    N, C, H, W, = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant")  # (N, C, H+2*pad, W+2*pad)

    H_out = generate_output_size(H, HH, stride, pad)
    W_out = generate_output_size(W, WW, stride, pad)

    # 1. step by step for loop
    # out = np.empty((N, F, H_out, W_out))
    # for n in range(N):
    #     for f in range(F):
    #         for i in range(H_out):
    #             for j in range(W_out):
    #                 h_start, w_start = i*stride, j*stride
    #                 out[n, f, i, j] = np.sum(x_pad[n, :, h_start:h_start+HH, w_start:w_start+WW] * w[f]) + b[f]

    # 2. vectorized with `np.lib.stride_tricks.as_strided`
    N_s, C_s, H_s, W_s = x_pad.strides
    windowed_x = np.lib.stride_tricks.as_strided(
        x=x_pad,
        shape=(N, C, H_out, W_out, HH, WW),
        strides=(N_s, C_s, stride * H_s, stride * W_s, H_s, W_s)
    )
    out = np.einsum("nchwij,fcij->nfhw", windowed_x, w, optimize="greedy") + b[None, :, None, None]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def calculate_pad_for_backward(image_size, output_size, kernel_size):
        return (image_size - 1 + kernel_size - output_size) // 2

    def dial(x, k=2):
        n, c, h, w = x.shape
        o = np.zeros((n, c, h * k - 1, w * k - 1), dtype=x.dtype)
        o[:, :, ::k, ::k] = x
        return o

    x, w, b, conv_param = cache

    N, C, H, W, = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # ================== dw ==================
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant")  # (N, C, H+2*pad, W+2*pad)

    # 1. step by step for loop
    # dw = np.empty_like(w)  # (F, C, HH, WW)
    # for f in range(F):
    #     for c in range(C):
    #         for i in range(HH):
    #             for j in range(WW):
    #                 h_start, w_start = i*stride, j*stride
    #                 dw[f, c, i, j] = np.sum(x_pad[:, c, h_start:h_start+H_out, w_start:w_start+W_out] * dout[:, f, :, :])

    # 2. vectorized with `np.lib.stride_tricks.as_strided`
    N_s, C_s, H_s, W_s = x_pad.strides
    windowed_x = np.lib.stride_tricks.as_strided(
        x=x_pad,
        shape=(N, C, H_out, W_out, HH, WW),
        strides=(N_s, C_s, stride * H_s, stride * W_s, H_s, W_s)
    )
    dw = np.einsum("ncijhw,nfij->fchw", windowed_x, dout, optimize="greedy")

    # ================== db ==================
    # 1. step by step for loop
    # db = np.empty_like(b) # (F,)
    # for f in range(F):
    #     db[f] = np.sum(dout[:, f, :, :])

    # 2. vectorized
    db = np.sum(dout, axis=(0, 2, 3))

    # ================== dx ==================
    dx = np.empty_like(x)            # (N, C, H, W)
    w_rot = np.flip(w, axis=(2, 3))  # (F, C, HH, WW)
    if stride > 1:
        dout = dial(dout, stride)

    _, _, H_out, W_out = dout.shape
    pad_w = calculate_pad_for_backward(W, W_out, WW)
    pad_h = calculate_pad_for_backward(H, H_out, HH)

    if pad_w > 0 or pad_h > 0:
        dout = np.pad(dout, ((0, 0), (0, 0), (pad_h, pad_w), (pad_h, pad_w)), "constant")

    # 1. step by step for loop
    # for n in range(N):
    #     for c in range(C):
    #         for i in range(H):
    #             for j in range(W):
    #                 dx[n, c, i, j] = np.sum(dout[n, :, i:i+HH, j:j+WW] * w_rot[:, c, :, :])
    #
    # 2. vectorized with `np.lib.stride_tricks.as_strided`
    N_s, C_s, H_s, W_s = dout.strides
    windowed_dout = np.lib.stride_tricks.as_strided(
        x=dout,
        shape=(N, F, H, W, HH, WW),
        strides=(N_s, C_s,  H_s, W_s, H_s, W_s)
    )
    dx = np.einsum("nfhwij,fcij->nchw", windowed_dout, w_rot, optimize="greedy")

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def generate_output_size(image_size, kernel_size, stride):
        return (image_size - kernel_size) // stride + 1

    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]

    H_out = generate_output_size(H, pool_height, stride)
    W_out = generate_output_size(W, pool_width, stride)

    # 1. step by step for loop
    # out = np.empty((N, C, H_out, W_out))
    # for n in range(N):
    #     for c in range(C):
    #         for i in range(H_out):
    #             for j in range(W_out):
    #                 h_start, w_start = i*stride, j*stride
    #                 out[n, c, i, j] = np.max(x[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width])

    # 2. vectorized with `np.lib.stride_tricks.as_strided`
    N_s, C_s, H_s, W_s = x.strides
    windowed_x = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(N, C, H_out, W_out, pool_height, pool_width),
        strides=(N_s, C_s, stride * H_s, stride * W_s, H_s, W_s)
    )
    out = np.max(windowed_x, axis=(4, 5))  # (N, C, H_out, W_out)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def generate_output_size(image_size, kernel_size, stride):
        return (image_size - kernel_size) // stride + 1

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]

    H_out = generate_output_size(H, pool_height, stride)
    W_out = generate_output_size(W, pool_width, stride)

    # 1. step by step for loop with duplicates
    # dx = np.zeros_like(x)
    # for n in range(N):
    #     for c in range(C):
    #         for i in range(H_out):
    #             for j in range(W_out):
    #                 h_start, w_start = i*stride, j*stride
    #                 window = x[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width]
    #                 dx[n, c, h_start:h_start+pool_height, w_start:w_start+pool_width] += dout[n, c, i, j] * (window == np.max(window))

    # 2. vectorized with `np.lib.stride_tricks.as_strided`
    N_s, C_s, H_s, W_s = x.strides
    windowed_x = np.lib.stride_tricks.as_strided(
        x=x,
        shape=(N, C, H_out, W_out, pool_height, pool_width),
        strides=(N_s, C_s, stride * H_s, stride * W_s, H_s, W_s)
    )
    N_s, C_s, H_s, W_s = dout.strides
    dout = np.lib.stride_tricks.as_strided(x=dout, shape=windowed_x.shape, strides=(N_s, C_s, H_s, W_s, 0, 0))

    # 2.1 with duplicates
    windowed_x_max_mask = windowed_x == np.max(windowed_x, axis=(4, 5), keepdims=True)  # (N, C, H_out, W_out, pool_height, pool_width)

    # 2.2 without duplicates
    # reshaped_windowed_x = windowed_x.reshape(N, C, H_out * W_out, pool_height * pool_width)
    # reshaped_windowed_x_argmax = np.argmax(reshaped_windowed_x, axis=3)
    # windowed_x_max_mask = np.zeros_like(reshaped_windowed_x)
    # windowed_x_max_mask[np.arange(N)[:, None, None], np.arange(C)[None, :, None], np.arange(H_out * W_out)[None, None, :], reshaped_windowed_x_argmax] = 1
    # windowed_x_max_mask = windowed_x_max_mask.reshape(N, C, H_out, W_out, pool_height, pool_width)

    dx = windowed_x_max_mask * dout           # (N, C, H_out, W_out, pool_height, pool_width)
    dx = dx.swapaxes(3, 4).reshape(x.shape)   # (N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W, = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # (N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W, = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # (N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_shape = x.shape
    N, C, H, W, = x.shape
    x = x.reshape(N * G, -1)                                  # (N * G, C // G * H * W)
    gamma = np.broadcast_to(gamma, x_shape).reshape(x.shape)  # (N * G, C // G * H * W)
    beta = np.broadcast_to(beta, x_shape).reshape(x.shape)    # (N * G, C // G * H * W)

    gn_param["axis"] = (0, 2, 3)
    gn_param["shape"] = x_shape

    out, cache = layernorm_forward(x, gamma, beta, gn_param)
    out = out.reshape(x_shape)                                # (N, C, H, W)

    cache = (G, cache)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    G, cache = cache
    dout_shape = dout.shape
    N, C, H, W, = dout_shape

    dout = dout.reshape(N * G, -1)                        # (N * G, C // G * H * W)

    dx, dgamma, dbeta = layernorm_backward(dout, cache)
    dx = dx.reshape(dout_shape)                           # (N, C, H, W)
    dgamma = np.expand_dims(dgamma, axis=(0, 2, 3))       # (1, C, 1, 1)
    dbeta = np.expand_dims(dbeta, axis=(0, 2, 3))         # (1, C, 1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
