from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)      # (C,)
        scores -= np.max(scores)  # (C,), for numerical stability
        scores = np.exp(scores)   # (C,)
        scores /= np.sum(scores)  # (C,)
        loss += -np.log(scores[y[i]])

        dW += X[i][:, None] @ scores[None, :]  # (D, 1) @ (1, C) = (D, C)
        dW[:, y[i]] -= X[i]

    loss = loss / num_train + reg * np.sum(W**2)
    dW = dW / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X @ W                                   # (N, D) @ (D, C) = (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)  # (N, C), for numerical stability
    scores = np.exp(scores)                          # (N, C)
    scores /= np.sum(scores, axis=1, keepdims=True)  # (N, C)
    loss = np.sum(-np.log(scores[np.arange(num_train), y])) / num_train + reg * np.sum(W**2)

    scores[np.arange(num_train), y] -= 1
    dW = X.T @ scores / num_train + 2 * reg * W      # (D, N) @ (N, C) = (D, C)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
