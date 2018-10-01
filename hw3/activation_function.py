import numpy as np


def relu(x):
    return max(0, x)


def sqexp(x):
    return np.exp(-(x**2))


function_names = {
  sqexp: "sqexp",
  np.tanh: "tanh",
  relu: "relu"
}
