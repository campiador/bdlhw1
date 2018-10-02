import autograd.numpy as np


class Unit:
    """
        This class represents a node in a neural network
    """
    def __init__(self, output=0, weight_vector_length=1, weight_mean=0, weight_sigma=1, bias_mean=0, bias_sigma=1):
        self.weight_vector = np.random.normal(weight_mean, weight_sigma, weight_vector_length)

        # Scaler values
        self.bias = np.random.normal(bias_mean, bias_sigma)

        self.output = output