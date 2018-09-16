# BDL HW2 by Behnam Heydarshahi
# September 2018

import numpy as np


"""
Regression of Bayesian Neural Network
"""
def problem1():
    bnn = Bnn([1, 2, 3], 2, 2)
    bnn.set_activation_func(np.tanh)

    y = bnn.propagate_forward_and_calculate_output()

    print(y)


class Unit:
    def __init__(self, output=0, weight=1, bias=1):
        #TODO: these should be vectors given by Gaussian Processes!
        self.weight = weight
        self.bias = bias

        # Scaler value
        self.output = output

class Bnn:
    def __init__(self, input_x_vector, num_hidden_layers, num_units_per_layer, activation_func=None):

        self.input_layer = [Unit(x) for x in input_x_vector]

        self.hidden_layers = []

        for layer_index in range(0, num_hidden_layers):
            new_layer = []

            for unit_index in range(0, num_units_per_layer):
                new_unit = Unit()
                new_layer.append(new_unit)

            self.hidden_layers.append(new_layer)

        # should be set separately
        self.activation_func = activation_func

        self.output_unit = Unit()

    def set_activation_func(self, activation_func):
        self.activation_func = activation_func

    def propagate_forward_and_calculate_output(self):

        network_output = 0

        if self.activation_func is None:
            print("Error: Activation function should be set before moving ahead with forward propagation!")
            exit(1)

        for current_layer_index, current_layer in enumerate(self.hidden_layers):

            if current_layer_index == 0:
                previous_layer = self.input_layer
            else:
                previous_layer = self.hidden_layers[current_layer_index - 1]

            # Actual propagation
            for current_unit in current_layer:
                sum = current_unit.bias

                for previous_unit in previous_layer:
                    sum += current_unit.weight * previous_unit.output

                current_unit.output = self.activation_func(sum)

            # Calculate the output
            output_sum = self.output_unit.bias
            for previous_unit in self.hidden_layers[-1]:
                output_sum += self.output_unit.weight * previous_unit.output

            # this might look redundant for now, but it is nice for the future extensions
            self.output_unit.output = output_sum

            network_output = self.output_unit.output

        return network_output

problem1()








