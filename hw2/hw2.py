# BDL HW2 by Behnam Heydarshahi
# September 2018

import numpy as np


"""
Regression of Bayesian Neural Network
"""
def problem1():
    bnn = Bnn([1, 2, 3], 2, 2)
    bnn.set_activation_func(np.tanh) #TODO try other activation functions

    y = bnn.propagate_forward_and_calculate_output()

    print(y)


class Unit:
    def __init__(self, output=0, weight=[], bias=0):
        #TODO: these should be Gaussian Processes!
        self.weight = weight

        # Scaler values
        #TODO: make gaussian!
        self.bias = bias

        self.output = output

class Bnn:
    def __init__(self, input_x_vector, num_hidden_layers, num_units_per_layer, activation_func=None):

        self.input_layer = [Unit(x) for x in input_x_vector]

        self.hidden_layers = []

        for layer_index in range(0, num_hidden_layers):
            new_layer = []

            for unit_index in range(0, num_units_per_layer):

                # Implementation detail: the units in first hidden layer should have weight vectors of length equal
                # to the input vector length. In the rest of hidden layers, the units should have weight vectors with
                # length of the previous layer length. In this implementation, all hidden layers have same num of units!
                # Therefore, the units have a weight vector of the length of num_units_per_layer
                if layer_index == 0:
                    new_unit = Unit(0, [0] * len(self.input_layer))
                else:
                    new_unit = Unit(0, [0] * num_units_per_layer)

                new_layer.append(new_unit)

            self.hidden_layers.append(new_layer)

        # should be set separately
        self.activation_func = activation_func

        self.output_unit = Unit(0, [0] * num_units_per_layer)

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

                for previous_unit_index, previous_unit in enumerate(previous_layer):
                    sum += current_unit.weight[previous_unit_index] * previous_unit.output

                current_unit.output = self.activation_func(sum)

            # Calculate the output
            output_sum = self.output_unit.bias
            for previous_unit_index, previous_unit in enumerate(self.hidden_layers[-1]):
                output_sum += self.output_unit.weight[previous_unit_index] * previous_unit.output

            # this might look redundant for now, but it is nice for the future extensions
            self.output_unit.output = output_sum

            network_output = self.output_unit.output

        return network_output

problem1()








