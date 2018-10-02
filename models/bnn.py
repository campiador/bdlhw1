import numpy as np

from autograd.scipy.stats import norm
from autograd import numpy as ag_np

from models.unit import Unit


class Bnn:
    def __init__(self, input_x_vector, num_hidden_layers, num_units_per_layer, activation_func=None):

        if num_units_per_layer < 1:
            print("num units per layer should be at least 1. If you want no hidden layers, set num hidden layers to 0")
            exit(1)

        self.input_layer = [Unit(x) for x in input_x_vector] #weights are not important here

        self.hidden_layers = []

        for layer_index in range(0, num_hidden_layers):
            new_layer = []

            for unit_index in range(0, num_units_per_layer):

                # Implementation detail: the units in first hidden layer should have weight vectors of length equal
                # to the input vector length. In the rest of hidden layers, the units should have weight vectors with
                # length of the previous layer length. In this implementation, all hidden layers have same num of units!
                # Therefore, the units have a weight vector of the length of num_units_per_layer
                if layer_index == 0: # first hidden layer (not the input layer)
                    new_unit = Unit(0, len(self.input_layer))
                else:
                    new_unit = Unit(0, num_units_per_layer)

                new_layer.append(new_unit)

            self.hidden_layers.append(new_layer)

        # should be set separately
        self.activation_func = activation_func

        self.output_unit = Unit(0, num_units_per_layer)

    def set_input(self, unit):
            self.input_layer = [unit]

    def set_activation_func(self, activation_func):
        self.activation_func = activation_func

    def propagate_forward_and_calculate_output(self, unit):
        self.set_input(unit)

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
                    sum += current_unit.weight_vector[previous_unit_index] * previous_unit.output

                current_unit.output = self.activation_func(sum)

            # Calculate the output
            output_sum = self.output_unit.bias
            for previous_unit_index, previous_unit in enumerate(self.hidden_layers[-1]):
                output_sum += self.output_unit.weight_vector[previous_unit_index] * previous_unit.output

            # this might look redundant for now, but it is nice for the future extensions
            self.output_unit.output = output_sum

            network_output = self.output_unit.output

        return network_output

    def get_log_p_w(self):
        log_p_w = 0

        total_layers = self.hidden_layers + [[self.output_unit]]

        for layer in total_layers:
            for unit in layer:
                for w in unit.weight_vector:
                    log_p_w += norm.logpdf(w, 0, 1)

        return log_p_w

    def get_log_p_b(self):
        log_p_b = 0

        total_layers = self.hidden_layers + [[self.output_unit]]

        for layer in total_layers:
            for unit in layer:
                log_p_b += norm.logpdf(unit.bias, 0, 1)
        return log_p_b

    # def likelihood(self, n_inputs, sigma=0.1):
    #     lik = 1
    #
    #     for input in n_inputs:
    #         f_x_w_b = self.propagate_forward_and_calculate_output(Unit(input))
    #         lik *= np.random.normal(f_x_w_b, sigma)
    #
    #     return lik

    def log_likelihood(self, n_outputs, n_inputs, sigma=0.1):
        loglik = 0

        for index, input in enumerate(n_inputs):
            f_x_w_b = self.propagate_forward_and_calculate_output(Unit(input))
            loglik += norm.logpdf(n_outputs[index], f_x_w_b, sigma**2)

        return loglik


    def get_log_q_w_b_given_m_s(self, m_tilda, m_bar, s_tilda, s_bar):
        log_q = 0.0

        total_layers = self.hidden_layers + [[self.output_unit]]

        for layer in total_layers:
            for unit in layer:
                for w in unit.weight_vector:
                    log_q += norm.logpdf(w, m_tilda, ag_np.exp(s_tilda)**2)

                log_q += norm.logpdf(unit.bias, m_bar, ag_np.exp(s_bar)**2)

        return log_q

