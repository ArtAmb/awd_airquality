import math

import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivative_of_sigmoid(x):
    f = sigmoid(x)
    return f * (1 - f)


LEARNING_RATE = 0.7


class Neuron:
    # wages = []

    def __init__(self, number_of_inputs):
        self.wages = []
        for x in range(0, number_of_inputs):
            self.wages.append(np.random.rand(1)[0])
        # self.wages = np.random.rand(number_of_inputs)

    def activation_func(self, value):
        return sigmoid(value)

    def derivative_activation_func(self, value):
        return derivative_of_sigmoid(value)

    def calculate_output(self, inputs):
        result = np.multiply(inputs, self.wages)
        self.last_output = self.activation_func(sum(result))
        return self.last_output

    # def backpropagation(self, errors):
    #     result = np.multiply(errors, self.wages)
    #     return self.derivative_activation_func(sum(result))
    def update_wages(self, layer_errors):
        deltas = np.multiply(layer_errors, self.last_output * LEARNING_RATE)
        self.wages = np.add(self.wages, deltas)


class Layer:
    # neurons = []
    # bias = 0.0

    def __init__(self, number_of_inputs, number_of_neurons, bias):
        self.bias = bias
        self.neurons = []
        for i in range(0, number_of_neurons):
            self.neurons.append(Neuron(number_of_inputs + 1))

    def calculate_output(self, inputs):
        inputs.append(self.bias)
        return [n.calculate_output(inputs) for n in self.neurons]

    def derivative_activation_func(self, value):
        return derivative_of_sigmoid(value)

    def get_neuron_wages(self, errors):
        if errors.__len__() == self.neurons.__len__():
            return [n.wages for n in self.neurons]
        else:
            res = [n.wages for n in self.neurons]
            l = res[0].__len__()
            res.append(np.zeros(l))
            return res

    def backpropagation(self, errors):
        # errors.append(0)  ????????? TODO co z bias ????
        neuron_wages = np.array(self.get_neuron_wages(errors))
        length = neuron_wages.shape[1]

        return [self.calculate_backpropagation(errors, neuron_wages[:, colIdx]) for colIdx in range(0, length)]

    def calculate_backpropagation(self, errors, wages):
        result = np.multiply(errors, wages)
        return self.derivative_activation_func(sum(result))

    def update_wages(self, layer_errors):
        # waga = waga + współczynnik_uczenia * wyjście * błąd
        for n in self.neurons:
            n.update_wages(layer_errors)


NUMBER_OF_NEURONS = 0
BIAS_VALUE = 1


class NeutralNetwork:
    # layers = []

    def __init__(self, number_of_inputs, layers):
        self.layers = []
        number_of_outputs_of_prev_layer = number_of_inputs
        for layer in layers:
            num_of_neurons = layer[NUMBER_OF_NEURONS]
            bias_val = layer[BIAS_VALUE]
            self.layers.append(Layer(number_of_outputs_of_prev_layer, num_of_neurons, bias_val))
            number_of_outputs_of_prev_layer = num_of_neurons

    def calculate_output(self, inputs):
        output_of_last_layer = inputs

        for layer in self.layers:
            output_of_last_layer = layer.calculate_output(output_of_last_layer)

        return output_of_last_layer

    def backpropagation(self, errors):
        errors_of_prev_layer = errors
        # print(errors)
        for layer in reversed(self.layers):
            layer_errors = layer.backpropagation(errors_of_prev_layer)
            layer.update_wages(layer_errors)

            errors_of_prev_layer = layer_errors

        return errors_of_prev_layer
