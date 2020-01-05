import math

import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivative_of_sigmoid(x):
    f = sigmoid(x)
    return f * (1 - f)


LEARNING_RATE = 0.5
MOMENTUM_RATE = 0.1
MOMENTUM_ACTIVE = True


class Neuron:
    # wages = []

    def __init__(self, number_of_inputs):
        self.wages = []
        self.prev_wages = []
        for x in range(0, number_of_inputs):
            self.wages.append(np.random.rand(1)[0])
            # self.wages.append(0.25)
            self.prev_wages.append(0)
            # self.wages = np.random.rand(number_of_inputs)


    def activation_func(self, value):
        return sigmoid(value)

    def derivative_activation_func(self, value):
        return derivative_of_sigmoid(value)

    def calculate_output(self, inputs):
        self.last_inputs = inputs
        result = np.multiply(inputs, self.wages)
        suma = sum(result)

        self.last_output = self.activation_func(suma)
        self.last_derivative_output = self.derivative_activation_func(suma)

        return self.last_output

    # def backpropagation(self, errors):
    #     result = np.multiply(errors, self.wages)
    #     return self.derivative_activation_func(sum(result))
    def update_wages(self, error):
        old_wages = self.wages
        tmp = LEARNING_RATE * self.last_derivative_output * error
        deltas = np.multiply(self.last_inputs, tmp)
        self.wages = np.add(self.wages, deltas)

        if MOMENTUM_ACTIVE:
            momentum_value = np.multiply(np.subtract(self.wages, self.prev_wages), MOMENTUM_RATE)
            self.wages = np.add(self.wages, momentum_value)

        self.prev_wages = old_wages

    def calculate_error(self, errors):
        return np.sum(np.multiply(self.wages, errors))


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

    # def backpropagation(self, errors):
    #     return [n.calculate_error(errors) for n in self.neurons]

    def save_errors(self, errors):
        self.current_errors = errors

    def backpropagation(self, errors):
        # errors.append(0)  ????????? TODO co z bias ????
        neuron_wages = np.array(self.get_neuron_wages(errors))
        length = neuron_wages.shape[1]

        return [self.calculate_backpropagation(errors, neuron_wages[:, colIdx]) for colIdx in range(0, length)]

    def calculate_backpropagation(self, errors, wages):
        result = np.multiply(errors, wages)
        return sum(result)

    def update_wages(self, layer_errors):
        # waga = waga + współczynnik_uczenia * wyjście * błąd

        if layer_errors.__len__() != self.neurons.__len__():
            if layer_errors.__len__() - self.neurons.__len__() == 1:
                del layer_errors[-1]
            else:
                raise Exception("layer_errors no equals to neurons")

        for idx in range(0, layer_errors.__len__()):
            self.neurons[idx].update_wages(layer_errors[idx])


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
            layer.save_errors(errors_of_prev_layer)

            layer_errors = layer.backpropagation(errors_of_prev_layer)
            layer.update_wages(errors_of_prev_layer)

            errors_of_prev_layer = layer_errors

        return errors_of_prev_layer
