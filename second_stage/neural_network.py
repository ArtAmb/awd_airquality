import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class Neuron:
    # wages = []

    def __init__(self, number_of_inputs):
        self.wages = [1 in range(0, number_of_inputs)]

    def activation_func(self, value):
        return sigmoid(value)

    def derivative_activation_func(self, value):
        f = sigmoid(value)
        return f * (1 - f)

    def calculate_output(self, inputs):
        result = np.multiply(inputs, self.wages)
        return self.activation_func(sum(result))

    # def backpropagation(self, errors):
    #     result = np.multiply(errors, self.wages)
    #     return self.derivative_activation_func(sum(result))



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

    def backpropagation2(self, errors):
        [n.wages for n in self.neurons]

    def backpropagation(self, errors):
        #errors.append(0)  ????????? TODO co z bias ????
        return [n.backpropagation(errors) for n in self.neurons]


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

        for layer in reversed(self.layers):
            layer.backpropagation()