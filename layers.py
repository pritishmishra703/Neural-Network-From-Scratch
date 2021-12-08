import pickle
import numpy as np
from dlthon import initializers

class Dense:

    def __init__(self, units, input_shape=None, activation=None, use_bias=True, weight_initializer='glorot_uniform', 
    bias_initializer='zeros', weight_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    weight_constraint=None, bias_constraint=None):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.weight_regularizer = weight_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.weight_constraint = weight_constraint
        self.bias_constraint = bias_constraint
        self.input_shape = input_shape

        if self.units < 1:
            raise ValueError("'units' should be strictly greater than or equal to 1")

        self.weight_initializer_func = pickle.loads(initializers.deserialize(self.weight_initializer))
        self.bias_initializer_func = pickle.loads(initializers.deserialize(self.bias_initializer))


    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases


    def backward(self, inputs):
        self.dweights = np.dot(self.inputs.T, inputs)
        self.dbiases = np.sum(inputs, axis=0)
        self.dinputs = np.dot(inputs, self.weights.T)


    def initialize(self):
        self.weights = self.weight_initializer_func(shape=(self.input_shape[-1], self.units))
        self.biases = self.bias_initializer_func(shape=(self.units,))
        self.initialized = True
