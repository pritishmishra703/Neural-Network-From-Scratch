import numpy as np

class SGD:

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
    
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases
