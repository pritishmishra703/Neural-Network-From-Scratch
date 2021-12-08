import pickle
import numpy as np
from dlthon import activation_functions
from tqdm import tqdm


class Sequential:

    def __init__(self, layers=None):
        if layers == None:
            self.layers = []
        else:
            self.layers = layers


    def add(self, layer):
        self.layers.append(layer)


    def build(self, input_shape=None):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'initialize'):
                if i == 0:
                    if input_shape == None:
                        layer.initialize()
                        output_shape = (np.dot(np.zeros(input_shape), layer.weights) \
                         + layer.biases).shape
                        layer.output_shape = output_shape
                    else:
                        layer.input_shape = input_shape
                        layer.initialize()
                        output_shape = (np.dot(np.zeros(input_shape), layer.weights) \
                            + layer.biases).shape
                        layer.output_shape = output_shape
                else:
                    layer.input_shape = output_shape
                    layer.initialize()
                    output_shape = (np.dot(np.zeros(output_shape), layer.weights) \
                         + layer.biases).shape
                    layer.output_shape = output_shape

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                if layer.activation != None:
                    activation = pickle.loads(activation_functions.deserialize(layer.activation))()
                    self.layers.insert(i+1, activation)

        self.finalized = True


    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics


    def fit(self, X, y, epochs=10):
        if not hasattr(self, 'finalized'):
            self.build(X.shape)

        for epoch in range(1, epochs + 1):
            # FORWARD PASS
            for i, layer in enumerate(self.layers):
                if i == 0:
                    layer.forward(X)
                    last_forward = layer.output.copy()
                else:
                    layer.forward(last_forward)
                    last_forward = layer.output.copy()
            
            # CALCULATING LOSS AND METRICS
            predictions = last_forward.ravel()
            loss = self.loss.forward(y, predictions)
            metrics = self.metrics.forward(y, predictions)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}, Metrics: {metrics}')

            # BACKWARD PASS
            self.loss.backward(y, predictions)
            for i, layer in enumerate(self.layers[::-1]):
                if i == 0:
                    layer.backward(np.expand_dims(self.loss.dinputs, axis=1))
                    last_backward = layer.dinputs.copy()
                else:
                    layer.backward(last_backward)
                    last_backward = layer.dinputs.copy()
            
            # UPDATING PARAMETERS
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    self.optimizer.update_params(layer)
            self.optimizer.post_update_params()


    def predict(self, X):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.forward(X)
                last_forward = layer.output.copy()
            else:
                layer.forward(last_forward)
                last_forward = layer.output.copy() 
        
        return last_forward.ravel()
