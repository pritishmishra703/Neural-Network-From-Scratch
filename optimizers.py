import numpy as np

class SGD:

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
    
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate= learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0
    
    def update_params(self, layer):
        if not hasattr(layer, 'mb'):
            layer.mw = np.zeros_like(layer.weights)
            layer.vw = np.zeros_like(layer.weights)
            layer.mb = np.zeros_like(layer.biases)
            layer.vb = np.zeros_like(layer.biases)

        layer.mw = (self.beta1*layer.mw) + (1.0 - self.beta1)*layer.dweights
        layer.mb = (self.beta1*layer.mb) + (1.0 - self.beta1)*layer.dbiases
        mw_hat = layer.mw/(1.0 - self.beta1**(self.iterations + 1))
        mb_hat = layer.mb/(1.0 - self.beta1**(self.iterations + 1))

        layer.vw = (self.beta2*layer.vw) + (1.0 - self.beta2)*layer.dweights**2
        layer.vb = (self.beta2*layer.vb) + (1.0 - self.beta2)*layer.dbiases**2
        vw_hat = layer.vw/(1.0 - self.beta2**(self.iterations + 1))
        vb_hat = layer.vb/(1.0 - self.beta2**(self.iterations + 1))

        layer.weights -= self.learning_rate*mw_hat/(np.sqrt(vw_hat) + self.epsilon)
        layer.biases -= self.learning_rate*mb_hat/(np.sqrt(vb_hat) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1
