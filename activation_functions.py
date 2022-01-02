import numpy as np
import pickle

class elu:

    def forward(self, x, alpha=1.0):
        output_matrix = []
        for i in np.nditer(x):
            if i > 0:
                output_matrix.append(i)
            else:
                output_matrix.append(alpha*(np.exp(i) - 1))
        return np.array(output_matrix).reshape(x.shape).astype(np.float32)


class exponential:

    def forward(self, x):
        return np.exp(x).astype(np.float32)


class hard_sigmoid:

    def forward(self, x):
        output = []
        for i in np.nditer(x):
            if i < -2.5:
                output.append(0)
            elif i > 2.5:
                output.append(1)
            elif -2.5 <= i <= 2.5:
                output.append(0.2*i + 0.5)

        return np.array(output).reshape(x.shape).astype(np.float32)


class relu:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs).astype(np.float32)


    def backward(self, dx):
        self.dinputs = dx.copy()
        self.dinputs[self.inputs <= 0] = 0


class selu:

    def forward(self, x):
        alpha=1.67326324
        scale=1.05070098
        return scale*elu(x, alpha).astype(np.float32)


class sigmoid:

    def forward(self, x):
        return 1/(1 + np.exp(-x)).astype(np.float32)


class softmax:
    
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, keepdims=True, axis=1))
        self.output = exp_values/np.sum(exp_values, keepdims=True, axis=1)


    def backward(self, dx):
        self.dinputs = np.empty_like(dx)

        for i, (output_i, dx_i) in enumerate(zip(self.output, dx)):
            output_i = output_i.reshape(-1, 1)
            jacobian_matrix = np.diagflat(output_i) - np.dot(output_i, output_i.T)
            self.dinputs[i] = np.dot(jacobian_matrix, dx_i)


class softplus:

    def forward(self, x):
        return np.log(np.exp(x) + 1).astype(np.float32)


class swish:
    
    def forward(self, x):
        return x * sigmoid(x).astype(np.float32)


class tanh:

    def forward(self, x):
        return (np.exp(x) - np.exp(-x)) \
            /(np.exp(x) + np.exp(-x)).astype(np.float32)


class leaky_relu:

    def forward(self, x, alpha=0.3):
        return np.maximum(alpha*x, x).astype(np.float32)


class linear:

    def forward(self, x):
        self.output = x

    def backward(self, dx):
        self.dinputs = dx.copy()


def deserialize(name, **kwargs):
    if name == 'elu': return pickle.dumps(elu)
    elif name == 'exponential': return pickle.dumps(exponential)
    elif name == 'hard_sigmoid': return pickle.dumps(hard_sigmoid)
    elif name == 'relu': return pickle.dumps(relu)
    elif name == 'selu': return pickle.dumps(selu)
    elif name == 'sigmoid': return pickle.dumps(sigmoid)
    elif name == 'softmax': return pickle.dumps(softmax)
    elif name == 'softplus': return pickle.dumps(softplus)
    elif name == 'swish': return pickle.dumps(swish)
    elif name == 'tanh': return pickle.dumps(tanh)
    elif name == 'leaky_rellu': return pickle.dumps(leaky_relu)
    elif name == 'linear': return pickle.dumps(linear)
    else: raise ValueError(f'Unknown activation function: {name}')
