import numpy as np

def elu(x, alpha=1.0):
    output_matrix = []
    for i in np.nditer(x):
        if i > 0:
            output_matrix.append(i)
        else:
            output_matrix.append(alpha*(np.exp(i) - 1))
    return np.array(output_matrix).reshape(x.shape)

def exponential(x):
    return np.exp(x)

def hard_sigmoid(x):
    output = []
    for i in np.nditer(x):
        if i < -2.5:
            output.append(0)
        elif i > 2.5:
            output.append(1)
        elif -2.5 <= i <= 2.5:
            output.append(0.2*i + 0.5)

    return np.array(output).reshape(x.shape)

def relu(x):
    return np.maximum(0, x)

def selu(x):
    alpha=1.67326324
    scale=1.05070098
    return scale*elu(x, alpha)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    exp_values = np.exp(x - np.max(x, keepdims=True, axis=1))
    output = exp_values/np.sum(exp_values, keepdims=True, axis=1)
    return output

def softplus(x):
    return np.log(np.exp(x) + 1)

def swish(x):
    return x * sigmoid(x)

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def leaky_relu(x, alpha=0.3):
    return np.maximum(alpha*x, x)
