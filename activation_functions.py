import numpy as np

class elu:

    def __init__(self, x, alpha=1.0):
        self.x = x
        self.alpha = alpha

    def forward(self):
        output_matrix = []
        for i in np.nditer(self.x):
            if i > 0:
                output_matrix.append(i)
            else:
                output_matrix.append(self.alpha*(np.exp(i) - 1))
        return np.array(output_matrix).reshape(self.x.shape).astype(np.float32)

class exponential:

    def __init__(self, x):
        self.x = x

    def forward(self):
        return np.exp(self.x).astype(np.float32)

class hard_sigmoid:

    def __init__(self, x):
        self.x = x

    def forward(self):
        output = []
        for i in np.nditer(self.x):
            if i < -2.5:
                output.append(0)
            elif i > 2.5:
                output.append(1)
            elif -2.5 <= i <= 2.5:
                output.append(0.2*i + 0.5)

        return np.array(output).reshape(self.x.shape).astype(np.float32)

class relu:

    def __init__(self, x):
        self.x = x

    def forward(self):
        return np.maximum(0, self.x).astype(np.float32)

class selu:

    def __init__(self, x):
        self.x = x

    def forward(self):
        alpha=1.67326324
        scale=1.05070098
        return scale*elu(self.x, alpha)

class sigmoid:

    def __init__(self, x):
        self.x = x

    def forward(self):
        return 1/(1 + np.exp(-self.x))

class softmax:

    def __init__(self, x):
        self.x = x
    
    def forward(self):
        exp_values = np.exp(self.x - np.max(self.x, keepdims=True, axis=1))
        output = exp_values/np.sum(exp_values, keepdims=True, axis=1)
        return output

class softplus:

    def __init__(self, x):
        self.x = x

    def forward(self):
        return np.log(np.exp(self.x) + 1)

class swish:

    def __init__(self, x):
        self.x = x
    
    def forward(self):
        return self.x * sigmoid(self.x)

class tanh:

    def __init__(self, x):
        self.x = x

    def forward(self):
        return (np.exp(self.x) - np.exp(-self.x))/(np.exp(self.x) + np.exp(-self.x))

class leaky_relu:

    def __init__(self, x, alpha=0.3):
        self.x = x
        self.alpha = alpha

    def forward(self):
        return np.maximum(self.alpha*self.x, self.x)
