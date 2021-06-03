import numpy as np

class elu:

    def forward(self, x, alpha):
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

    def forward(self):
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

    def forward(self, x):
        return np.maximum(0, x).astype(np.float32)

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
        output = exp_values/np.sum(exp_values, keepdims=True, axis=1)
        return output.astype(np.float32)

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
