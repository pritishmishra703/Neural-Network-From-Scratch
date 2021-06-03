import numpy as np

def Constant(shape, value=0):
    return np.full(shape=shape, fill_value=value, dtype=np.float32)


def Identity(shape, gain=1.0):
    if len(shape) != 2:
        raise ValueError('shape must be exactly 2D for Identity initializer')

    return gain * np.eye(N=shape[0], M=shape[1])


def Ones(shape):
    return np.ones(shape=shape, dtype=np.float32)


def RandomNormal(shape, mean=0.0, stddev=0.05, seed=None):
    np.random.seed(seed)
    return np.random.normal(loc=mean, scale=stddev, size=shape)


def RandomUniform(shape, minval=-0.05, maxval=0.05, seed=None):
    np.random.seed(seed)
    return np.random.uniform(low=minval, high=maxval, size=shape)


def Zeros(shape):
    return np.zeros(shape=shape, dtype=np.float32)


def VarienceScaling(shape, scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None):
    np.random.seed(seed=seed)
    # arguments compatibility check up
    if scale <= 0:
        raise ValueError('scale can only be a positive float')
    
    if mode not in ['fan_in', 'fan_out', 'fan_avg']:
        raise ValueError(f"Invalid 'mode' argument: {mode}")

    if distribution not in ['uniform', 'truncated_normal', 'untruncated_normal']:
        raise ValueError(f"Invalid 'distribution' argument: {distribution}")

    # Declaring fan values
    if len(shape) < 1:
        fan_in = fan_out = 1

    elif len(shape) == 1:
        fan_in = fan_out = shape[0]

    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    
    else:
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim

        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size

        fan_in = int(fan_in)
        fan_out = int(fan_out)

    if mode == 'fan_in':
        scale /= max(1.0, fan_in)
    
    elif mode == 'fan_out':
        scale /= max(1.0, fan_out)
    
    else:
        scale /= max(1.0, (fan_in + fan_out) / 2.0)

    if distribution == 'truncated_normal':
        stddev = np.sqrt(scale) / 0.87962566103423978
        return np.random.normal(loc=0.0, scale=stddev, size=shape).astype(np.float32)

    elif distribution == 'untruncated_normal':
        stddev = np.sqrt(scale)
        return np.random.normal(loc=0.0, scale=stddev, size=shape).astype(np.float32)
    
    else:
        limit = np.sqrt(3.0 * scale)
        return np.random.uniform(low=-limit, high=limit, size=shape).astype(np.float32)


def GlorotNormal(shape, seed=None):
    return VarienceScaling(shape, scale=1.0, mode='fan_avg', distribution='truncated_normal', seed=seed)


def GlorotUniform(shape, seed=None):
    return VarienceScaling(shape, scale=1.0, mode='fan_avg', distribution='uniform', seed=seed)


def HeNormal(shape, seed=None):
    return VarienceScaling(shape=shape, scale=2.0, mode='fan_in', distribution='truncated_normal', seed=seed)


def HeUniform(shape, seed=None):
    return VarienceScaling(shape=shape, scale=2.0, mode='fan_in', distribution='uniform', seed=seed)


def LecunNormal(shape, seed=None):
    return VarienceScaling(shape=shape, scale=1.0, mode='fan_in', distribution='truncated_normal', seed=seed)


def LecunUniform(shape, seed=None):
    return VarienceScaling(shape=shape, scale=1.0, mode='fan_in', distribution='uniform', seed=seed)


def Orthogonal(shape, gain=1.0, seed=None):
    ####### REMAINING
    pass
