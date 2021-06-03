import numpy as np
import warnings
from dlthon.activation_functions import softplus

def _check_data_validity(y_true, y_pred):
    # Check if Data type is a numpy array
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Data should be of type 'numpy.ndarray'")

    # Check if length of both array is same
    if y_true.shape != y_pred.shape:
        raise ValueError("'y_true' and 'y_pred' should be of same shape.")
    
    if str(y_true.dtype) != 'float32':
        warn_message = "numpy array is being casted from dtype '{}' to 'float32'".format(y_true.dtype)
        warnings.warn(warn_message)
        y_true = y_true.astype(np.float32)
    
    if str(y_pred.dtype) != 'float32':
        warn_message = "numpy array is being casted from dtype '{}' to 'float32'".format(y_pred.dtype)
        warnings.warn(warn_message)
        y_pred = y_pred.astype(np.float32)


def BinaryCrossEntropy(y_true, y_pred):
    _check_data_validity(y_true, y_pred)

    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1 - y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)

    return -np.mean(term_0+term_1, axis=0)


def CategoricalCrossEntropy(y_true, y_pred):
    _check_data_validity(y_true, y_pred)
    

def CategoricalHinge(y_true, y_pred):
    _check_data_validity(y_true, y_pred)
    pos = np.sum(y_true * y_pred, axis=-1)
    neg = np.max((1. - y_true) * y_pred, axis=-1)
    return np.maximum(neg - pos + 1., 0.0)


def CosineSimilarity(y_true, y_pred):
    _check_data_validity(y_true, y_pred)

    # Calculate loss
    return -np.sum(
        np.mean(np.expand_dims(
            y_true/np.linalg.norm(y_true)*y_pred/np.linalg.norm(y_pred), 
            axis=1), axis=1))

def Hinge(y_true, y_pred):
    _check_data_validity(y_true, y_pred)
    return np.mean(np.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)

def Huber(y_true, y_pred, delta=1.0):
    _check_data_validity(y_true, y_pred)
    error = np.subtract(y_true, y_pred)
    absolute_error = np.abs(error)
    return np.mean(np.where(absolute_error <= delta, 
    0.5 * np.square(error), 0.5 + np.square(delta) + delta * (absolute_error - delta)), axis=-1)

def KLDivergence(y_true, y_pred):
    _check_data_validity(y_true, y_pred)
    y_true = np.clip(y_true, 1e-7, 1.0 - 1e-7)
    y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return np.sum(y_true*np.log(y_true/y_pred), axis=-1)

def LogCosh(y_true, y_pred):
    _check_data_validity(y_true, y_pred)
    error = np.subtract(y_true, y_pred)
    return np.mean(error + softplus(-2.0*error) - np.log(2.0))

def MAE(y_true, y_pred):
    _check_data_validity(y_true, y_pred)

    # Calculate Loss
    return np.mean(abs(y_true - y_pred), axis=-1)

def MAPE(y_true, y_pred):
    _check_data_validity(y_true, y_pred)

    # Calculate Loss
    return 100 * np.mean(abs(y_true - y_pred), axis=-1)

def MSE(y_true, y_pred):
    _check_data_validity(y_true, y_pred)

    # Calculate Loss
    return np.mean(np.square(y_true - y_pred), axis=-1)

def MSLE(y_true, y_pred):
    _check_data_validity(y_true, y_pred)

    # Calculate Loss
    return np.mean(np.square(np.log(y_true + 1) - np.log(y_pred + 1)), axis=-1)

def Poisson(y_true, y_pred):
    _check_data_validity(y_true, y_pred)
    return np.mean(y_pred - y_true * np.log(y_pred + 1e-7), axis=-1)

def SparseCategoricalCrossentropy(y_true, y_pred):
    _check_data_validity(y_true, y_pred)

def SquaredHinge(y_true, y_pred):
    _check_data_validity(y_true, y_pred)
