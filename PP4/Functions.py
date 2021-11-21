# Python functions for SVM regression implementation
import numpy as np


def Data_generation_sin(dim):
    """
    This function generates a regression problem where the target function is sin(x).
    Data have a gaussian noise.
    :param dim: dimension of the sample
    """
    # sample from uniform distribution
    x = np.random.uniform(0, 10, dim)
    # compute the sine
    y_raw = np.sin(x)
    # adding some gaussian noise
    noise = np.random.normal(0, 0.1, dim)
    y = y_raw + noise
    # reshaping for a 2D array
    x = x.reshape(-1, 1)
    return x, y

def tube_generation(X, y_pred, epsilon, SVR):
    """
    This function creater the value of the epsilon tube
    :param X: design matrix
    :param y: y predicted by the svm given
    :param epsilon: epsilon parameter of the SVM we are using
    :param SVR: an already fitted SVM
    :return: two arrays containing the values of the upper and lower tube boundaries
    """
    pass
