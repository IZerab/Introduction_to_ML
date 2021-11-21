# This is a file containing various useful functions
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


def load_toy_regression():
    """
    Function that load a toy dataset
    :return: the toy dataset
    """
    hf = h5py.File("toy-regression.h5", "r")
    X_train = np.array(hf.get("x_train"))
    y_train = np.array(hf.get("y_train"))
    X_test = np.array(hf.get("x_test"))
    y_test = np.array(hf.get("y_test"))
    hf.close()
    return X_train, X_test, y_train, y_test


def load_toy_classification():
    """
    Function that load a toy dataset
    :return: the toy dataset already split and shuffled
    """
    hf = h5py.File("toy-classification.h5", "r")
    X_train = np.array(hf.get("x_train"))
    y_train = np.array(hf.get("y_train"))
    X_test = np.array(hf.get("x_test"))
    y_test = np.array(hf.get("y_test"))
    hf.close()
    # X = X_train + X_test
    # y = y_train + y_test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3953459)  # :)
    return X_train, X_test, y_train, y_test


def plot_polynomial_features(X_train, y_train, X_test, y_test, max_degree):
    """
    Function that calculates the mse error on the test data set and plots it against the rank of the polynomial
    generation done on the X-train.
    :param X_train: for the training
    :param y_train: for the training
    :param X_test: to test the prediction of the regressor
    :param max_degree: max degree at which calculate the polynomial
    :param y_test: target to test the regressor
    :return: the aforementioned plot
    """
    # degree of the poly
    degree = range(2, max_degree)
    # where to store the MSE for various degrees
    mse = []
    for k in degree:
        poly = PolynomialFeatures(k)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.fit_transform(X_test)
        # I train my alg. with SK standard linear regressor because it is fast :)
        lin_reg = LinearRegression(fit_intercept=False)
        # fit
        lin_reg.fit(X=X_train_poly, y=y_train)
        # prediction
        predictions_test = lin_reg.predict(X=X_test_poly)
        mse.append(mean_squared_error(y_true=y_test, y_pred=predictions_test))

    # adjusting the plot
    plt.plot(degree, mse, marker="o")
    plt.xlabel("Degree")
    plt.ylabel("mse")
    plt.xticks(np.arange(1, max_degree, 2))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.title("Scatter plot of mse on the test set against the degree of the polynomial")
    plt.grid(color='grey', linestyle='-', linewidth=0.5)
    plt.show()
