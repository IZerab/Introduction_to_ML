# this file contains the custom classes
import math
import numpy as np
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import hermite
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Why does shuffling the data give us such bad mse?

"""
Loading the data
"""


def load_toy_shuffled(shuffle):
    """
    Function that load a toy dataset
    :shuffle: whether or not shuffle the data
    :return: the toy dataset already split and shuffled (if shuffle=True)
    """
    hf = h5py.File("toy-regression.h5", "r")
    X_train = np.array(hf.get("x_train"))
    y_train = np.array(hf.get("y_train"))
    X_test = np.array(hf.get("x_test"))
    y_test = np.array(hf.get("y_test"))
    hf.close()
    if shuffle:
        X = X_train + X_test
        y = y_train + y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3953459)  # :)

    return X_train, X_test, y_train, y_test


"""
Function to generate non linear festures
"""


def Hermite_poly(degree, x):
    """
    Computes the Hermite polynomial in the point x. The hermite polynomial is truncated at the k-th degree
    :param degree: degree of the polynomial
    :param x: point where to evaluate the hermite polynomial
    :return: value of the hermite polynomial in that point
    """
    p = hermite(degree, monic=True)
    return p(x)


def Legendre_poly(degree, x):
    """
    Computes the Legendre polynomial in the point x. The hermite polynomial is truncated at the k-th degree and it
    is generated by recursion.
    :param degree: degree of the polynomial
    :param x: point where to evaluate the Legendre polynomial
    :return: value of the Legendre polynomial in that point
    """
    if (degree == 0):
        return 1  # P0 = 1
    elif (degree == 1):
        return x  # P1 = x
    else:
        return (((2 * degree) - 1) * x * Legendre_poly(degree - 1, x) - (degree - 1) * Legendre_poly(degree - 2,
                                                                                                     x)) / float(degree)


"""
Class to generate non linear features
"""


class sinusoidal_features():
    """
    This class contains methods to create cos() and sin() new features in a dataframe
    """

    def __init__(self, df):
        """
        I upload the dataset to be used and store the names of the original columns
        """
        self.df = df
        self.columns = df.columns

    def add_tanh(self):
        """
        Adds a new column with the hyperbolic tangent of the value to the df stored in the class.
        It apply cos to each numerical column.
        """
        for i in self.columns:
            # check to exclude categorical values
            if isinstance(self.df[i][0], float) or isinstance(self.df[i][0], int):
                self.df["tanh " + str(i)] = self.df[i].apply(lambda x: math.tanh(x))

    def add_cosh(self):
        """
        Adds a new column with the hyperbolic cosine of the value to the df stored in the class.
        It apply cos to each numerical column.
        """
        for i in self.columns:
            # check to exclude categorical values
            if isinstance(self.df[i][0], float) or isinstance(self.df[i][0], int):
                self.df["cosh " + str(i)] = self.df[i].apply(lambda x: math.cosh(x))

    def add_tan(self):
        """
        Adds a new column with the tangent of the value to the df stored in the class.
        It apply cos to each numerical column.
        """
        for i in self.columns:
            # check to exclude categorical values
            if isinstance(self.df[i][0], float) or isinstance(self.df[i][0], int):
                self.df["tan " + str(i)] = self.df[i].apply(lambda x: math.tan(x))

    def add_Legendre(self, max_degree):
        """
        Adds a new column with the Legendre Polynomial evaluated in each one of the elements of the df stored in the
        class. It apply cos to each numerical column.
        """
        for i in self.columns:
            # check to exclude categorical values
            if isinstance(self.df[i][0], float) or isinstance(self.df[i][0], int):
                for j in range(2, max_degree):
                    self.df["Legendre " + str(j) + "  " + str(i)] = self.df[i].apply(lambda x: Legendre_poly(j, x))

    def add_Hermite(self, max_degree):
        """
        Adds a new column with the Hermite Polynomial evaluated in each one of the elements of the df stored in the
        class. It apply cos to each numerical column.
        """
        for i in self.columns:
            # check to exclude categorical values
            if isinstance(self.df[i][0], float) or isinstance(self.df[i][0], int):
                for j in range(2, max_degree):
                    print(j)
                    self.df["Hermite " + str(j) + "  " + str(i)] = self.df[i].apply(lambda x: Hermite_poly(j, x))

    def add_circle(self):
        """
        Warning: this function in designed specifically for this Dataset.
        In the future it needs to be adapted.
        Adds a new column with the squared sum of each of the initial features of the df stored in the class.
        """
        temp1 = self.df[self.columns[0]].apply(lambda x: x ** 2)
        temp2 = self.df[self.columns[1]].apply(lambda x: x ** 2)
        self.df["sin_cos"] = temp1 + temp2

    def add_conic(self):
        """
        Warning: this function in designed specifically for this Dataset.
        In the future it needs to be adapted.
        Adds a new column with the conic representation of each of the initial features of the df stored in the class.
        """
        self.df["conic"] = np.sqrt((self.df[self.columns[0]] - 1) ** 2 + (self.df[self.columns[1]] - 2) ** 2) * 5

    def get_result(self):
        """
        :return: the dataframe stored in the class with some new features
        """
        return self.df


"""
MAIN
"""
# choose whether to shuffle or not
X_train, X_test, y_train, y_test = load_toy_shuffled(shuffle=False)

# since I have only two features in my dataset I plot them
plt.plot(X_train[:, 0])
# plt.show()
plt.plot(X_train[:, 1])
# plt.show()

X_train = pd.DataFrame(X_train, index=None)
X_test = pd.DataFrame(X_test, index=None)
y_train = pd.Series(y_train, index=None)
y_test = pd.Series(y_test, index=None)

print(X_test)

# both features seem to be typical wave signal with noise, therefore, I extract trigonometrical and hyperbolic features
X = [X_train, X_test]
for i in X:
    sinusoidal = sinusoidal_features(i)

    sinusoidal.add_tanh()
    sinusoidal.add_cosh()
    sinusoidal.add_tan()
    sinusoidal.add_Legendre(max_degree=5)  # 5 is the best one before overfitting
    sinusoidal.add_Hermite(max_degree=5)  # 5 is the best one before overfitting
    sinusoidal.add_circle()
    sinusoidal.add_conic()
    i = sinusoidal.get_result()

print(X_train)
print(X_test)

# I try to use the linear regression, I want to get below 0.01 in the test MSE
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_test)
print("The MSE is: ",mean_squared_error(y_test, predictions))

"""
If I shuffle them the result are interesting.
I think it kind of depend on some aliasing effect done by the sampling, but I don't have any strong proof at the moment
If I do not shuffle I get really nice results!
"""