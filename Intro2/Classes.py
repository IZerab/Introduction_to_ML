# this file contains the custom classes
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle


class non_linear_generator():
    """
    This class contains methods to create new non linear features in a dataframe.
    Functions: tanh, cosh, exp, tan, radius
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
        It apply tanh to each numerical column.
        """
        for i in self.columns:
            # check to exclude categorical values
            if isinstance(self.df[i][0], float) or isinstance(self.df[i][0], int):
                self.df["tanh " + str(i)] = self.df[i].apply(lambda x: math.tanh(x))

    def add_cosh(self):
        """
        Adds a new column with the hyperbolic cosine of the value to the df stored in the class.
        It apply cosh to each numerical column.
        """
        for i in self.columns:
            # check to exclude categorical values
            if isinstance(self.df[i][0], float) or isinstance(self.df[i][0], int):
                self.df["cosh " + str(i)] = self.df[i].apply(lambda x: math.cosh(x))

    def add_exp(self):
        """
        Adds a new column with the tangent of the value to the df stored in the class.
        It apply exp to each numerical column.
        """
        for i in self.columns:
            # check to exclude categorical values
            if isinstance(self.df[i][0], float) or isinstance(self.df[i][0], int):
                self.df["exp " + str(i)] = self.df[i].apply(lambda x: math.exp(x))

    def add_tan(self):
        """
        Adds a new column with the tangent of the value to the df stored in the class.
        It apply tan to each numerical column.
        """
        for i in self.columns:
            # check to exclude categorical values
            if isinstance(self.df[i][0], float) or isinstance(self.df[i][0], int):
                self.df["tan " + str(i)] = self.df[i].apply(lambda x: math.tan(x))

    def add_radius(self):
        """
        Warning: this function in designed specifically for this Dataset.
        In the future it needs to be adapted.
        Adds a new column with radius of the amplitudes of each of the initial features of the df stored in the class.
        """
        med0 = int(self.df[self.columns[0]].mean())
        med1 = int(self.df[self.columns[1]].mean())
        self.df["radius"] = np.sqrt((self.df[self.columns[0]] - med0) ** 2 + (self.df[self.columns[1]] - med1) ** 2)

    def get_result(self):
        """
        :return: the dataframe stored in the class with some new features
        """
        return self.df


class k_fold_validation:
    """
    This class performs a k fold cross validation of the data to select the best weight to fit the dataset.
    It also can perform the selection of the best degree for some polynomial feature on a dataset.
    The data are shuffled before being split into manifolds.
    """

    def __int__(self):
        self.folds = None
        self.y_folds = None
        self.metric = None
        self.k = None

    def split_manifolds(self, X_train, y_train, k):
        """
        This function saves the X_train and splits it into the k folds. Data must be scaled.
        :param y_train: y_train series to be split in the folding procedure
        :param k: number of manifolds to use
        :param X_train: X_train dataset to be split in the folding procedure
        :return: saves into the class the various manifolds of data
        """
        # containers for the folds i will create
        self.folds = []
        self.y_folds = []
        # Check if everything is ok
        if isinstance(k, int):
            self.k = k

        else:
            raise ValueError("The number of manifold must be an integer!")
        # this codes works with pandas DF
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("The X input given is not a pandas DataFrame")
        if not isinstance(y_train, pd.Series):
            raise ValueError("The y input given is not a pandas Series")

        # I shuffle the data while before saving them
        X_train, y_train = shuffle(X_train, y_train)  # :)

        # I reset the index to be able to work on indexes
        X_train.reset_index(inplace=True)
        y_train.reset_index(inplace=True, drop=True)

        # number of instances
        n = len(X_train.loc[:, 0])
        if n % k == 0:
            len_manifold = int(n / k)
        else:
            # we will add the spare instances directly to the first manifold
            len_manifold = int((n - n % k) / k)

        # I put the slices of the X_train into a list of pandas DF
        for i in range(k):
            # I create the boolean condition to slice the manifold I want
            condition_manifold = ((len_manifold * (i + 1)) > X_train.index) & (X_train.index > (len_manifold * i))
            # I append the features and the target
            self.folds.append(X_train.loc[condition_manifold])
            self.y_folds.append(y_train.loc[condition_manifold])

        # I add the spare instances if it is the case
        if not n % k == 0:
            # I append the spared instances in the first manifold
            self.folds[0].append(X_train.loc[X_train.index > len_manifold * k])
            # I append the spare targets
            self.y_folds[0].append(y_train.loc[y_train.index > len_manifold * k])

        # sanity check!
        if not len(self.folds) == k:
            raise ValueError("Ops, something went wrong since k is not equal to the number of folds!!")

    def cross_validation(self):
        """
        This function performs the k-fold cross validation with an approach opt one manifold out.
        The error function chosen is the MSE.
        The vector of all the errors is saved inside the class
        :return: the avarage of the various errors calculated in the cross validation
        """
        lin_reg = LinearRegression()
        counter = 0
        mse = []
        for i in range(self.k):
            # for each iteration I create a new variable containing the dataframe and the target
            temp_train = pd.DataFrame()
            temp_target = pd.Series()
            # I create the temp df by concatenating all the manifolds except the validation one
            for j in range(self.k):
                if not j == counter:
                    # appending the dataframe
                    temp_train = pd.concat([temp_train, self.folds[j]])
                    # appending the dataframe
                    temp_target = pd.concat([temp_target, self.y_folds[j]])

            lin_reg.fit(temp_train, temp_target)
            predictions = lin_reg.predict(self.folds[counter])
            mse.append(mean_squared_error(self.y_folds[counter], predictions))
            counter = counter + 1
        # I save the error vector inside the class
        self.metric = mse
        print(mse)
        print(max(mse))
        average_mse = np.mean(mse)
        exit()
        return average_mse


    def polynomial_selection(self, max_degree):
        """
        This function selects the best degree at which the linear regression (used with the polynomial features) yields
        to the best mse on the training data. The data must be provided scaled!!!!!
        :param max_degree: max degree at which computing the polynomial features
        :return: the degree at which we observe the best performances and its averaged mse
        """
        # oder of the polynomials
        degrees = range(1, max_degree + 1)
        lin_reg = LinearRegression()
        # training of the manifolds
        counter = 0

        # list where to save the different CV results at different degrees
        mse_poly = []

        # iteration over all the degrees
        for deg in degrees:
            poly = PolynomialFeatures(deg)
            # I split the train into train and validation

            # iterations over all the manifolds
            for i in range(self.k):
                # for each iteration I create a new variable containing the dataframe and the target
                temp_train = pd.DataFrame()
                temp_target = pd.Series()
                # list where to append MSE scores
                mse = []

                # I create the temp df by concatenating all the manifolds except the validation one
                for j in range(self.k):
                    if not j == counter:
                        # appending the dataframe
                        temp_train = pd.concat([temp_train, self.folds[j]])
                        # appending the dataframe
                        temp_target = pd.concat([temp_target, self.y_folds[j]])

                # create the polynomial features in each combination of the manifolds
                temp_train_poly = poly.fit_transform(temp_train)
                # feature generation for the opted out manifold
                temp_test_poly = []
                temp_test_poly = poly.fit_transform(self.folds[counter])

                # fit
                lin_reg.fit(X=temp_train_poly, y=temp_target)
                # prediction
                predictions = lin_reg.predict(X=temp_test_poly)
                # evaluation
                mse.append(mean_squared_error(predictions, self.y_folds[counter]))
            # averaging and saving the result to confront it with the other degrees
            mse_poly.append(np.mean(mse))

        best_mse = min(mse_poly)
        # the first degree given is 1!!
        best_degree = int(mse_poly.index(best_mse)) + 1
        return best_degree, best_mse
