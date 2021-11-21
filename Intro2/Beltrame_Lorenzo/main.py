# This is the main :)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

from Classes import k_fold_validation
from Classes import non_linear_generator
# custom lib
from Functions import load_toy_classification
from Functions import load_toy_regression
from Functions import plot_polynomial_features

# load data using the custom function
X_train, X_test, y_train, y_test = load_toy_regression()

# Polynomial features
#plot_polynomial_features(X_train, y_train, X_test, y_test, max_degree=10)

# since I have only two features in my dataset I plot them
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Scatter plot of the input data")
#plt.show()

X_train = pd.DataFrame(X_train, index=None)
X_test = pd.DataFrame(X_test, index=None)
y_train = pd.Series(y_train, index=None)
y_test = pd.Series(y_test, index=None)

# Non linear features
X = [X_train, X_test]
for i in X:
    nonLin_gen = non_linear_generator(i)
    # I upload the function that interest me the most
    nonLin_gen.add_tanh()
    nonLin_gen.add_exp()
    nonLin_gen.add_cosh()
    nonLin_gen.add_tan()
    nonLin_gen.add_radius()
    i = nonLin_gen.get_result()

# I perform the linear regression, I generated new features in order to have a mse score lover than 0.01
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_test)
print("The MSE for non polynomial features is: ", mean_squared_error(y_test, predictions))

# SUBTASK 2
X_train, X_test, y_train, y_test = load_toy_classification()

# first I plot my data and observ that the situazion is really similar to the previous one
plt.scatter(X_train[:, 0], X_train[:, 1])
#plt.show()
# I work better with pandas DF
X_train = pd.DataFrame(X_train, index=None)
X_test = pd.DataFrame(X_test, index=None)
y_train = pd.Series(y_train, index=None)
y_test = pd.Series(y_test, index=None)

# Non linear features
X = [X_train, X_test]
for i in X:
    nonLin_gen = non_linear_generator(i)
    # I upload the function that interest me the most
    # NOTE: I did not include the radius!
    nonLin_gen.add_tanh()
    nonLin_gen.add_exp()
    nonLin_gen.add_cosh()
    nonLin_gen.add_tan()
    i = nonLin_gen.get_result()

# I perform the classification choosing a linear kernel, I generated new features in order to have a mse score lower
# than 0.01
svm_class = SVC(kernel="linear")
svm_class.fit(X_train, y_train)
predictions = svm_class.predict(X_test)
print("The accuracy for non polynomial features is: ", accuracy_score(y_test, predictions))

# Now i print the confusion matrix
conf_mat = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()
#plt.show()

# TASK 2
# Subtask 1
california = datasets.fetch_california_housing()
data = pd.DataFrame(california.data, index=None)
target = pd.Series(california.target)

# Now I preprocess my data by scaling them
scaler = MinMaxScaler()
# I scale all the data together
scaled_data = scaler.fit_transform(data)
# I use the same seed to have consistency
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(scaled_data, target, test_size=0.7, random_state=20)
X_train_scaled = pd.DataFrame(X_train_scaled, index=None)
X_test_scaled = pd.DataFrame(X_test_scaled, index=None)

# I use the vanilla linear regression
lin_reg = LinearRegression()
# fit
lin_reg.fit(X=X_train_scaled, y=y_train)
# prediction
predictions = lin_reg.predict(X=X_test_scaled)
predictions_train = lin_reg.predict(X=X_train_scaled)
print("The MSE (TRAIN) of the lin. reg. is: ", mean_squared_error(y_true=y_train, y_pred=predictions_train))
print("The MSE (TEST) of the lin. reg. is: ", mean_squared_error(y_true=y_test, y_pred=predictions))

# Now i use a polynomial of the second order as generated feature
poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.fit_transform(X_test_scaled)
# I train using a least squares linear regressor
lin_reg = LinearRegression()
# fit
lin_reg.fit(X=X_train_poly, y=y_train)
# prediction
predictions = lin_reg.predict(X=X_test_poly)
predictions_train = lin_reg.predict(X=X_train_poly)
print("The MSE (TRAIN) of the lin. reg. with poly of order ", 2, " is: ",
      mean_squared_error(y_true=y_train, y_pred=predictions_train))
print("The MSE (TEST) of the lin. reg. with poly of order ", 2, " is: ",
      mean_squared_error(y_true=y_test, y_pred=predictions))

# Now i use just CV
# 10 MANIFOLDS
CV = k_fold_validation()

CV.split_manifolds(X_train_scaled, y_train, 10)
print("The result with the CV on 10 manifolds is: ", CV.cross_validation())

# 20 MANIFOLDS
CV = k_fold_validation()  # reset CV
CV.split_manifolds(X_train_scaled, y_train, 20)
print("The result with the CV on 20 manifolds is: ", CV.cross_validation())

# 10 MANIFOLDS
# I use cross validation on 100 repetitions and I calculate the variance
# I did not assign any seed in the shuffle inside split_manifolds!!
results_CV = []
for _ in range(100):
    CV = k_fold_validation()
    CV.split_manifolds(X_train_scaled, y_train, k=10)
    results_CV.append(CV.cross_validation())
print("K = 10")
print("The variance of the CV over 100 repetitions is: ", np.var(results_CV))
print("The mean of the CV over 100 repetitions is: ", np.mean(results_CV))

# 20 MANIFOLDS
# I use cross validation on 100 repetitions and I calculate the variance
# I did not assign any seed in the shuffle inside split_manifolds!!
results_CV = []
for _ in range(100):
    CV = k_fold_validation()
    CV.split_manifolds(X_train_scaled, y_train, k=20)
    results_CV.append(CV.cross_validation())

print("K = 20")
print("The variance of the CV over 100 repetitions is: ", np.var(results_CV))
print("The mean of the CV over 100 repetitions is: ", np.mean(results_CV))

# SUBTASK 3
# PART 1
# this vector contains the best degree of the polynomials chosen via various methods
final_best_degree = []
# Select the best model on the Training as a whole
max_degree = 5
# I start from degree 1!!
degrees = range(1, max_degree + 1)
lin_reg = LinearRegression()

# store my mse
mse = []
for deg in degrees:
    poly = PolynomialFeatures(deg)
    # new features
    X_train_poly = poly.fit_transform(X_train_scaled)
    # fit
    lin_reg.fit(X=X_train_poly, y=y_train)
    # prediction on the train
    predictions = lin_reg.predict(X=X_train_poly)
    # evaluation
    mse.append(mean_squared_error(predictions, y_train))

# Model selection
best_mse = min(mse)
# We start from degree 1
best_degree = int(mse.index(best_mse)) + 1
final_best_degree.append(best_degree)
print("The best degree is: ", best_degree, " with the following MSE (TRAIN): ", best_mse)

# PART 2
# I split my train data set into two part (new train and new test) of the same length
X_train_2fold, X_test_2fold, y_train_2fold, y_test_2fold = train_test_split(
    X_train_scaled, y_train, random_state=20, test_size=0.5)

# I clear my mse list
mse = []
# NOTE: I use previous degrees for consistency
for deg in degrees:
    poly = PolynomialFeatures(deg)
    # new features
    X_train_poly = poly.fit_transform(X_train_2fold)
    X_test_poly = poly.fit_transform(X_test_2fold)
    # fit
    lin_reg.fit(X=X_train_poly, y=y_train_2fold)
    # prediction on the train
    predictions = lin_reg.predict(X=X_test_poly)
    # evaluation
    mse.append(mean_squared_error(predictions, y_test_2fold))

# Model selection
best_mse = min(mse)
# We start from degree 1
best_degree = int(mse.index(best_mse)) + 1
final_best_degree.append(best_degree)
print("The best degree when selecting on half the train set is: {}".format(best_degree),
      " with the following MSE (TEST): {}".format(best_mse))

# PART 3
CV = k_fold_validation()  # reset CV
CV.split_manifolds(X_train_scaled, y_train, 10)
best_degree, best_mse = CV.polynomial_selection(max_degree=max_degree)
print("\nK = 10")
print("The best degree is: ", best_degree, " with the following MSE (CV): ", best_mse)
final_best_degree.append(best_degree)

# PART 4
CV = k_fold_validation()  # reset CV
CV.split_manifolds(X_train_scaled, y_train, 20)
best_degree, best_mse = CV.polynomial_selection(max_degree=max_degree)
print("\nK = 20")
print("The best degree is: ", best_degree, " with the following MSE (CV): ", best_mse)
final_best_degree.append(best_degree)

# EVALUATION ON THE TEST DF
# I compute the mses on the test data set for the previously chosen degree of the polynomial
lin_reg = LinearRegression()
for deg in final_best_degree:
    poly = PolynomialFeatures(deg)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.fit_transform(X_test_scaled)
    # fit
    lin_reg.fit(X=X_train_poly, y=y_train)
    # prediction
    predictions = lin_reg.predict(X=X_test_poly)
    print("The MSE (TEST) of the lin. reg. with poly of order ", deg, " is: ",
          mean_squared_error(y_true=y_test, y_pred=predictions))
