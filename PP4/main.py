# Pen & Paper 4
# implementation of a SVM - MAIN

# standard lib
import numpy as np
import matplotlib.pyplot as plt

# custom lib
from Functions import Data_generation_sin
from sklearn.svm import SVR

# create my data set
X, y = Data_generation_sin(1000)

# train my SVM for a regression problem
epsilon = 0.1
my_svr = SVR(epsilon=epsilon)
my_svr.fit(X=X, y=y)


# PLOT
# generate my data point for the plot
X_plot = np.arange(0,10,0.1).reshape(-1, 1)
# I compute the predictions of the given points
y_plot = my_svr.predict(X_plot)
# plot the results
plt.scatter(X_plot, y_plot, marker="o", label="predicted")
# I plot the upper and lower bounds of the epsilon tube
plt.plot(X_plot, y_plot + epsilon, label="Upper bound")
plt.plot(X_plot, y_plot - epsilon, label="Lower bound")
# I also plot the "true" fuction
plt.plot(X_plot, np.sin(X_plot), label="True function")
plt.legend()
plt.grid()
plt.show()



