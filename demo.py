import numpy as np
import random as rd
import utilities
import svm_models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" Testing SMO """
def func1(X):
    test1 = ((6 * X - 2) ** 2) * (np.sin(12 * X - 4))
    test2 = -(1.4-3*X)*np.sin(18*X) # [X: 0, 1.2]
    test3 = X*np.sin(X) + X*np.cos(2*X) #[X: 0, 10]
    return test3


def func2(X1, X2):
    test1 = np.cos(X1)**2 + np.sin(X2)**2
    test2 = ((6 * X1 - 2) ** 2) * (np.sin(12 * X2 - 4))
    test3 = X1**4 + np.sin(X2)
    test4 = X1**2 + X2**2
    rosen = (1-X1)**2 + 100*(X2-X1**2)**2
    himmelblau = (((X1*X1) + X2 - 11)*((X1*X1) + X2 - 11)) + ((X1 + (X2*X2) - 7)*(X1 + (X2*X2) - 7))

    # bounds: -5 <= x <= 10, 0 <= y <= 15
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    branin = a * (X2 - b*X1**2 + c*X1 - r)**2 + s*(1-t)*np.cos(X1) + s
    return test4


if __name__ == "__main__":
    X = np.linspace(0, 10).reshape(-1, 1)
    Z = func1(X).reshape(-1, 1)
    wholeData = np.hstack((X, Z))
    trainData, testData = utilities.train_test_split(wholeData)
    
    search = {'model_type': 'EpsilonSMO', 'model_param': [0.1, 2], 'C': [1, 1000], 'kernel_type': ['rbf'], 'kernel_param': [0.01, 10]}
    tuneObj = utilities.TuneParameters(search, trainData, testData, type='regression')
    bestParams, trainedModel = tuneObj.optuna(ntrials=50)
    
    pred = trainedModel.predict(wholeData)
    fig = plt.figure()
    plt.scatter(X, Z, c='k', label='True')
    plt.scatter(X, pred, c='m', label='Predicted')