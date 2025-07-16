import numpy as np

def linear_solve(data_set):  # helper function used in two places
    m, n = np.shape(data_set)
    X = np.mat(np.ones((m, n)));
    Y = np.mat(np.ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = data_set[:, 0:n - 1];
    Y = data_set[:, -1]  # and strip out Y
    xTx = X.T * X
    print(xTx)
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def model_leaf(data_set):
    ws, X, Y = linear_solve(data_set)
    return ws


def model_err(data_set):
    ws, X, Y = linear_solve(data_set)
    y_hat = X * ws
    print(sum(np.power(Y - y_hat, 2)))
    return sum(np.power(Y - y_hat, 2))


a=[[2,3],[3,4],[4,5],[5,6]]
a=np.mat(a)
print(model_err(a))

