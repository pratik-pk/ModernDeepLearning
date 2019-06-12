import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from datetime import datetime
from utility.util import get_transformed_data,forward,error_rate,cost,gradw,gradb,y2indictor


def main():
    X,_, Y, _ = get_transformed_data()
    X = X[:, :300]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mu) / std
    print('performing logistic regression')
    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    ytest = Y[-1000: ]
    N, D = Xtrain.shape
    Ytrain_ind = y2indictor(Ytrain)
    Ytest_ind = y2indictor(ytest)
    W = np.random.randn(D, 10)/28
    b = np.zeros(10)
    LL = []
    lr = 0.0001
    reg = 0.01
    to = datetime.now()
    for i in range(200):
        p_y = forward(Xtrain, W, b)
        W += lr *(gradw(Ytrain_ind, p_y, Xtrain) - reg * W)
        b += lr *(gradb(Ytrain_ind,p_y)-reg * b)
        p_y_test = forward(Xtest, W, b)
        ll = cost(p_y_test, Ytest_ind)
        LL.append(ll)
        if i % 10 == 0:
            err = error_rate(p_y_test,ytest)
            print(f'cost at iteration {i}, {ll}')
            print(f'error rate {err}')
    print(f'final error rate {error_rate(forward(Xtest, W, b), ytest)}')
    print(f'elapsed time for the full gd {datetime.now() - to}')


#Stochastic
    W = np.random.randn(D, 10)/28
    b = np.zeros(10)
    LL_stochastic = []
    lr = 0.0001
    reg = 0.01
    to = datetime.now()

    for i in range(1):  #taking 1 loop cause we have 41k data which will take lots of computation time
        tempx, tempy = shuffle(Xtrain, Ytrain_ind)
        for n in range(min(N, 500)):
            X = tempx[n, :].reshape(1, D)
            Y = tempy[n,:].reshape(1, 10)
            p_y = forward(X, W, b)
            W += lr*(gradw(Y, p_y, X)-reg * W)
            b += lr*(gradb(Y, p_y) - reg * b)
            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test,Ytest_ind)
            LL_stochastic.append(ll)
        if i % 1 == 0:
            err = error_rate(p_y_test,ytest)
            print(f'cost at iteration {i}, {ll}')
            print(f'error rate {err}')
    print(f'final error rate {error_rate(p_y,ytest)}')
    print(f'time elapsed in calculation {datetime.now()-to}')


    #batch

    W = np.random.randn(D, 10)/28
    b = np.zeros(10)
    LL_batch = []
    lr = 0.0001
    reg = 0.01
    batch_sz = 500
    n_batches = N//batch_sz
    to = datetime.now()

    for i in range(50):
        tempx, tempy = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            X = tempx[j*batch_sz:(j*batch_sz+batch_sz), :]
            Y = tempy[j*batch_sz:(j*batch_sz+batch_sz), :]
            p_y = forward(X, W, b)
            W += lr * (gradw(Y, p_y,X)-reg*W)
            b += lr * (gradb(Y, p_y)-reg*b)
            p_y_test = forward(Xtest,W,b)
            ll = cost(p_y_test,Ytest_ind)
            LL_batch.append(ll)
            if j %(n_batches/2) == 0:
                err = error_rate(p_y_test, ytest)
                print(f'cost at iteration i {i} : {ll}')
                print(f'error rate : {err}')
    print(f'final error rate {error_rate(p_y, ytest)}')
    print(f'elapsed time for batch :{datetime.now()-to}')

    X1 = np.linspace(0, 1, len(LL))
    plt.plot(X1, LL, label='full')
    X2 = np.linspace(0, 1, len(LL_stochastic))
    plt.plot(X2, LL_stochastic, label='stochastic')
    X3 = np.linspace(0, 1, len(LL_batch))
    plt.plot(X3, LL_batch, label ="batch")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()