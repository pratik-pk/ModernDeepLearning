import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def get_cloud():
    nClass = 500
    D = 2
    X1 = np.random.randn(nClass, D) + np.array([0, -2])
    X2 = np.random.randn(nClass, D) + np.array([2, 2])
    X3 = np.random.randn(nClass, D) + np.array([-2, -2])
    X = np.vstack([X1, X2, X3])
    Y = np.array([0]*nClass + [1]*nClass + [2]*nClass)
    return X, Y


def get_Sprial():
    radius = np.linspace(0, 10, 100)
    thetas = np.empty((0, 6))
    for i in range(6):
        start_angle = i*np.pi/3.0
        end_angle = start_angle + np.pi/2.0
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points

    #convert into cartesian coordinate

    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])

    X = np.empty((600, 2))
    X[:, 0] = x1.flatten()
    X[:, 1] = x2.flatten()
    #noise
    X +=np.random.randn(600,2)*0.5

    Y = np.array([0]*100 + [1]*100 + [0]*100 + [1]*100 + [0]*100 + [1] * 100)
    return X, Y


def get_transformed_data():
    print('Reading in transforming data')
    df = pd.read_csv('../data/train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0].astype(np.int32)
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]


    #center the data
    mu = Xtrain.mean(axis=0)
    Xtrain = Xtrain-mu
    Xtest = Xtest-mu


    pca=PCA()
    Ztrain = pca.fit_transform(Xtrain)
    Ztest = pca.transform(Xtest)

    plot_cumulative_variance(pca)
    #normalize Z

    Ztrain = Ztrain[:, :300]
    Ztest = Ztest[:, :300]

    mu = Ztrain.mean(axis=0)
    std = Ztrain.std(axis=0)
    Ztrain = (Ztrain-mu)/std
    Ztest = (Ztest - mu)/std

    return Ztrain, Ztest, Ytrain, Ytest


def get_normalized_data():
    print('Reading in and normalizing data')
    df = pd.read_csv('../data/train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)

    X = data[:, 1:]
    Y = data[:, 0]

    Xtrain = X[:-1000]
    Xtest = X[-1000:]
    Ytrain = Y[:-1000]
    Ytest = Y[-1000:]

    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    np.place(std, std == 0, 1)
    Xtrain = (Xtrain-mu)/std
    Xtest = (Xtest-mu)/std

    return Xtrain, Xtest,Ytrain, Ytest


def plot_cumulative_variance(pca):
    P = []
    for p in pca.explained_variance_ratio_:
        if len(P) == 0:
            P.append(p)
        else:
            P.append(p+P[-1])
    plt.plot(P)
    plt.show()
    return P


def forward(X, W, b):
    #softmax
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y


def predict(p_y):
    return np.argmax(p_y, axis=1)


def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)


def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()


def gradw(t, y, X):
    return X.T.dot(t-y)


def gradb(t,y):
    return (t-y).sum(axis=0)


def y2indictor(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def benchmark_full():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    N, D = Xtrain.shape
    Ytrain_ind = y2indictor(Ytrain)
    Ytest_ind = y2indictor(Ytest)
    W = np.random.randn(D, 10)/np.sqrt(D)
    b= np .zeros(10)

    LL = []
    LLtest = []
    Crtest = []
    lr = 0.00004
    reg = 0.01

    for i in range(500):
        p_y =forward(Xtrain,W,b)
        ll =cost(p_y,Ytrain_ind)
        LL.append(ll)

        p_yt=forward(Xtest,W,b)
        ll1 = cost(p_yt, Ytest_ind)
        LLtest.append(ll1)

        err=error_rate(p_yt,Ytest)

        Crtest.append(err)

        W += lr*(gradw(Ytrain_ind, p_y, Xtrain)-reg*W)
        b += lr*(gradb(Ytrain_ind,p_y)-reg * b)
        if i%10 == 0:
            print(f'iteration : {i},  cost : {ll}')
            print(f'error : {err}')

        p_y = forward(Xtest, W, b)
        print(f'Final error rate:{ error_rate(p_y, Ytest)}')
        iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(Crtest)
    plt.show()


if __name__ == '__main__':
    benchmark_full()















