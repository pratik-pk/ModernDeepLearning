import numpy as np
import matplotlib.pyplot as plt
from utility.util import get_normalized_data,error_rate,cost,y2indictor
from utility.mlp import forward, derivative_b2,derivative_w1,derivative_w2,derivative_b1


def main():
    #batch
    #sgd with momentum
    #sgd with nestrov momentum

    max_iter = 20
    print_period = 50

    Xtrain, Xtest,Ytrain, Ytest = get_normalized_data()
    lr = 0.0001
    reg = 0.001

    Ytrain_ind = y2indictor(Ytrain)
    Ytest_ind = y2indictor(Ytest)

    N, D = Xtrain.shape
    batchsz = 500
    n_batch = N // batchsz

    M = 300
    K = 10
    W1 = np.random.randn(D, M)/np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)/np.sqrt(M)
    b2 = np.zeros(K)

    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()

    #batch
    looses_batch = []
    error_batch = []
    for i in range(max_iter):
        for j in range(n_batch):
            Xbatch = Xtrain[j*batchsz:(j*batchsz+batchsz),]
            Ybatch = Ytrain_ind[j*batchsz:(j*batchsz+batchsz),]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            #updates

            W2 -= lr*(derivative_w2(Z, Ybatch, pYbatch)+reg*W2)
            b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
            W1 -= lr*(derivative_w1(Xbatch,Z,Ybatch,pYbatch,W2) + reg*W1)
            b1 -= lr*(derivative_b1(Z,Ybatch,pYbatch,W2) + reg*b1)

            if j % print_period == 0:
                pY, _ = forward(Xtest,W1,b1,W2,b2)
                l = cost(pY, Ytest_ind)
                looses_batch.append(l)
                print(f'cost at i: {i} , j: {j} is l:{l} ')
                er = error_rate(pY, Ytest)
                error_batch.append(er)
                print(f'error rate {er}')
    py, _ = forward(Xtest, W1, b1, W2, b2)
    print(f'final error rate : {error_rate(py,Ytest)}')



    #batch with momentum
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()
    looses_momentum = []
    error_momentum = []
    mu = 0.9
    dw1 = 0
    db1 = 0
    dw2 = 0
    db2 = 0

    for i in range(max_iter):
        for j in range(n_batch):
            Xbatch = Xtrain[j*batchsz:(j*batchsz+batchsz), ]
            Ybatch = Ytrain_ind[j*batchsz:(j*batchsz+batchsz), ]
            pYbatch, Z =forward(Xbatch, W1, b1, W2, b2)

            #gradient
            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1
            gw2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
            gb2 = derivative_b2(Ybatch, pYbatch)+ reg*b2

            #velocity update

            dw1 = mu *dw1 - lr * gW1
            db1 = mu *db1 - lr * gb1
            dw2 = mu *dw2 - lr * gw2
            db2 = mu *db2 - lr * gb2

            W1 += dw1
            b1 += db1
            W2 += dw2
            b2 += db2

            if j % print_period == 0:
                py, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(py, Ytest_ind)
                print(f'cost at i: {i} , j: {j} is l: {l}')
                looses_momentum.append(l)
                er = error_rate(py, Ytest)
                error_momentum.append(er)
                print(f'error rate: {er}')

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print(f'final error rate : {error_rate(pY, Ytest)}')



    #nestrove momentum with batch

    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    looses_nestrove = []
    error_nestrove = []

    mu = 0.9
    vw1 = 0
    vw2 = 0
    vb1 = 0
    vb2 = 0

    for i in range(max_iter):
        for j in range(n_batch):
            Xbatch = Xtrain[j*batchsz:(j*batchsz+batchsz), ]
            Ybatch = Ytrain_ind[j*batchsz:(j*batchsz+batchsz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            #update

            gw1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
            gw2 = derivative_w2(Z,Ybatch,pYbatch) + reg * W2
            gb1 = derivative_b1(Z,Ybatch,pYbatch,W2) + reg *b1
            gb2 = derivative_b2(Ybatch,pYbatch) + reg* b2

            #update velocity

            vw1 = mu*vw1 - lr * gw1
            vw2 = mu*vw2 - lr * gw2
            vb1 = mu*vb1 - lr * gb1
            vb2 = mu*vb2 - lr * gb2

            W1 += mu * vw1 - lr * gw1
            W2 += mu * vw2 - lr * gw2
            b1 += mu * vb1 - lr * gb1
            b2 += mu * vb2 - lr * gb2

            if j % print_period == 0:
                py, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(py,Ytest_ind)
                looses_nestrove.append(l)
                print(f'cost at i : {i} , j: {j}, l: {l}')
                er = error_rate(py, Ytest)
                error_nestrove.append(er)
                print(f'error : {er}')
    py , _ = forward(Xtest, W1, b1, W2, b2)
    print(f'final error for nestrove {error_rate(py, Ytest)}')

    plt.plot(looses_batch,label='batch cost')
    plt.plot(looses_momentum, label='momentum cost')
    plt.plot(looses_nestrove, label='nestrove cost')
    plt.legend()
    plt.show()

    plt.plot(error_batch,label='error batch')
    plt.plot(error_momentum,label='error momentum')
    plt.plot(error_nestrove,label='error nestrove')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


















