import numpy as np
import matplotlib.pyplot as plt

from utility.util import get_normalized_data, error_rate, cost, y2indictor
from utility.mlp import forward, derivative_w1, derivative_b1, derivative_w2, derivative_b2


def main():
    max_iter = 20
    print_perios = 10
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    lr = 0.0004
    reg = 0.01

    Ytrain_ind = y2indictor(Ytrain)
    Ytest_ind = y2indictor(Ytest)
    N, D = Xtrain.shape
    batchsz = 500
    nbatches = N // batchsz
    M = 300
    K = 10

    W1 = np.random.randn(D, M)/np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)/np.sqrt(M)
    b2 = np.zeros(K)

    #constant batch

    LL_batch = []
    CR_batch = []

    for i in range(max_iter):
        for j in range(nbatches):
            Xbatch = Xtrain[j*batchsz:(j*batchsz+batchsz),]
            Ybatch = Ytrain_ind[j*batchsz:(j*batchsz+batchsz),]
            Pybatch, Z = forward(Xbatch, W1, b1, W2, b2)

            #constant weight update

            W1 -=lr*(derivative_w1(Xbatch,Z,Ybatch,Pybatch,W2)+reg*W1)
            b1 -=lr*(derivative_b1(Z,Ybatch,Pybatch,W2)+reg * b1)
            W2 -=lr*(derivative_w2(Z,Ybatch,Pybatch)+reg * W2)
            b2 -=lr*(derivative_b2(Ybatch, Pybatch)+reg * b2)

            if j % print_perios == 0 :
                Py, _ = forward(Xtest, W1, b1, W2, b2)
                #cost update
                print(f'Predicted target for constant update is  {Py}')
                ll = cost(Py, Ytest_ind)
                LL_batch.append(ll)
                print(f'cost for constant update  at i: {i}  j:  {j}  is  {ll}')

                err = error_rate(Py, Ytest)
                CR_batch.append(err)
                print(f'error for constant update at i: {i}  j:  {j} is {err} ')

    Py, _ = forward(Xtest, W1, b1, W2, b2)
    print(f'final error rate for constant update : {error_rate(Py,Ytest)}')


    #rmsprop
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)
    Cache_w1= 1
    Cache_b1 = 1
    Cache_w2 = 1
    Cache_b2 = 1
    lr_rms = 0.00004
    reg_rms = 0.01
    eps = 1e-10
    decay = 0.999
    LL_rms = []
    CR_rms = []

    for i in range(max_iter):
        for j in range(nbatches):
            Xbatch = Xtrain[j*batchsz:(j*batchsz+batchsz), ]
            Ybatch = Ytrain_ind[j*batchsz:(j*batchsz+batchsz), ]
            Pybatch, Z = forward(Xbatch, W1, b1, W2, b2)

            #weight update using RMS Prop
            gw1 = derivative_w1(Xbatch, Z, Ybatch, Pybatch, W2) + reg_rms*W1
            Cache_w1 = decay * Cache_w1 + (1-decay) * gw1**2
            W1 -= lr_rms * gw1/(np.sqrt(Cache_w1) + eps)

            gb1 = derivative_b1(Z, Ybatch, Pybatch, W2) + reg_rms * b1
            Cache_b1 = decay * Cache_b1 + (1-decay) * gb1**2
            b1 -= lr_rms * gb1/(np.sqrt(Cache_b1) + eps)

            gw2 = derivative_w2(Z, Ybatch, Pybatch) + reg_rms * W2
            Cache_w2 = decay * Cache_w2 + (1-decay) * gw2**2
            W2 -= lr_rms * gw2/(np.sqrt(Cache_w2) + eps)

            gb2 = derivative_b2(Ybatch, Pybatch) + reg_rms * b2
            Cache_b2 = decay * Cache_b2 + (1-decay) * gb2**2
            b2 -= lr_rms * gb2/(np.sqrt(Cache_b2) + eps)



            if j % print_perios == 0 :
                py, _ =forward(Xtest, W1, b1, W2, b2)
                ll = cost(py, Ytest_ind)
                LL_rms.append(ll)
                print(f'RMS cost at i : {i} j: {j} is {ll}')
                err = error_rate(py, Ytest)
                CR_rms.append(err)
                print(f'RMS error at i : {i} j: {j} is {err}')

    py, _ = forward(Xtest, W1, b1, W2, b2)
    print(f'overall error in caase of rms prop : {error_rate(py ,Ytest)}')

    plt.plot(LL_batch , label='const')
    plt.plot(LL_rms, label='rms')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
