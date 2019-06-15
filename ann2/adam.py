import numpy as np
import matplotlib.pyplot as plt

from utility.util import get_normalized_data,cost,error_rate,y2indictor
from utility.mlp import forward,derivative_b2,derivative_w2,derivative_b1,derivative_w1


def main():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    Ytrain_ind = y2indictor(Ytrain)
    Ytest_ind = y2indictor(Ytest)

    batchsz = 500
    max_iter = 10
    print_period = 10
    N, D = Xtrain.shape
    nbatch = N // batchsz
    M = 300
    K = 10
    W1_0 = np.random.randn(D, M) / np.sqrt(D)
    b1_0 = np.zeros(M)
    W2_0 = np.random.randn(M, K) / np.sqrt(M)
    b2_0 = np.zeros(K)
    reg = 0.01

    # 1 adam

    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    #1st moment

    mW1 = 0
    mb1 = 0
    mW2 = 0
    mb2 = 0

    #2nd moment

    vW1 = 0
    vW2 = 0
    vb1 = 0
    vb2 = 0

    lr_adam = 0.0001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8


    loss_adam = []
    error_adam = []
    t = 1

    for i in range(max_iter):
        for j in range(nbatch):
            Xbatch = Xtrain[j*batchsz:(j*batchsz+batchsz), ]
            Ybatch = Ytrain_ind[j*batchsz:(j*batchsz+batchsz), ]
            Pybatch, Z = forward(Xbatch, W1, b1, W2, b2)

            #update the weight

            gw1 = derivative_w1(Xbatch, Z, Ybatch, Pybatch, W2) + reg*W1
            mW1 = beta1 * mW1 + (1-beta1) * gw1
            vW1 = beta2 * vW1 + (1-beta2) * gw1**2

            #baised correction
            correction1 = 1 - beta1**t
            correction2 = 1 - beta2**t
            mcapW1 = mW1/correction1
            vcapW1 = vW1/correction2

            W1 -= lr_adam * mcapW1/np.sqrt(vcapW1+eps)

            gw2 = derivative_w2(Z, Ybatch, Pybatch) + reg * W2
            mW2 = beta1 * mW2 + (1-beta1) * gw2
            vW2 = beta2 * vW2 + (1-beta2) * gw2**2
            t +=1
            mcapW2 = mW2/correction1
            vcapW2 = vW2/correction2

            W2 -= lr_adam * mcapW2/np.sqrt(vcapW2+eps)

            gb1 = derivative_b1(Z, Ybatch, Pybatch, W2)+reg*b1
            mb1 = beta1 * mb1 +(1-beta1) * gb1
            vb1 = beta2 * vb1 +(1-beta2) * gb1**2

            mcapb1 = mb1/correction1
            vcapb1 = vb1/correction2

            b1 -= lr_adam * mcapb1/np.sqrt(vcapb1+eps)

            gb2 = derivative_b2(Ybatch, Pybatch) + reg* b2
            mb2 = beta1 * mb2 + (1-beta1) * gb2
            vb2 = beta2 * vb2 + (1-beta2) * gb2**2

            mcapb2 = mb2 / correction1
            vcapb2 = vb2 / correction2

            b2 -= lr_adam * mcapb2/np.sqrt(vcapb2+eps)

            if j % print_period == 0:
                py, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(py, Ytest_ind)
                loss_adam.append(ll)
                print(f'adam loss at i : {i} j: {j} is {ll}')

                err = error_rate(py, Ytest)
                error_adam.append(err)
                print(f'adam error at i : {i} j: {j} is {err}')

    py, _ = forward(Xtest,W1, b1, W2, b2)
    print(f'final error rate for adam  is {error_rate(py,Ytest)}')



    # 1 rms prop with momentum

    W1 = W1_0.copy()
    W2 = W2_0.copy()
    b1 = b1_0.copy()
    b2 = b2_0.copy()

    loss_rms = []
    error_rms = []

    lr_rms = 0.0001
    decay = 0.999
    eps = 1e-8
    mu = 0.9

    cacheW1 = 1
    cacheW2 = 1
    cacheb1 = 1
    cacheb2 = 1

    dw1 = 0
    db1 = 0
    dw2 = 0
    db2 = 0

    for i in range(max_iter):
        for j in range(nbatch):
            Xbatch = Xtrain[j*batchsz:(j*batchsz+batchsz), ]
            Ybatch = Ytrain_ind[j*batchsz:(j*batchsz+batchsz), ]
            Pybatch, Z = forward(Xbatch, W1, b1, W2, b2)

            #update weight

            gw1 = derivative_w1(Xbatch, Z, Ybatch, Pybatch, W2) + reg*W1
            gw2 = derivative_w2(Z, Ybatch, Pybatch) + reg * W2
            gb1 = derivative_b1(Z, Ybatch, Pybatch, W2) + reg * b1
            gb2 = derivative_b2(Ybatch,Pybatch) + reg * b2

            cacheW1 = decay * cacheW1 + (1-decay) * gw1**2
            dw1 = mu*dw1+(1-mu)*lr_rms*gw1/np.sqrt(cacheW1+eps)
            W1 -= dw1

            cacheW2 = decay * cacheW2 + (1-decay)* gw2**2
            dw2 = mu*dw2 +(1-mu)*lr_rms*gw2/np.sqrt(cacheW2+eps)
            W2 -= dw2

            cacheb1 = decay * cacheb1 +(1-decay)*gb1**2
            db1 = mu*db1 + (1-mu) * lr_rms*gb1/np.sqrt(cacheb1+eps)
            b1 -= db1

            cacheb2 = decay * cacheb2 + (1-decay)*gb2**2
            db2 = mu*db2 +(1-mu)*lr_rms*gb2/np.sqrt(cacheb2+eps)
            b2 -=db2

            if j % print_period == 0:
                py, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(py, Ytest_ind)
                loss_rms.append(ll)
                print(f'rms loss at i : {i} j: {j} is {ll}')

                err = error_rate(py, Ytest)
                error_rms.append(err)
                print(f'rms error at i : {i} j: {j} is {err}')

    py, _ = forward(Xtest, W1, b1, W2, b2)
    print(f'final error rate for rms  is {error_rate(py,Ytest)}')

    plt.plot(loss_adam, label='adam loss')
    plt.plot(loss_rms, label= 'rms with momentum loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()