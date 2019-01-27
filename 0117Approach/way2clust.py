import numpy as np
from scipy.spatial.distance import cdist, euclidean

def euclidean_distance(a,b):
    return np.linalg.norm(a-b)

# Code from https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps: # P4B Annot : using norm can be faster
            return y1

        y = y1

def k_objfunc(x,k,func,maxiter=1e4,eps=0.1):
    #init
    label=np.zeros(x.shape[0])
    evpoints = np.zeros((k, x.shape[1]))
    err=100
    itercnt=0
    oldevpoints=None
    while itercnt < maxiter:
        if err<=eps:
            break
        oldevpoints=np.copy(evpoints)
        itercnt+=1
        # recalc label
        for i,j in enumerate(x):
            dist=euclidean(j,evpoints)
            label[i]=np.argmin(dist)

        # recalc evpoints
        for i in range(k):
            assocpoints = [x[j] for j in range(x.shape[0]) if label[j] == i]
            if assocpoints:
                evpoints[i] = func(assocpoints)
        err=euclidean(oldevpoints,evpoints)
    return evpoints,label

print(k_objfunc(np.array([[1.,1.],[1.,3.]]),1,geometric_median))