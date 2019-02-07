import numpy as np
from scipy.spatial.distance import cdist, euclidean
from sklearn.datasets import load_iris
from functools import partial

import random,os,timeit
from time import gmtime, strftime

import matplotlib.pyplot as plt

titi = ""

def ternary_search_X0(p,s,a,e):
    # s: point(x0,x1), idx: variable argument
    l = s[0]-e
    r = s[0]+e
    xi = s[1]

    while True:
        lp = l+(r-l)/3
        rp = r-(r-l)/3

        lv = sum_distances(np.array([[lp, xi]]),p)
        rv = sum_distances(np.array([[rp, xi]]),p)

        if lv <= rv:
            r = rp
        if lv >= rv:
            l = lp
        if r-l < a:
            break

    return np.array([l,xi])

def ternary_search_X1(p,s,a,e):
    # s: point(x0,x1), idx: variable argument
    l = s[1]-e
    r = s[1]+e
    xi = s[0]

    while True:
        lp = l+(r-l)/3
        rp = r-(r-l)/3

        lv = sum_distances(np.array([[xi, lp]]),p)
        rv = sum_distances(np.array([[xi, rp]]),p)

        if lv <= rv:
            r = rp
        if lv >= rv:
            l = lp
        if r-l < a:
            break

    return np.array([xi,l])

def cordinate_descent(p,a,e):
    X = np.sum(p,0) / p.shape[0]
    
    while True:
        Xt = ternary_search_X0(p,X,a,e)
        Y  = ternary_search_X1(p,Xt,a,e)
        diff = cdist([X],[Y])
        X = Y
        if diff < a:
            break
    return X

def sum_distances(x,p):
    # x: variable value, p: const points 
    # print(x)
    # print()
    distance = cdist(p,x)
    res = np.sum(distance, 0)
    return res

def k_objfunc(x,k,func,maxiter=1e5):
    curiter=0
    Tm = np.min(x,0)
    TM = np.max(x,0)
    #evpoints=((TM-Tm)*np.random.sample((k,2)))+Tm #init func
    evpoints=np.random.permutation(x)[:k]
    labels=np.argmin(cdist(x,evpoints),axis=1)
    #labels=[np.argmin(np.linalg.norm(evpoints-i,axis=0)) for i in x]
    #assign init labels
    while curiter<maxiter:
        curiter+=1
        #calc evpoints
        for i in range(k):
            iassoc=np.array([x[j] for j in range(x.shape[0]) if labels[j]==i])
            if len(iassoc)>0:
                evpoints[i]=func(iassoc)
        newlabels=np.argmin(cdist(x,evpoints),axis=1)
        if (labels==newlabels).all():
            break
        labels=np.copy(newlabels)
    return evpoints,labels

def pltdraw(evp,lab,xp):
    xevp=np.concatenate([xp,evp])
    kt=[[] for i in range(evp.shape[0])]
    for i in range(lab.shape[0]):
        kt[lab[i]].append(xp[i])
    dist=0
    for i in range(evp.shape[0]):
        if kt[i]!=[]:
            dist+=sum_distances([evp[i]],kt[i])
    return dist


import timeit
from statistics import mean,median
def main():
    tc=[
        #(10,6000),
        #(13,6000),
        #(15,6000),
        #(5,6000),
        (5,900),
        (6,1000),
        (7,2000),
        (8,2000),
        (9,4000)
    ]
    for k,s in tc:    
        p=np.random.sample((s,2))*500
        Tm = np.min(p,0)
        TM = np.max(p,0)
        e = max(TM[0] -Tm[0], TM[1]-Tm[1])
        cdobj=partial(cordinate_descent,a=1e-5,e=e)
        timelist=[]
        distlist=[]
        for i in range(10):
            start = timeit.default_timer()
            ev,l=k_objfunc(p,k,cdobj)
            end = timeit.default_timer()
            timelist.append(end - start)
            distlist.append(pltdraw(ev,l,p)[0])
        print(k,s,min(timelist),mean(timelist),median(timelist),\
            min(distlist),mean(distlist),median(distlist))
if __name__=="__main__":
    main()