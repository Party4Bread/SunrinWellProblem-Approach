import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.spatial import ConvexHull
from sklearn.datasets import load_iris

import random
import timeit
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
    evpoints=np.random.choice(x,k,False)
    labels=np.argmin(cdist(x,evpoints),axis=1)
    #labels=[np.argmin(np.linalg.norm(evpoints-i,axis=0)) for i in x]
    pltdraw(evpoints,labels,x)
    #assign init labels
    while curiter<maxiter:
        curiter+=1
        #calc evpoints
        for i in range(k):
            iassoc=np.array([x[j] for j in range(x.shape[0]) if labels[j]==i])
            if len(iassoc)>0:
                evpoints[i]=func(iassoc)

        newlabels=np.argmin(cdist(x,evpoints),axis=1)
        pltdraw(evpoints,newlabels,x)
        plt.savefig('{1}\\{0:05d}.png'.format(curiter,titi))
        if (labels==newlabels).all():
            break
        labels=np.copy(newlabels)
    return evpoints,labels

def pltdraw(evp,lab,xp):
    xevp=np.concatenate([xp,evp])
    xy=np.split(xevp,2,1)
    x=np.concatenate(xy[0])
    y=np.concatenate(xy[1])
    c=np.concatenate([lab/(evp.shape[0]+1),np.ones(evp.shape[0])])
    plt.clf()
    plt.scatter(x,y,np.ones(len(xy))*50,c)
    kt=[[] for i in range(evp.shape[0])]
    for i in range(lab.shape[0]):
        kt[lab[i]].append(xp[i])
    dist=0
    for i in range(evp.shape[0]):
        if kt[i]!=[]:
            dist+=sum_distances([evp[i]],kt[i])
    #dist=np.sum(sum_distances(evp,xp))
    plt.title(str(dist))
    kt=np.array(kt)
    for i in kt:
        if len(i)<3:
            continue
        i=np.array(i)
        hull = ConvexHull(i)
        for simplex in hull.simplices:
            plt.plot(i[simplex,0], i[simplex,1], 'k-')

import os

def main():
    global titi
    from functools import partial
    k=10
    s=600
    p = load_iris()
    p = p.data.reshape(300,2)
    #p=np.random.sample((s,2))*100
    plt.figure(figsize=(14,14))
    #print(p)
    Tm = np.min(p,0)
    TM = np.max(p,0)
    e = max(TM[0] -Tm[0], TM[1]-Tm[1])
    cdobj=partial(cordinate_descent,a=1e-5,e=e)
    titi="0206pics\\"+strftime("%Y-%m-%d_%H_%M_%S", gmtime())+"k%ds%d"%(k,s)
    os.mkdir(titi)

    ev,l=k_objfunc(p,k,cdobj)
    pltdraw(ev,l,np.array(p))
    
if __name__=="__main__":
    main()