import numpy as np
from scipy.spatial.distance import cdist, euclidean
import random
import timeit

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

def runtime(f):
    def wrapper(*args, **kwargs):
        import timeit
        start = timeit.default_timer()
        res=f(*args)
        end = timeit.default_timer()
        print(end - start)
        return res
    return wrapper

#@runtime
def k_objfunc(x,k,func,maxiter=1e3):
    curiter=0
    Tm = np.min(x,0)
    TM = np.max(x,0)
    evpoints=((TM-Tm)*np.random.sample((k,2)))+Tm #init func
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
    return evpoints,labels



def main():
    from functools import partial
    p=np.random.sample((7,2))*1000
    print(p)
    #p=np.array([[1.,1.],[1.,3.],[8.,2.],[8.,4.]])
    Tm = np.min(p,0)
    TM = np.max(p,0)
    e = max(TM[0] -Tm[0], TM[1]-Tm[1])
    cdobj=partial(cordinate_descent,a=1e-5,e=e)

    print(k_objfunc(p,3,cdobj))
    
if __name__=="__main__":
    main()