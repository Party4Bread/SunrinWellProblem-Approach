import numpy as np
from scipy.spatial.distance import cdist, euclidean

def euclidean_distance(a,b):
    return np.linalg.norm(a-b)

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

def k_objfunc(x,k,func,maxiter=1e4,eps=0.1):
    #init
    label=np.zeros(x.shape[0])
    evpoints = np.random.sample((k, x.shape[1]))
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
            dist=np.linalg.norm(j-evpoints,axis=0)
            label[i]=np.argmin(dist)
        
        # recalc evpoints
        for i in range(k):
            assocpoints = [x[j] for j in range(x.shape[0]) if label[j] == i]
            if assocpoints:
                evpoints[i] = func(np.array(assocpoints))
        #np.sort(oldevpoints,axis=1)
        #print(oldevpoints,'\n', evpoints,'\n\n')
        err=abs(np.sum(oldevpoints-evpoints))
        #print(oldevpoints,evpoints,'\n\n')
    return evpoints,label

def main():
    from functools import partial
    p=np.array([[1.,1.],[1.,3.],[4,2],[4.,4.]])
    Tm = np.min(p,0)
    TM = np.max(p,0)
    e = max(TM[0] -Tm[0], TM[1]-Tm[1])
    cdobj=partial(cordinate_descent,a=1e-5,e=e)
    print(k_objfunc(p,4,cdobj))
    
if __name__=="__main__":
    main()
