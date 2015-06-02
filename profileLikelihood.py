# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:53:24 2015

@author: ross
This is a port of the R code found here:
## Time-stamp: <getElbows.R zma 2010-01-17 12:05>
## from https://github.com/dpmcsuss/graph-classify-code/blob/master/R/getElbows.R

"""
import numpy as np
import scipy.stats as stat
from random import uniform
import matplotlib.pyplot as plt
import sys

def getElbows(d,n=3,threshold = False):
    """
    Given a decreasingly sorted vector, return the given number of elbows

    Args:
        d: the decreasingly sorted vector (e.g. a vector of standard deviations)
        n: the number of returned elbows.
        threshold: either FALSE or a number. If threshold is a number, then all
        the elements in d that are not larger than the threshold will be ignored.

    Return:
        q: a vector of length n.

    Reference:
        Zhu, Mu and Ghodsi, Ali (2006), "Automatic dimensionality selection from
        the scree plot via the use of profile likelihood", Computational
        Statistics & Data Analysis
    """
    dnorm = stat.norm
    #d=D.sort(reverse=True)

    if type(threshold) != bool:
        d = d[d > threshold]

    p = len(d)
    if p <= 1:
        sys.exit("d must have elements that are larger than the threshold ")
        #print("d must have elements that are larger than the threshold ", threshold, "!")

    lq = np.zeros(p)
    dnorm = stat.norm
    for q in np.arange(1,p+1):
        if q == p:
            #print("q",q)
            mu1 = np.mean(d[:q])
            mu2 = np.mean(0)
            sigma2 = (np.sum((d[:q] - mu1)**2)) / (p - 1 - (q < p))
            lq[q-1] = np.sum( dnorm(mu1, np.sqrt(sigma2)).logpdf(d[:q]) )
        else:
            mu1 = np.mean(d[:q])
            mu2 = np.mean(d[q:])              # = NaN when q = p
            sigma2 = (np.sum((d[:q] - mu1)**2) + np.sum((d[q:] - mu2)**2)) / (p - 1 - (q < p))
            lq[q-1] = np.sum( dnorm(mu1, np.sqrt(sigma2)).logpdf(d[:q]) ) + np.sum( dnorm(mu2, np.sqrt(sigma2)).logpdf(d[q:]) )


   #q = which.max(lq)
   #if (n > 1 and q < p):
   #    return c(q, q + getElbows(d[(q+1):p], n-1))
   #else:
    #return np.argmax(lq)
    return lq

if __name__ == '__main__':
    a = [uniform(0,45) for p in range(0,50)]
    a.sort(reverse=True)
    b = [uniform(55,100) for p in range(0,50)]
    b.sort(reverse=True)
    b.extend(a)
    zz = getElbows(b,1)

    fig = plt.figure(1)
    plt.scatter(np.arange(1,101),zz, color='b') #titles, documents
    plt.show()

