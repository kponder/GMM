#!/usr/bin/python

##  Callable module to implement the Gelman-Rubin test and Autocorrelation tests

import numpy as np
from time import strftime
import matplotlib.pyplot as plt
import emcee

## Gelman-Rubin Test

def W(m ,n , all_pos):
    s = np.zeros(len(all_pos))
    for i in xrange(0, len(all_pos)):
        s[i] = 1.0/(n-1.0) * np.sum( (all_pos[i]- np.mean(all_pos[i]))*(all_pos[i]- np.mean(all_pos[i]))) ## variance for a given chain
    return 1.0/m * np.sum(s)

def B(m, n, all_pos): 
    chain_mean = np.zeros(len(all_pos))
    for i in xrange(0, len(all_pos)):
        chain_mean[i] = np.mean(all_pos[i])
    theta_bb = 1.0/ m * np.sum(chain_mean)
    return n/(m-1.0) * np.sum( (chain_mean - theta_bb)*( chain_mean - theta_bb) )

def Var(m ,n , all_pos):
    return (1.0 - 1.0/n)*W(m ,n , all_pos) + 1.0/n * B(m ,n , all_pos)

def R_hat(m ,n , all_pos):
    return np.sqrt(Var(m ,n , all_pos)/W(m ,n , all_pos))

# <codecell>

# Correlation function test

def correlation(k, pos):
    ave = np.mean(pos)
    n = len(pos)
    num =0.0
    for i in xrange(0, n-k):   
        num += (pos[i] - ave)*(pos[i+k] - ave)
    
    den = np.sum((pos - ave)*(pos-ave))
    return num/den
    

def multi_corr(x, pos):
    y = np.zeros(len(x))
    for i in xrange(0, len(x)):
        #x[i]= int(x[i])
        y[i] = correlation(int(x[i]), pos)
    #plt.plot(x, y)
    return y
                 

# <codecell>


def plotting(labels, y, sampler, savefig = False):
    for i in xrange(0, len(labels)):
        plt.figure()
        interm = multi_corr(y, sampler[:,i])
        plt.plot(y, interm, '.')
        plt.vlines(y, 0, interm)
        plt.xlabel('%s'%labels[i])
        if savefig:
            dim = len(labels)
            step = len(sampler)
            if dim > 5:
                kind = 'GMM'
            else:
                kind = '1G'
            plt.savefig('%s_autocorr_%s_%s_d_%s_step_%s.pdf'%(strftime('%Y%m%d'),labels[i], dim, step, kind))
        plt.close()
    return "I have created auto correlation plots for ", labels

