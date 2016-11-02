from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9
from time import strftime
import emcee

def lnlike_1G(theta, y):
    mu, sig = theta 
    model = mu 
    return -(np.sum((y-model)**2/(2.0*sig**2) + np.log(np.sqrt(2.0*np.pi*sig**2))))

def lnprior_1G(theta):
    mu, sig  = theta 
    if -100.0 < mu < 100.0 and 0.00001 < sig < 1.0:
        return 0.0
    return -np.inf

def lnprob_1G(theta, y):
    lp = lnprior_1G(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_1G(theta, y)



def lnlike_GMM(theta, x, y, dim = 5 , n_A = None, dM_3 = None):
    if dim == 5:
        mu_1, sig_1, sig_2, dM, w_A = theta

    elif dim == 4:
        mu_1, sig_1, sig_2, dM = theta
        if n_A is not None:
            w_A = n_A

    elif dim == 3:
        mu_1, sig_1, sig_2 = theta
        if n_A is not None:
            w_A = n_A
        if dM_3 is not None:
            dM = dM_3

    mu_2 = mu_1 - dM
    population_A = w_A*np.exp(-0.5* (y-mu_1)**2./sig_1**2)/ (sig_1*np.sqrt(2.0*np.pi))
    population_B = (1.0 - w_A) * np.exp(-0.5* (y-mu_2)**2./sig_2**2)/(sig_2*np.sqrt(2.0*np.pi))
    return np.sum(np.log( population_A + population_B))

def lnprior_GMM(theta, dim = 5 , n_A = None, dM_3 = None):
    if dim == 5:
         mu_1, sig_1, sig_2, dM, w_A = theta
         if -25.50 < mu_1 < -15.50 and 0.01 < sig_1 < 1.0 and 0.000001 < dM < 5.0 and 0.01 < sig_2 < 1.0 and 0.2 < w_A < 0.80:
             return 0.0
    elif dim == 4 :
        mu_1, sig_1, sig_2, dM = theta
        if -25.50 < mu_1 < -15.50 and 0.01 < sig_1 < 1.0 and 0.000001 < dM < 5.0 and 0.01 < sig_2 < 1.0:
            return 0.0    
    elif dim == 3 :
        mu_1, sig_1, sig_2= theta
        if -25.50 < mu_1 < -15.50 and 0.01 < sig_1 < 1.0 and 0.01 < sig_2 < 1.0:
            return 0.0
    return -np.inf


def lnprob_GMM(theta,x, y, dim = 5 , n_A = None, dM_3 = None):
    lp = lnprior_GMM(theta, dim  , n_A , dM_3 )
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_GMM(theta, y,  dim , n_A , dM_3 )


def MCMC_1G(initial, M, nwalkers, step, a = 2.5, ball = True, savetxt = False):
    ndim = len(initial)
    if ball:
        pos0 = emcee.utils.sample_ball(initial, 0.1*np.ones(ndim), nwalkers)
    else :
        pos0 = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_1G, a, threads = 23, args= [M])
    pos, prob, state = sampler.run_mcmc(pos0, step) ## (initial position, burn in)
    sampler.reset()   ## reset chain to remove burn in
    pos, prob, state = sampler.run_mcmc(pos, step, rstate0=state)
    if savetxt:
        np.savetxt("%s_absmag_1G_%s_st_%s_d_%s_w_50data_chain.dat"%(strftime('%Y%m%d'),step, ndim, nwalkers), sampler.chain[:, :, :].reshape((-1, ndim)))
    af = sampler.acceptance_fraction
    print "Acceptance rate:", np.mean(af)
    return sampler


def MCMC_GMM(initial, z, M, nwalkers, step, a, ball = True, savetxt = False):
    #M, z = y
    ndim = len(initial)
    if ball:
        pos0 = emcee.utils.sample_ball(initial, 0.1*np.ones(ndim), nwalkers)
    else :
        pos0 = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_GMM(ndim), a, threads =23, args=[z, M] )
    pos, prob, state = sampler.run_mcmc(pos0, step) ## (initial position, burn in)
    sampler.reset()   ## reset chain to remove burn in
    pos, prob, state = sampler.run_mcmc(pos, step, rstate0=state)    
    af = sampler.acceptance_fraction
    print "Acceptance rate:", np.mean(af)
    return sampler

    

def iter_plot( labels, thin,nwalkers, step, sampler, kind, savefig = False ):
    ndim = len(labels)
    fig = plt.figure(figsize=(16,9))
    values = results
    loop = 1

    # Creates plot to show position at certain iterations
    for i in range(ndim):
        plt.subplot(ndim,ndim,loop)
        new_array = np.zeros(nwalkers*step/thin) #nwalkers*step
        for j in range (0, nwalkers*step/thin):
            new_array[j] = sampler.flatchain[j+thin, i]
        plt.plot(new_array)#,marker='.',linestyle='none')
        plt.xlabel('Iteration')
        plt.xticks(rotation = 45)
        plt.ylabel(labels[i])
        loop += 1

    # Creates histogram of 1-D probabilites. Same thing as in the corner plots.
    for i in range(ndim):
        plt.subplot(ndim,ndim,loop)
        plt.hist(sampler.flatchain[:,i], 50)
        plt.xlabel(labels[i])
        plt.xticks(rotation = 45)
        plt.ylabel('Frequency')
        #plt.axvline(values[i], color='red', linewidth=4)
        loop += 1
    plt.tight_layout()
    if savefig:
        fig.savefig("%s_absmag_%s_%s_st_%s_d_%s_w_50data_%s_thin_iter.pdf"%(strftime('%Y%m%d'),step, kind, ndim, nwalkers, thin))
    return
