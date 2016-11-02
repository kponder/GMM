# Fits cosmological parameters using a single Gaussian model using emcee hammer
# packages needed: numpy, math, emcee, acor, triangular_plot, os, matplotlib, scipy, astropy
from __future__ import division

import numpy as np
import emcee
import scipy.integrate as integrate
import math
from time import strftime
import sys
from scipy.interpolate import interp1d

c1 = 2.9979E5 #km / s

import os
data_file_name = sys.argv[1]
datadir='/global/u2/k/kap146/carver/GMM_Likelihood/Mock_Data' #/global/scratch2/sd/kap146/Input_Data'
#out_datadir = '/global/scratch2/sd/kap146/Output_Data'
datafile_1G =os.path.join(datadir, data_file_name)

M, z, dm = np.loadtxt(datafile_1G, unpack = True, usecols=[0, 1, 2])

## if len(M) >= 100000:
##     def distmod(Om, Ol, w, x):
##         xmod = np.linspace(0.000, 2.0, 10000*50+1) 

##         Ok = 1.0-Om-Ol
##         test = lambda z: (Om*(1.0+z)**3. + Ok*(1.0+z)*(1.0+z) + Ol *  (1.0+z)**(3.0 * (1.0 + w)))**(-0.5)

##         dE = test(xmod)
##         Efoo = integrate.cumtrapz(dE, xmod) #, initial=0)
##         Efoo = np.insert(Efoo, 0, 0.0)
##         f = interp1d(xmod, Efoo)

##         E = f(x)

##         if Ok > 0. :
##             a = (1.0+x)*np.sinh(np.abs(Ok)**0.5*E)/np.abs(Ok)**0.5
##         elif Ok < 0. :
##             a = (1.0+x)*np.sin( np.abs(Ok)**0.5*E)/np.abs(Ok)**0.5
##         elif Ok == 0. :
##             a = (1.0+x)*E 

##         return 5.0 *np.log10(a)
## else:
##     def distmod(Om, Ol, w, x):
##         Ok = 1.0-Om-Ol
##         a = np.zeros(len(x))
##         test = lambda x: (Om*(1.0+x)**3. + Ok*(1.+x)*(1.+x) + Ol *  (1. + x)**(3. * (1. + w)))**(-0.5)
##         for i in xrange(0, len(x)):
##             try:
##                 E = integrate.romberg(test, 0., x[i])
##             except ValueError:  # Catch the non-physical regions
##                 E = np.nan
##             if Ok > 0. :
##                 a[i] = (1.0 + x[i])*np.sinh(np.abs(Ok)**0.5*E)/np.abs(Ok)**0.5
##             elif Ok < 0. :
##                 a[i] = (1.0+x[i])*np.sin(np.abs(Ok)**0.5*E)/np.abs(Ok)**0.5
##             elif Ok == 0. :
##                 a[i] = (1.0+x[i])*E  #integrate.romberg(test, 0.0, x)
##         return 5.0 *np.log10(a)
from decimal import Decimal
import lumdist as ld
def distmod(Om,Ol,w,x):
    b = np.zeros(len(x))
    Ok = 1.00-Om-Ol
    a = 1./(1.+x)
    for i in xrange(0, len(x)):
        try:
           E = ld.chi_qwa(1, a[i],np.array([Om,Ol,w,Decimal(0.0)]),0.0000001)
        except ValueError:  # Catch the non-physical regions
           E = np.nan
        if Ok > 0. :
            b[i] = (1.0 + x[i])*np.sinh(np.abs(Ok)**0.5*E)/np.abs(Ok)**0.5
        elif Ok < 0. :
            b[i] = (1.0+x[i])*np.sin(np.abs(Ok)**0.5*E)/np.abs(Ok)**0.5
        elif Ok == 0. :
            b[i] = (1.0+x[i])*E
    return 5.0 *np.log10(b)

# Define log likelihood function. Can add varying number of fitted parameters
def lnlike(theta, x, y, yerr):
    z = x
    Omega_M, Omega_L, w, sig_int, script_M = theta
    var = yerr*yerr + sig_int*sig_int
    model = script_M + distmod(Omega_M, Omega_L, w, z) + 5.0*np.log10(c1)
    population = np.exp(-0.5* (y-model)**2./var) / (np.sqrt(2.0*np.pi*var))  
    return np.sum(np.log(population)) 


# Define log prior on parameters. Here I have use a flat prior
def lnprior(theta):
    Omega_M, Omega_L, w, sig_int, script_M = theta
    if -15.0 < script_M < 5.0 and 0.001 < Omega_M < 1.0 and 0.001 < Omega_L < 1.0 and -3.0 < w < 1.0 and 0.0 < sig_int < 0.3:
        return 0.0
    return -np.inf


# Define log probability function that includes the log prior and log likelihood
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
    


#m = M + dm     
#m = -19.5 + dm_for_data     ### for practice with standard model

sig = 0.10 # Don't fit for sig because it is intrinsic to SN. #np.abs(sig)

# Set true values.
script_M_true, Omega_M_true, Omega_L_true, w_true, sig_int_true  = -3.6, 0.30, 0.70, -0.98, 0.14 #-3.704, 0.30, 0.70, -1.0, 0.15


# Set starting position for parameters. Different options for fitting different parameters
initial =Omega_M_true, Omega_L_true, w_true, sig_int_true, script_M_true


### Beginning of MCMC

# Define number of walkers: must be an even number and at least twice the number of dimensions. Should be hundreds
# Define number of steps FOR EACH WALKER

ndim, nwalkers = len(initial), 500
step = int(sys.argv[2])
thin_f = int(sys.argv[3])
# Create the initial position for each walker.
pos0 = [initial + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]

# Define the way to sample. This could be the normal Goodman-Weare samples, parallel-tempering, Metropolis-Hastings, or you could define your own.
## a is by default set to 2.0. This is the only parameter that controls percent accepted.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a = 2.5,threads = 8, args = (z, dm, sig))


# Run MCMC.
### Burn in run
## outputs position, probability, and random number seed
pos, prob, state = sampler.run_mcmc(pos0, step/10, thin = 2) ## (initial position, number of steps)
## reset chain to remove burn in
sampler.reset()   ## reset chain to remove burn in

# Starting from the final position in the burn-in chain run the remainder of MCMC and start at the random number seed that the burn in left off at.
pos, prob, state = sampler.run_mcmc(pos0, step, rstate0=state, thin=thin_f)

# Calculates percent accepted. and tells you how to change. If it is around 0.0, you have a bigger problem.
af = sampler.acceptance_fraction
#np.savetxt('/global/scratch2/sd/kap146/Output_Data/%s_accepatance_fraction_1G_%s_st_%s_d_%s_w_%s_data.dat'%(strftime('%Y%m%d'),step, ndim, nwalkers, len(M)), af)
af_arr = np.mean(af, dtype = np.float64)
print af_arr

f = open('/global/scratch2/sd/kap146/Output_Data/%s_accepatance_fraction_1G_%s_st_%s_d_%s_w_%s_data_0.dat'%(strftime('%Y%m%d'),step,ndim, nwalkers, len(M)), 'w')
f.write('%.2f' % af_arr)
f.close()

## Print the estimated values from the MCMC
fit = np.mean(pos, axis = 0, dtype=np.float64)
np.savetxt('/global/scratch2/sd/kap146/Output_Data/%s_est_param_1G_%s_st_%s_d_%s_w_%s_data_0.dat'%(strftime('%Y%m%d'),step, ndim, nwalkers, len(M)), (fit))

######################### Plotting #########################
# Flatten samples
samples = sampler.chain[:, :, :].reshape((-1, ndim))
np.savetxt("/global/scratch2/sd/kap146/Output_Data/%s_cosmofit_1G_%s_st_%s_d_%s_w_%s_data_samples_0.dat"%(strftime('%Y%m%d'),step, ndim, nwalkers, len(M)), samples)

