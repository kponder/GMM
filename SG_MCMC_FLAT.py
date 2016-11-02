from __future__ import division
import numpy as np
import emcee
import scipy.integrate as integrate
from scipy.special import betainc, gamma
import math
from time import strftime
import sys


c1 = 2.9979E5 #km / s


# In[2]:

import os
data_file_name = sys.argv[1]
datadir='/global/u2/k/kap146/carver/GMM_Likelihood/Mock_Data' #/global/scratch2/sd/kap146/Input_Data'
#out_datadir = '/global/scratch2/sd/kap146/Output_Data'
datafile_1G =os.path.join(datadir, data_file_name)

M, z, dm = np.loadtxt(datafile_1G, unpack = True, usecols=[0, 1, 2])



# In[5]:

def distmod(Om,w,z):
    Ol = 1.0 - Om
    x0 = Ol/(Om+Ol)
    x = Ol*(1.+z)**(3.*w)/(Om + Ol*(1.+z)**(3.*w))
    m = -1./(6.*w)
    bx0 = betainc(m, 0.5-m, x0)*gamma(m)*gamma(0.5-m)/gamma(0.5)
    bx = betainc(m, 0.5-m, x)*gamma(m)*gamma(0.5-m)/gamma(0.5)
    r = 2.*m/np.sqrt(Om) * (Om/Ol)**m *(bx0 - bx)
    b = (1.0+z)*r
    return 5.0*np.log10(b)

# Define log likelihood function.
def lnlike(theta, x, y, yerr):
    z = x
    Omega_M, w, sig_int, script_M = theta
    var = yerr*yerr + sig_int*sig_int
    model = script_M + distmod(Omega_M, w, z) + 5.0*np.log10(c1)
    population = np.exp(-0.5* (y-model)**2./var) / (np.sqrt(2.0*np.pi*var))  
    return np.sum(np.log(population)) 


# In[6]:

def lnprior(theta):
    Omega_M, w, sig_int, script_M = theta
    if -15.0 < script_M < 5.0 and 0.001 < Omega_M < 1.0 and -3.0 < w < 1.0 and 0.0 < sig_int < 0.3:
        return 0.0
    return -np.inf


# In[7]:

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)
    


# In[8]:

sig = 0.10 # assuming know SN very well


script_M_true, Omega_M_true, w_true, sig_int_true  = -3.6, 0.30,  -0.98, 0.14 #-3.704, 0.30, 0.70, -1.0, 0.15


# Set starting position for parameters. Different options for fitting different parameters
initial = Omega_M_true,  w_true, sig_int_true, script_M_true


# In[13]:

ndim, nwalkers = len(initial), 50
step = 100
thin_f = 1
pos0 = [initial + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]


# In[14]:

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a = 2.5,threads = 8, args = (z, dm, sig))

# In[15]:

## outputs position, probability, and random number seed
pos, prob, state = sampler.run_mcmc(pos0, step/10, thin = 2) ## (initial position, number of steps)
sampler.reset()   ## reset cha


# In[16]:

pos, prob, state = sampler.run_mcmc(pos0, step, rstate0=state, thin = thin_f)

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
