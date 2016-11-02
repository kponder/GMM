from __future__ import division
import numpy as np
import emcee
import math
from time import strftime
import sys
from scipy.special import betainc, gamma

c1 = 2.9979E5 #km / s


# In[2]:

import os
data_file_name = sys.argv[1]
datadir='/global/u2/k/kap146/carver/GMM_Likelihood/Mock_Data' #/global/scratch2/sd/kap146/Input_Data'
#out_datadir = '/global/scratch2/sd/kap146/Output_Data'
datafile_1G =os.path.join(datadir, data_file_name)


# In[3]:

z1, dm1 = np.loadtxt(datafile_1G, unpack = True, usecols=[1, 2])

combined = np.column_stack((z1, dm1))
combined_sort = combined[np.argsort(combined[:, 0])]
z, dm = combined_sort[:,0], combined_sort[:,1]


# In[4]:

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


# In[5]:

# Define log likelihood function.
def lnlike(theta, x, y, yerr):
    z = x #s, N_bins
    Omega_M, w, sig_int_A, sig_int_B, script_M_1, dM, c, d = theta
    n_A = c*z + d 
    sig=0.15      ## should be fitted for before this stage and read in as a "yerr"
    if n_A[0] > 1.0 or n_A[0] < 0.0 or n_A[-1] > 1.0 or n_A[-1] < 0.0: #if any(t < 0.0 or t > 1.0 for t in w_A):
        return -np.inf    
    #script_M_1 = -19.5 + 25.0+ 5.0*np.log10(1.0/70.0)      ## comment out when fitting for this
    #script_M_2 = -19.5 + 25.0 + 5.0*np.log10(1.0/70.0)     ## comment out when fitting for this

    dist_mod = distmod(Omega_M, w, z)
    
    model_1 = script_M_1 + dist_mod + 5.0*np.log10(c1)
    model_2 = script_M_1 - dM + dist_mod + 5.0*np.log10(c1)

    var_A = yerr*yerr + sig_int_A*sig_int_A
    var_B = yerr*yerr + sig_int_B*sig_int_B
    
    population_A = n_A*np.exp(-0.5* (y-model_1)**2./var_A)/ (np.sqrt(2.0*np.pi*var_A))              # Gaussian PDF
    population_B = (1.0 - n_A) * np.exp(-0.5* (y-model_2)**2./var_B)/(np.sqrt(2.0*np.pi*var_B))
    return np.sum(np.log( population_A + population_B))


# In[6]:

def lnprior(theta):
    Omega_M, w, sig_int_A, sig_int_B, script_M_1, dM, c, d = theta
    if 0.0001 < Omega_M < 1.0 and -3.0 < w < 0.01 and -15.0 < script_M_1 < 5.0 and 0.00001 < dM < 5.0 and -1.0 < c < 0.00001 and 0.000001 < d < 2.0 and 0.0 < sig_int_A < 0.3 and 0.0 < sig_int_B < 0.3 :
        return 0.0
    return -np.inf


# In[7]:

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    LL = lnlike(theta, x, y, yerr)
    if np.isnan(LL):
        return -np.inf
    return lp + LL
    


# In[8]:

sig = 0.10 # assuming know SN very well


script_M_1_true, dM_true,alpha_true, Omega_M_true, Omega_L_true, w_true, c_true, d_true, sig_A_true, sig_B_true  = -3.604, 1.0, 1.14,  0.28, 0.72, -1.0, -0.69, 1.03, 0.1, 0.1


initial = Omega_M_true, w_true, sig_A_true, sig_B_true, script_M_1_true,  dM_true, c_true, d_true


# In[9]:

ndim, nwalkers = len(initial), 500
step = int(sys.argv[2])
thin_f = int(sys.argv[3])
pos0 = [initial + 1e-1*np.random.randn(ndim) for i in range(nwalkers)] # usually 1e-4


# In[15]:

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads = 8, args = (z,dm, sig), a =2.50)


# In[16]:

## outputs position, probability, and random number seed
pos, prob, state = sampler.run_mcmc(pos0, step/10, thin = 2) ## (initial position, number of steps)
sampler.reset()   ## reset cha


# In[17]:

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

