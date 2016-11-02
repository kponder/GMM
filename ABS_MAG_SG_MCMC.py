import numpy as np
import emcee
from time import strftime
import os
import sys

def lnlike(theta, y, yerr):
    mu, sig = theta 
    model = mu 
    var = yerr*yerr + sig*sig
    return (np.sum(np.log(np.exp(-(y-model)**2/(2.0*sig**2))/np.sqrt(2.0*np.pi*sig**2))))

def lnprior(theta):
    mu, sig  = theta 
    if -100.0 < mu < 100.0 and 0.00001 < sig < 2.0: 
        return 0.0
    return -np.inf

def lnprob(theta, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, y, yerr)



data_file_name = sys.argv[1]
datadir='/Users/karaponder/Spring14/Data_Sets' #'/global/u2/k/kap146/carver/GMM_Likelihood/Mock_Data' #

datafile_1G =os.path.join(datadir,data_file_name)

M = np.loadtxt(datafile_1G, unpack = True, usecols=[0])

M_true, sig_true = -19.5, 0.15
results = M_true , sig_true
sigphot = 0.1
ndim, nwalkers, step = 2, 50, 100
pos0 = [results + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,threads =8, a =5.5, args=(M, sigphot))

pos, prob, state = sampler.run_mcmc(pos0, step) ## (initial position, burn in)
sampler.reset()   ## reset chain to remove burn in
pos, prob, state = sampler.run_mcmc(pos, step, rstate0=state)

af = sampler.acceptance_fraction
af_ave = np.mean(af, dtype = np.float64)
print af_ave
#f = open('/global/scratch2/sd/kap146/Output_Data/%s_acceptance_fraction_ab1G_%s_st_%s_d_%s_w_%s_data.dat'%(strftime('%Y%m%d'),step, ndim, nwalkers, len(M)), 'w')
#f.write('%d' % af_ave)
#f.close()

fit = np.mean(pos, axis = 0, dtype=np.float64)
#np.savetxt('/global/scratch2/sd/kap146/Output_Data/%s_est_param_ab1G_%s_st_%s_d_%s_w_%s_data.dat'%(strftime('%Y%m%d'),step, ndim, nwalkers, len(M)), (fit))

# Flatten samples
samples = sampler.chain[:, :, :].reshape((-1, ndim))
#np.savetxt("/global/scratch2/sd/kap146/Output_Data/%s_absmag_1G_%s_st_%s_d_%s_w_%s_data_samples.dat"%(strftime('%Y%m%d'),step, ndim, nwalkers, len(M)), samples)
