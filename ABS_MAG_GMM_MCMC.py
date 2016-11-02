import numpy as np
import emcee
from time import strftime
import sys

def lnlike(theta,x, y, yerr):
    mu_1,sig_int_A, sig_int_B, dM, c, d = theta
    n_A = c * x + d
    if n_A[0] > 1.0 or n_A[0] < 0.0 or n_A[-1] > 1.0 or n_A[-1] < 0.0:
        return -np.inf
    mu_2 = mu_1 - dM
    var_A = yerr*yerr + sig_int_A*sig_int_A
    var_B = yerr*yerr + sig_int_B*sig_int_B
    population_A = n_A*np.exp(-0.5* (y-mu_1)**2./var_A)/ (np.sqrt(2.0*np.pi*var_A))
    population_B = (1.0 - n_A) * np.exp(-0.5* (y-mu_2)**2./var_B)/(np.sqrt(2.0*np.pi*var_B))
    return np.sum(np.log( population_A + population_B))

def lnprior(theta): 
    mu_1, sig_1, sig_2, dM, c, d = theta
    if -25.50 < mu_1 < -15.50 and 0.01 < sig_1 < 1.0 and 0.000001 < dM < 5.0 and 0.01 < sig_2 < 1.0 and - 1.0 < c < 0.0001 and 0.000001 < d < 2.0:
        return 0.0
    return -np.inf

def lnprob(theta,x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


import os
data_file_name = sys.argv[1]
datadir='/Users/karaponder/Spring14/Data_Sets' #'/global/u2/k/kap146/carver/GMM_Likelihood/Mock_Data'
datafile_1G =os.path.join(datadir,data_file_name)
M1, z1 = np.loadtxt(datafile_1G, unpack = True, usecols=[0, 1])
combined = np.column_stack((M1, z1))
combined_sort = combined[np.argsort(combined[:, 1])]
M, z = combined_sort[:,0], combined_sort[:,1]

M_1_true, sig_1_true, sig_2_true, dM_true, c_true, d_true = -19.45, 0.15, 0.15, 0.1, -0.62, 1.003

#print M
results = [M_1_true, sig_1_true, sig_2_true, dM_true, c_true, d_true ]

sigphot = 0.1

ndim, nwalkers, step = len(results), 50, 100
#pos0 = [results + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
pos0 = emcee.utils.sample_ball(results, 0.1*np.ones(ndim), nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,threads =8, args=(z, M, sigphot), a = 3.0 )

pos, prob, state = sampler.run_mcmc(pos0, step) ## (initial position, burn in)
sampler.reset()   ## reset chain to remove burn in
# Starting from the final position in the burn-in chain, sample for 1000
# steps.
pos, prob, state = sampler.run_mcmc(pos, step, rstate0=state)

af = sampler.acceptance_fraction
af_ave = np.mean(af, dtype = np.float64)
print af_ave
#f = open('/global/scratch2/sd/kap146/Output_Data/%s_accept_fraction_abGMM_%s_st_%s_d_%s_w_%s_data.dat'%(strftime('%Y%m%d'),step, ndim, nwalkers, len(M)), 'w')
#f.write('%.2f' % af_ave)
#f.close()

fit = np.mean(pos, axis = 0, dtype=np.float64)
#np.savetxt('/global/scratch2/sd/kap146/Output_Data/%s_est_param_abGMM_%s_st_%s_d_%s_w_%s_data.dat'%(strftime('%Y%m%d'),step, ndim, nwalkers, len(M)), (fit))

# Flatten samples
samples = sampler.chain[:, :, :].reshape((-1, ndim))
#np.savetxt("/global/scratch2/sd/kap146/Output_Data/%s_absmag_GMM_%s_st_%s_d_%s_w_%s_data_samples.dat"%(strftime('%Y%m%d'),step, ndim, nwalkers, len(M)), samples)
