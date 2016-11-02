# Generate GMM data sets with different Delta M
from __future__ import division
import numpy as np
from time import strftime
from astropy.cosmology import WMAP9

# Relative normalization of population A 
def n_A (z):
    return  - 0.62695924764890282131661442 * z + 1.0031347962382445141065830721 #- 0.689655172414 * z + 1.034482758621

## This format is based on Rodrigo Nemmen da Silva's code. Taken from:
### http://astropython.blogspot.com/2011/12/generating-random-numbers-from.html
def randomvariate(pdf, p, n=1000, xmin=0, xmax=1):
    #mu_A, sig_A, mu_B, sig_B = p
    shift_M, sig_A, sig_B = p
    x=np.linspace(xmin, xmax, 1000)
    pmin= pdf(x, p).min()
    pmax= pdf(x, p).max()
    naccept = 0
    ntrial = 0
    ran=[]
    while naccept < n:
        x=np.random.uniform(xmin,xmax)
        y=np.random.uniform(pmin, pmax)
        if y< pdf(x, p):
            ran.append(x)
            naccept=naccept+1
        ntrial=ntrial+1
    ran=np.asarray(ran)
    
    return ran, ntrial


def mock_data(z, p, N, M_ave = -19.5, flatten = False, save_redshiftbin = False, save_flatten = False, series = 0):
    """
    Generate a data set with any redshift range, intrinsic standard deviation, Delta M, absolute Magnitude and save the data in two different forms!

    Parameters
    ----------
    z : array_like
        redshift range

    p : array_like
        Takes the form np.array([ delta_M, sigma_A, sigma_B ] )

    N : array_like
        Number of supernova in simulation. Only get right number if N/len(z) is 

    M_ave : array_like (optional)
        Current estimate for the average Absolute Magnitude of SNeIa

    flatten : bool (option)
        Return flattened array of generated SNeIa. If false, return SNeIa binned by redshift

    save_redshiftbin : bool (optional)
        Saves the generated SNe in bins of redshift. Good for plotting histograms with redshift color coding.

    save_flatten : bool (optional)
        Saves the generated SNe in a flattened array. Also keeps track of associated redshift. Good for MCMC fits

    Returns
    -------
    Generated supernovae still binned by redshift

    """
  
    M1=[]
    p = np.array(p)
    num_in_bins=(N+0.0)/(len(z)+0.0)
    for i in range(0, len(z)):
        def GMM(x, p):
            shift_M, sig_A, sig_B = p
            mu_A= M_ave + shift_M/2.0  # Mean for distribution A
            mu_B= M_ave - shift_M/2.0  # Mean for distribution B
            p_A = np.sqrt(2*np.pi*sig_A**2)**(-1.0)*np.exp(-(x-mu_A)**2/(2*sig_A**2)) ##sig**2=variance
            p_B = np.sqrt(2*np.pi*sig_B**2)**(-1.0)*np.exp(-(x-mu_B)**2/(2*sig_B**2))
            return (n_A(z[i])*p_A + (1-n_A(z[i])) * p_B)#/(p_A + weight * p_B).sum()
        M,trials = randomvariate( GMM, p, n=num_in_bins, xmin = M_ave - 2.0, xmax = M_ave + 2.0)
        M1.append(M)
    if num_in_bins != int(num_in_bins) + 0.0:
        print "Number of supernova divided number of redshift bins does NOT produce an integer. Your new total number of supernova is: ", len(M)*len(z)
    shift_M, sig_A, sig_B = p
    if save_redshiftbin:
        np.savetxt("/Users/karaponder/Spring14/Data_Sets/%s_data_set_gen_testing_%s_SNnum_%.0f_NumBin_%s_dM_%s.dat"%(strftime('%Y%m%d'),N, len(M),shift_M, series), (M1))
    if flatten:
        S = np.ndarray.flatten(np.array(M1))
        if save_flatten:
            z1 = np.repeat(z, num_in_bins)
            distmod = S + WMAP9.distmod(z1).value
            np.savetxt("/Users/karaponder/Spring14/Data_Sets/%s_flatten_data_%s_SNnum_phot01_%.0f_%.0f_%.0f_%s.dat"%(strftime('%Y%m%d'), N, shift_M*100,sig_A*100,sig_B*100, series),  np.column_stack((S, z1, distmod)))
            print "File saved as: ", "%s_flatten_data_%s_SNnum_phot01_%.0f_%.0f_%.0f_%s.dat"%(strftime('%Y%m%d'), N, shift_M*100,sig_A*100,sig_B*100, series)
        return S
    else:
        return M1

