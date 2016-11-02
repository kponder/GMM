#### Code to Generate Hubble Diagram with residual for both SGM and GMM with outputs from MCMC ###
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.gridspec as gridspec
from astropy.cosmology import WMAP9
from matplotlib.ticker import MaxNLocator
from scipy.special import betainc, gamma
from time import strftime


#### Define the distance modulus function. 
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


## Define the directory for the data which is the reporistory's "Data" directory. 
datadir='../Data'

### Load mock data set used for parameters
mock = '20140806_flatten_data_10000_SNnum_phot01_50_14_14_0.dat'

#### Load the estimated parameters from MCMC
######### The files have two columns, one for the mean and one for the standard deviation of the parameter from the chains.
data_SG ='Est_Params_Flat/20150215_est_param_1G_50000_st_4_d_500_w_50_dM_10000_data_0.dat'#SGM_med_params_20_0.txt'                                 
data_GMM = 'Est_Params_Flat/20150317_est_param_GMM_500000_st_8_d_500_w_50_dM_10000_data_0.dat'#GMM_med_params_20_0.txt'#
datafile_GMM = os.path.join(datadir,data_GMM)
datafile_1G = os.path.join(datadir, data_SG)

#### Load the covariances from MCMC
cov_SG = 'Covariance/20150604_cov_SG_4_d_10000_data_50_dM_0.dat'
cov_GM = 'Covariance/20150604_cov_GM_8_d_10000_data_50_dM_0.dat'
data_cov_SG = os.path.join(datadir,cov_SG)
data_cov_GM = os.path.join(datadir,cov_GM)


## Read in SGM parameters
OM_fit_SG = np.loadtxt(datafile_1G)[0][0]
w_fit_SG =  np.loadtxt(datafile_1G)[1][0]
M_fit_SG =np.loadtxt(datafile_1G)[3][0]-(25.0+ 5.0*np.log10(1.0/69.3))  ### Adjusted by (25-5 log H_0) to convert between script M and absolute magnitude
sig_fit_SG = np.loadtxt(datafile_1G)[2][0]

## Read in GMM parameters
OM_fit_GMM = np.loadtxt(datafile_GMM)[0][0]
w_fit_GMM =  np.loadtxt(datafile_GMM)[1][0]
M_fit_GMM  = np.loadtxt(datafile_GMM)[4][0]-(25.0+ 5.0*np.log10(1.0/69.3))  ### Adjusted by (25-5 log H_0) to convert between script M and absolute magnitude
sigA_fit_GMM = np.loadtxt(datafile_GMM)[2][0]
sigB_fit_GMM = np.loadtxt(datafile_GMM)[3][0]
dm_fit = np.loadtxt(datafile_GMM)[5][0]
c_fit= np.loadtxt(datafile_GMM)[6][0]
d_fit = np.loadtxt(datafile_GMM)[7][0]

## Read in variances from covariances. Could do this from the "datafile_1G" and "datafile_GMM" but keeping like this in case
##### we decide we need covariances
cov_SG = np.loadtxt(data_cov_SG)
cov_GM = np.loadtxt(data_cov_GM)
VAR_OM_SG = cov_SG[0,0]
VAR_w_SG = cov_SG[1,1]
VAR_sig_SG = cov_SG[2,2]
VAR_M_SG = cov_SG[3,3]
VAR_OM_GM = cov_GM[0,0]
VAR_w_GM = cov_GM[1,1]
VAR_sigA_GM = cov_GM[2,2]
VAR_sigB_GM = cov_GM[3,3]
VAR_MA_GM = cov_GM[4,4]
VAR_DM_GM = cov_GM[5,5]
VAR_f_GM = cov_GM[6,6]
VAR_d_GM = cov_GM[7,7]


## Read in mock data set
raw_datafile = os.path.join(datadir,'Mock_Data/%s'%(mock)) ## go up to 10000
M, z1, m1 = np.loadtxt(raw_datafile, unpack = True, usecols=[0, 1, 2])

# Sorting with respect to redshift (highest ot lowest)
combined = np.column_stack((M, z1, m1))
combined_sort = combined[np.argsort(combined[:, 1])]
M, z, m = combined_sort[:,0], combined_sort[:,1], combined_sort[:,2]

## Binning mock data set and keeping track on the means per bin
bins = np.linspace(0.05, 1.5, 41)
digitized = np.digitize(z, bins)
z_bins = [z[digitized == i].mean() for i in range(1, len(bins))]
m_means = np.array([m[digitized == i].mean() for i in range(1, len(bins))])
M_means = np.array([M[digitized == i].mean() for i in range(1, len(bins))])
N_bins = [np.sum(digitized == i)for i in range(1, len(bins))]


## Create LCDM model for comparison
z_model = np.linspace(0.005,1.6,1000)
mu_model = WMAP9.distmod(z_model).value


## Define distance modulus for SGM and GMM using fit values.
## Adjusted by (25-5 log H0 + 5 log c) to account for constants not included in the definition of the function
c1 = 2.9979E5 #km / s
mu_model_SG = distmod(OM_fit_SG, w_fit_SG, z_model)+(25.0 - 5.0*np.log10(69.3)+ 5.0*np.log10(c1))
mu_model_GM = distmod(OM_fit_GMM, w_fit_GMM, z_model)+(25.0 - 5.0*np.log10(69.3)+ 5.0*np.log10(c1))

# Define Absolute magnitude to be used in mu = m-M equation
M_intrinsic_SGM = M_fit_SG# mag
M_intrinsic_GMM = M_fit_GMM - (1.-(c_fit*np.array(z_bins)+d_fit))*dm_fit

## Use the above definition of absolute magnitude and the apparent magnitude from the mean of each bin to create
#### the distance modulus derived from supernova data points
distmod_points_SGM = m_means-M_intrinsic_SGM
distmod_points_GMM = m_means-M_intrinsic_GMM

## Distance modulus for given redshift bins to compare to data
distmod_table = WMAP9.distmod(z_bins).value

## Find the residual with respect to cosmological parameters for SGM and GMM versus LCDM
residual_SGM = mu_model_SG-mu_model
residual_GMM = mu_model_GM-mu_model

## Find the residual with respect to supernova data points and population parameters for SGM and GMM versus LCDM
residual_pts_SGM = distmod_points_SGM - distmod_table
residual_pts_GMM = distmod_points_GMM - distmod_table


### Define the standard deviation for "data points"
var_GMM = (c_fit*np.array(z_bins) + d_fit)*sigA_fit_GMM**2 +(1.-c_fit*np.array(z_bins)- d_fit)*sigB_fit_GMM**2+\
         (c_fit*np.array(z_bins) + d_fit)*M_fit_GMM**2 +(1.-c_fit*np.array(z_bins)- d_fit)*(M_fit_GMM-dm_fit)**2 \
       - ((c_fit*np.array(z_bins) + d_fit)*M_fit_GMM +(1.-c_fit*np.array(z_bins)- d_fit)*(M_fit_GMM-dm_fit))**2
stdev_GM = np.sqrt((var_GMM+0.1**2)/999.)
stdev_SG = np.sqrt((sig_fit_SG**2+0.1**2)/999.)



############################################################
############################################################
# Begin creating figure

#Define plot size and dimensions
fig=plt.figure(figsize=(6, 6))#(12.5,12.5))
#fig.suptitle('Hubble Diagram with Two-Population Shift = %+6.2f mag' % (dM) ,  fontsize=25)

gs=gridspec.GridSpec(2, 1, height_ratios=[3,1])
gs.update(hspace=0.001)
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1], sharex=ax1)


xmin = 0.0
xmax = 1.6


## Plot distance modulus from cosmological models
ax1.plot(z_model,  mu_model ,'k', linewidth = 3, label = "$\Lambda CDM,\Omega_M = %0.2f, w = %0.2f$"%(WMAP9.Om0, -1.0))
ax1.plot(z_model,  mu_model_SG ,'b', linewidth = 3, label = 'SGM, $\Omega_M = %0.2f, w = %0.2f$'%(OM_fit_SG, w_fit_SG)) 
ax1.plot(z_model,  mu_model_GM ,'r--', linewidth = 3, label = 'GMM,$\Omega_M = %0.2f, w = %0.2f$'%(OM_fit_GMM,w_fit_GMM)) 

## Plot distance modulus from mock data points and population parameters
ax1.errorbar(z_bins, distmod_points_SGM, yerr=stdev_SG, fmt='bo')#, alpha = 0.3)
ax1.errorbar(z_bins, distmod_points_GMM,yerr=stdev_GM, fmt= 'ro')#, alpha = 0.3)

# set plot limits
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(36.5, 46.)
ax1.set_yticks(np.arange(37, 46., 2.0))
#ax1.annotate('$\Delta M = 0.5$ mag', xy=(0.05, 44.75), fontsize=30)

# remove labels or adjusting size
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(),fontsize = 15)#5)

# Add legend
ax1.legend(numpoints=1, loc=4, fontsize = 15)#4)


fig.add_subplot(ax1)

ax2.set_xlabel("Redshift",fontsize=20)

# Add residual line at zero to represent LCDM model
ax2.axhline(0, color = 'k', lw = 2)

# Plot residual from cosmological models
ax2.plot(z_model,residual_SGM, 'b', lw = 3)
ax2.plot(z_model,residual_GMM, 'r--', lw = 3)

## Plot residuals from mock data points and population parameters
ax2.errorbar(z_bins,residual_pts_SGM,yerr=stdev_SG, fmt='bo', lw = 1)#, alpha = 0.5)
ax2.errorbar(z_bins,residual_pts_GMM,yerr = stdev_GM,fmt= 'ro', lw = 1)#, alpha = 0.5)

ax2.set_xlim((xmin, xmax))


## Increasing tick labels, removing unwanted ticks, setting tick range
plt.setp(ax2.get_xticklabels(),fontsize = 15)#5)
plt.setp(ax2.get_yticklabels(),fontsize = 13.5)#20)#5)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
ax2.set_yticks(np.arange(-1, 0.5, 0.25))

ax2.set_ylim((-0.5, 0.5))
fig.add_subplot(ax2)

### These texts will show up too far to the left on the pop-up window, but if you save the plot as a pdf, they are in the right spot
## used this instead of "ylabel" to make sure they did not interfere with y-axis tick labels
plt.figtext(0.0,0.3,'$\mu - \mu_{\Lambda CDM}$',fontdict={'fontsize':20},rotation=90)
plt.figtext(0.0,0.59,'$\mu$',fontdict={'fontsize':20},rotation=90)

#plt.savefig('../Figures/%s_hubble_50.pdf'%(strftime('%Y%m%d')))
plt.show()
