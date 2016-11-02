## Create butterfly plots
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.cosmology import WMAP9
from time import strftime
from matplotlib.ticker import MaxNLocator
import Data_Set_Generator as DSG  ## for up to date relative normalization


def one_G(x, p):
    mu, sig = p
    p_A = np.sqrt(2*np.pi*sig**2)**(-1.0)*np.exp(-(x-mu)**2/(2*sig**2)) ##sig**2=variance
    return p_A

def GMM(x, p):
    mu_A, sig_A, mu_B, sig_B, w_A = p
    p_A = np.sqrt(2*np.pi*sig_A**2)**(-1.0)*np.exp(-(x-mu_A)**2/(2*sig_A**2))
    p_B = np.sqrt(2*np.pi*sig_B**2)**(-1.0)*np.exp(-(x-mu_B)**2/(2*sig_B**2))
    return (w_A*p_A + (1-w_A) * p_B)

def data_GMM(mean_A, dM, n_A):
    return n_A*mean_A + (1.0-n_A)*(mean_A -dM) #(2.0*mean_A - dM)/2.0

def std_err( p, Nbins, GMM = False):
    if GMM:
        sig_A, sig_B, n_A = p
        return (n_A*sig_A**2/Nbins +(1.0-n_A)*sig_B**2/Nbins)**(0.5)
    else:
        return p/np.sqrt(Nbins)

def residual( M_data, M_model):
    return M_data - M_model

# Create GMM model
def apparent_mag(z, p , GMM = False):
    if GMM:
        n_A, peak_A, peak_B = p
        app_mag = WMAP9.distmod(z).value + n_A*peak_A + (1.0-n_A)*peak_B
    else:
        app_mag = WMAP9.distmod(z).value + p
    return app_mag

#def distmod_residual():
#    return

def bin_dat(data, bin_num, xmin, xmax):
    bins = np.linspace(xmin, xmax, bin_num)
    digitized = np.digitize(data, bins)
    bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
    return bin_means, digitized, bins

def bin_means(data, digitized, Nbins, bins):
    digitized = np.repeat(digitized, Nbins)
    bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
    return bin_means


def residual_plot( One_G_data, sigma_1G, GMM_data, p_GMM, z, dM, Nbins, App_Mag = False, One_G = True, GMM = True, Model = True , save_fig = False, **kwargs):
    """
    Residual plot testing
    """

    figure_size = kwargs.pop('fig_size', (12.5,12.5))
    title_size = kwargs.pop('suptitle_fontsize', 25)
    y_label = kwargs.pop('y_label', 'Distance Modulus')
    x_size = kwargs.pop('x_fontsize', 30)
    y_size = kwargs.pop('y_fontsize', 30)
    extra = kwargs.pop('save_fig_label', 'Normal')
    line = kwargs.pop('linewidth', 3)
    model_line = kwargs.pop('model_color', 'k-')
    One_G_fmt = kwargs.pop('One_G_fmt', 'bo')
    GMM_fmt = kwargs.pop('GMM_fmt', 'ro')
    One_G_label = kwargs.pop('One_G_label', '1G')
    GMM_label = kwargs.pop('GMM_label', 'GMM')
    M_ave = kwargs.pop('M_ave', -19.5)

    y_lim = kwargs.pop('y_lim', (38, 48))
    xmin = kwargs.pop('x_min', 0.0)
    xmax = kwargs.pop('x_max', 2.0)

    z_model = np.linspace(0.001 , 2.5, 1000)


    fig= plt.figure(figsize=figure_size)
    fig.suptitle('Hubble Diagram with Two-Population Shift = %+6.2f mag' % (dM) ,  fontsize=title_size)

    fig.subplots_adjust(hspace=0.001)
    gs=gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax1=plt.subplot(gs[0])
    ax2=plt.subplot(gs[1], sharex=ax1)

    if Model:
        mu_model = WMAP9.distmod(z_model).value
        if not App_Mag:
            ax1.plot(z_model, mu_model, model_line, linewidth = line)
        if App_Mag:
            ax1.plot(z_model, mu_model + M_ave, model_line, linewidth = line)

    if not App_Mag:
        Model_data = WMAP9.distmod(z).value
    if App_Mag:
        Model_data = WMAP9.distmod(z).value + M_ave
            

    if One_G:
        yerr_1G = std_err(sigma_1G, Nbins, GMM = False)
        ax1.errorbar(z, One_G_data, yerr=yerr_1G, fmt=One_G_fmt, label=One_G_label,  lw = line)
        ax2.errorbar(z, residual(One_G_data, Model_data),yerr=yerr_1G, fmt= One_G_fmt, lw = line)

    if GMM:
        yerr_GMM = std_err(p_GMM, Nbins, GMM = True)
        ax1.errorbar(z, GMM_data, yerr=yerr_GMM, fmt=GMM_fmt, label=GMM_label,  lw = line)
        ax2.errorbar(z,residual(GMM_data, Model_data), yerr=yerr_GMM, fmt=GMM_fmt,  lw = line)

    ax1.set_ylim(y_lim)
    plt.setp(ax1.get_xticklabels(), visible=False)

    if App_Mag:
         ax1.set_ylabel('Apparent Magnitude', fontsize=y_size)
    else:
        ax1.set_ylabel('%s'%(y_label), fontsize=y_size)

    ax2.plot(np.linspace(xmin, xmax, len(z_model)), np.zeros(len(z_model)),  model_line, lw=line)
    ax2.set_xlim(xmin, xmax)
    plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))

    #fig.add_subplot(ax1)
    #fig.add_subplot(ax2)

    ax2.set_xlabel('Redshift', fontsize = x_size)

    if save_fig:
        plt.savefig('/Users/karaponder/Spring14/Figures/%s_Hubble_Diagram_wResdiual_%s.pdf'%(strftime('%Y%m%d'), extra))
    #plt.show()
    return #fig



#def hubble_diagram( dM ,residual = False, save_fig = False, **kwargs):
#    if residual:
#    return



def butterfly(  One_G_data, sigma_1G, GMM_data, p_GMM, z , Nbins, hist_Nbins = 11, save_fig = False, residual_label = False, legend = True, **kwargs):

    figure_size = kwargs.pop('fig_size', (12.5,12.5))
    x_size = kwargs.pop('x_fontsize', 30)
    One_G_fmt = kwargs.pop('One_G_fmt', 'bo')
    GMM_fmt = kwargs.pop('GMM_fmt', 'ro')
    One_G_label = kwargs.pop('One_G_label', '1G')
    GMM_label = kwargs.pop('GMM_label', 'GMM')
    marker = kwargs.pop('markersize', 9)
    M_ave = kwargs.pop('M_ave', -19.5)
    x_size = kwargs.pop('x_fontsize', 30)
    legend_size = kwargs.pop('legend_fontsize', 30)
    model_line = kwargs.pop('model_color', 'k-')
    line = kwargs.pop('linewidth', 3)

    extra = kwargs.pop('save_fig_label', 'Normal')
    
    xmin = kwargs.pop('x_min', 0.0)
    xmax = kwargs.pop('x_max', 2.0)
    y_lim = kwargs.pop('y_lim', (18, 28))
    

    fig=plt.figure(figsize=figure_size) 
    fig.subplots_adjust(hspace=0.001)
    gs=gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax1=plt.subplot(gs[0])
    ax2=plt.subplot(gs[1], sharex=ax1)

    GMM_A_peak, dM = GMM_data
    dM = np.mean(dM, dtype=np.float64)
    plt.suptitle('Hubble Diagram with Two-Population Shift = %+6.2f mag' % (dM) ,  fontsize=30)

    z_model = np.linspace(0.001 , 2.5, 1000)
    Model_data = WMAP9.distmod(z).value + M_ave
    mu_model = WMAP9.distmod(z_model).value
    ax1.plot(z_model, mu_model+M_ave, 'k', linewidth = 3)

    # Set x values for subplots
    zero=np.zeros(100)
    m = np.linspace(-1.5, +1.5, 100) + M_ave
    r = np.linspace(-1.5, +1.5, 100)

    # Define my parameters for the PDFs. ONLY WANT A FEW FOR THE DATA.
    binned_z, digitized, bins = bin_dat( z, hist_Nbins, xmin, xmax)
    
    binned_One_G_data = bin_means( One_G_data, digitized, Nbins, bins)
    if len(One_G_data) == len(sigma_1G):
        binned_sigma_1G = bin_means( sigma_1G, digitized, Nbins, bins)
        p_1G = np.column_stack((binned_One_G_data, binned_sigma_1G))
    else:
        mult_sigma_1G = [sigma_1G]*len(One_G_data)
        binned_sigma_1G = bin_means( mult_sigma_1G, digitized, Nbins, bins)
        p_1G = np.column_stack((binned_One_G_data, binned_sigma_1G))

    binned_GMM_A_peak = bin_means( GMM_A_peak, digitized, Nbins, bins)
    binned_GMM_B_peak = bin_means( GMM_A_peak - dM,digitized, Nbins, bins)
    sigma_A_GMM, sigma_B_GMM, n_A = p_GMM
    if len(GMM_A_peak) == len(sigma_A_GMM):
        binned_sigma_A = bin_means( sigma_A_GMM, digitized, Nbins, bins)
        binned_sigma_B = bin_means( sigma_B_GMM,digitized, Nbins, bins)
        binned_n_A = bin_means( n_A, digitized, Nbins, bins)
        p_2G = np.column_stack((binned_GMM_A_peak, binned_sigma_A , binned_GMM_B_peak, binned_sigma_B , binned_n_A)) # mu_A, sig_A, mu_B, sig_B = p
    else:
        sigma_A_GMM, sigma_B_GMM, n_A = [sigma_A_GMM]*len(GMM_A_peak) , [sigma_B_GMM]*len(GMM_A_peak), [n_A]*len(GMM_A_peak)
        binned_sigma_A = bin_means( sigma_A_GMM,digitized, Nbins, bins)
        binned_sigma_B = bin_means( sigma_B_GMM, digitized, Nbins, bins)
        binned_n_A = bin_means( n_A, digitized, Nbins, bins)
        p_2G = np.column_stack((binned_GMM_A_peak, binned_sigma_A , binned_GMM_B_peak, binned_sigma_B , binned_n_A))

    dz = np.mean(binned_z[1:] - binned_z[0:-1])
    # MWV: I just chose a factor of 3 because it looked good with the given PDFs.  You could have much more sharply peaked PDFs that would extend too far.
    pdf_scale = dz / 3  
    ## Plot the PDFs on top of my data points
    distmod_table = WMAP9.distmod(binned_z).value
    for i in range ( 0 , len(binned_z)):
        ax1.plot(-one_G(m, p_1G[i])*pdf_scale +binned_z[i], m+distmod_table[i], 'b', lw=3) #what to plot
        ax1.plot( GMM(m, p_2G[i])  *pdf_scale +binned_z[i], m+distmod_table[i], 'r', lw=3)   #what to plot
        ax2.plot(-one_G(m, p_1G[i])*pdf_scale +binned_z[i], r, 'b', lw=3) #what to plot      r+residual_mag_10z[i]
        ax2.plot( GMM(m, p_2G[i])  *pdf_scale +binned_z[i], r, 'r', lw=3)   #what to plot

    if legend:
        ax1.legend(numpoints=1, loc = 4, fontsize=legend_size)
    ax1.set_ylim(y_lim)
    ax1.set_xlim(xmin, xmax)

    ax2.plot(np.linspace(xmin, xmax, len(z_model)), np.zeros(len(z_model)),  model_line, lw=line)

    GMM_data_1 = data_GMM(GMM_A_peak, dM, n_A)
    
    yerr_1G = std_err(sigma_1G, Nbins, GMM = False)
    ax1.errorbar(z, One_G_data + distmod_table, yerr=yerr_1G, fmt=One_G_fmt, label=One_G_label,  lw = line, markersize=marker)
    ax2.errorbar(z, residual(One_G_data + distmod_table, Model_data),yerr=yerr_1G, fmt= One_G_fmt, lw = line, markersize=marker)

    yerr_GMM = std_err(p_GMM, Nbins, GMM = True)
    ax1.errorbar(z, GMM_data_1 + distmod_table, yerr=yerr_GMM, fmt=GMM_fmt, label=GMM_label,  lw = line, markersize=marker)
    ax2.errorbar(z,residual(GMM_data_1 + distmod_table, Model_data), yerr=yerr_GMM, fmt=GMM_fmt,  lw = line,  markersize=marker)

    if residual_label:
        ax2.set_ylabel(r'$m - m_{\Lambda CDM}$',fontsize = y_size)

    ax2.set_ylim((-2.0, 2.0))
    plt.setp(ax1.get_xticklabels(), visible=False)

    yticks = ax2.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    ax1.tick_params(axis='y', labelsize=30)
    ax2.tick_params(axis='x', labelsize=30)
    ax2.tick_params(axis='y', labelsize=30)

    ax1.set_ylabel('Apparent Magnitude [mag]', fontsize=x_size)
    ax2.set_xlabel('Redshift', fontsize = x_size)

    ax2.set_yticks([ -1.5, 0.0, 1.5])
    
    if save_fig:
        plt.savefig('/Users/karaponder/Spring14/Figures/%s_Butterfly_wResdiual_%s.pdf'%(strftime('%Y%m%d'), extra))

    plt.show()
    print m
    print p_2G[0]
    p_test = 15.66462843, 0.1,14.66462843,0.1, 0.5  
    plt.plot(m, GMM(m,p_test))
    plt.show()
    return


