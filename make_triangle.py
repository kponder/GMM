import triangle
import numpy as np
import sys
import os
from time import strftime

data_file_name = sys.argv[1]
datadir='/global/u2/k/kap146/carver/GMM_Likelihood/Output_Data' #'/Users/karaponder/Spring14/Data' # 
datafile =os.path.join(datadir,data_file_name)

samples = np.loadtxt(datafile)
#print len(samples)
dim = int(sys.argv[2])
#print dim
smooth = float(sys.argv[3])
#print smooth
series = sys.argv[4]
#print series
if dim == 9:
    labels = [ "$\Omega_M$", "$\Omega_L$", "$w$","$\sigma_{int, A}$", "$\sigma_{int, B}$", "$\mathcal{M}$", "$\Delta M$", "c", "d"]
    kind = 'GMM'
elif dim == 6:
    labels = [ "M","$\Delta M$", "$\sigma_{int, A}$", "$\sigma_{int, B}$", "c", "d"]
    kind = 'abGMM'
elif dim == 5:
    labels = [ "$\Omega_M$", "$\Omega_L$", "$w$","$\sigma_{int}$", "$\Delta M$"]
    kind = 'SG'
elif dim == 2:
    labels = [ "M", "$\sigma_{int}$"]
    kind = 'abSG'

#print kind
#print labels
fig = triangle.corner(samples, labels = labels, truths=None, plot_datapoints = False, sigma = smooth, quantiles=[0.16, 0.5, 0.84], show_titles=True, save_quantiles = True, title_args={"fontsize": 20})

fig.savefig("%s_%s_%s_d_%.0f_smo_%s.pdf"%(strftime('%Y%m%d'), kind, dim, smooth*10, series))
