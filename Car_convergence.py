import numpy as np
import Convergence_testing as CT
import sys

data_file = sys.argv[1]
samples = np.loadtxt(data_file)

labels = [ "$\Omega_M$", "$\Omega_L$", "$w$", "$\sigma_{int}$","$\mathcal{M}$"]

y = np.linspace(0, 99, 100)
CT.plotting(labels, y, samples, savefig = True)
