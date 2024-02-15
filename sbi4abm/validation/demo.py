import sbi4abm.sbi
from sbi4abm.sbi import analysis, utils
from sbi4abm.utils import sampling, plotting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('text', usetex=True)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})
plt.rcParams.update({
    'text.latex.preamble':r"\usepackage{amsmath}"+"\n"+r"\usepackage{bm}"
})

# hop = np.loadtxt("../../exp_dir/1708014932.055341/samples.txt")
hop = np.loadtxt("exp_dir/1708016836.266269/samples.txt")

plt.rcParams.update({'font.size':18}) # something about latex font
plot = analysis.pairplot(hop, limits=[[-1,1], [-1,1], [-1,1], [-1,1]], points=[np.array([0.25, 0.15, 0.45, 0.5])],
                      points_colors='r', labels=[r"$\rho$", r"$\epsilon$", r"$\lambda$", r"$\alpha$"],
                      hist_diag={"alpha": 1.0, "bins": 25, "density": False, "histtype": "step"})

plt.show()