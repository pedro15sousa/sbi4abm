import sbi4abm.sbi
from sbi4abm.sbi import analysis, utils
from sbi4abm.utils import sampling, plotting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

matplotlib.rc('text', usetex=True)
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})
plt.rcParams.update({
    'text.latex.preamble':r"\usepackage{amsmath}"+"\n"+r"\usepackage{bm}"
})

# hop = np.loadtxt("../../exp_dir/1708014932.055341/samples.txt")
# hop = np.loadtxt("exp_dir/1708021338.2990642/samples.txt")
hop = np.loadtxt("exp_dir/1708076421.561745/samples.txt")
# plt.rcParams.update({'font.size':18}) # something about latex font
# plot = analysis.pairplot(hop, limits=[[-1,1], [-1,1], [-1,1], [-1,1]], points=[np.array([0.25, 0.15, 0.45, 0.5])],
#                       points_colors='r', labels=[r"$\rho$", r"$\epsilon$", r"$\lambda$", r"$\alpha$"],
#                       hist_diag={"alpha": 1.0, "bins": 25, "density": False, "histtype": "step"})

# plt.show()

file_name = "1708076421.561745"
with open(f"exp_dir/{file_name}/posteriors.pkl", 'rb') as file:
    posterior = pickle.load(file)
    posterior = posterior[0]

print(posterior)

x = torch.tensor([10.225096881558523,
                14.164716771097027,
                7.1184287049926915,
                1.4364261138679228,
                0.14048044047959973])

# Assume `posterior` is your DirectPosterior object and `x` is your data
samples = posterior.sample((1000,), x)  # Generate 1000 samples from the posterior
log_probs = posterior.log_prob(samples, x)  # Compute the log probability of the samples

print(samples.shape)
print(log_probs.shape)

# Convert the tensor to a numpy array
samples_np = samples.numpy()
print(samples_np)

# Plot the first dimension of the samples
plt.hist(samples_np[:, 0], bins=30, density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Posterior Samples')
plt.show()