# from sbi4abm.sbi.utils.torchutils import IntegerUniform, CompositeUniform, BoxUniform
# from sbi4abm.sbi.utils.sbiutils import within_support
# import torch

# # Example usage:
# low = torch.tensor([0.0, 5])
# high = torch.tensor([1.0, 15])
# distribution = CompositeUniform(low, high)
# # distribution = BoxUniform(low=box_low, high=box_high)
# print(distribution)

# # Sampling from the distribution
# samples = distribution.sample((20,))
# print(samples)

# # Getting the mean of the distribution
# # mean = distribution.mean
# # print(mean)

# # Calculating log_prob of a sample
# log_prob = distribution.log_prob(samples)
# print("cona")
# print(log_prob)

# # Testing the support functionality
# test_samples = torch.tensor([
#     [0.5, 5],
#     [1.5, 5],  # Out of support for BoxUniform
#     [0.5, 2],  # Out of support for BoxUniform
#     [0.5, 15], # Out of support for IntegerUniform
#     [0.0, 12], # Within support
#     [0.3, 17]
# ])

# support_check = distribution.support.check(test_samples)
# print("Support check for test samples:\n", support_check)

# support_check = within_support(distribution, test_samples)
# print("Support check for test samples using within_support:\n", support_check)



from sbi4abm.sbi.utils.torchutils import CompositeUniform
from sbi4abm.sbi.utils.sbiutils import within_support
import torch

# Example usage:
low = torch.tensor([0.1, 0.0002, 0.0002, 10, 10])
high = torch.tensor([0.8, 0.0016, 0.0016, 25, 25])
distribution = CompositeUniform(low, high)
print(distribution)

# Sampling from the distribution
samples = distribution.sample((20,))
print("Samples:")
print(samples)

# Getting the mean of the distribution
# mean = distribution.mean
# print("Mean:")
# print(mean)

# Calculating log_prob of a sample
log_prob = distribution.log_prob(samples)
print("Log probabilities:")
print(log_prob)

# Testing the support functionality
test_samples = torch.tensor([
    [0.5, 0.001, 0.001, 15, 15],    # Within support
    [0.9, 0.001, 0.001, 15, 15],    # Out of support for BoxUniform
    [0.5, 0.001, 0.001, 15, 5],     # Out of support for IntegerUniform
    [0.5, 0.001, 0.001, 30, 15],    # Out of support for BoxUniform and IntegerUniform
    [0.2, 0.0005, 0.0005, 20, 20],  # Within support
    [0.3, 0.002, 0.002, 20, 30]     # Out of support for BoxUniform and IntegerUniform
])

support_check = distribution.support.check(test_samples)
print("Support check for test samples:")
print(support_check)

support_check = within_support(distribution, test_samples)
print("Support check for test samples using within_support:")
print(support_check)


# from sbi4abm.sbi.utils.torchutils import BoxUniform
# from sbi4abm.sbi.utils.sbiutils import within_support
# import torch

# # Example usage:
# low = torch.tensor([0.0, 0.0])
# high = torch.tensor([1.0, 1.0])
# distribution = BoxUniform(low=low, high=high)
# print(distribution)

# # Sampling from the distribution
# samples = distribution.sample((20,))
# print("Samples:")
# print(samples)

# # Getting the mean of the distribution
# mean = distribution.mean
# print("Mean:")
# print(mean)

# # Calculating log_prob of a sample
# log_prob = distribution.log_prob(samples)
# print("Log probabilities:")
# print(log_prob)

# # Testing the support functionality
# test_samples = torch.tensor([
#     [0.5, 0.5],
#     [1.5, 0.5],  # Out of support for BoxUniform
#     [0.5, -0.2], # Out of support for BoxUniform
#     [0.0, 1.0],  # On the edge of support
#     [0.3, 0.7],  # Within support
#     [1.0, 1.0]   # On the edge of support (exclusive upper bound)
# ])

# support_check = distribution.support.check(test_samples)
# print("Support check for test samples:")
# print(support_check)

# support_check_within = within_support(distribution, test_samples)
# print("Support check for test samples using within_support:")
# print(support_check_within)