import torch
import numpy as np

def flock_summariser(x):
    print("X shape to compute stats: ", x.shape)
    if x.ndim == 2:
        print("True data (final state).")
        final_state = x
    else:
        final_state = x[-1]
        
    center = np.mean(final_state, axis=0)
    final_cohesion = np.mean([np.linalg.norm(agent - center) for agent in final_state])

    distances = [np.linalg.norm(agent - other) for agent in final_state for other in final_state if not np.array_equal(agent, other)]
    final_separation_std = np.std(distances)
    final_separation_avg = np.mean(distances)

    cohesion_separation_ratio = final_cohesion / final_separation_std

    flock_density = 1 / final_separation_std if final_separation_std != 0 else float('inf')

    sx = np.array([
        cohesion_separation_ratio,
        flock_density
    ])

    return sx


def virus_summariser(x):
    print("X shape to compute stats: ", x.shape)
    # if x.ndim == 2:
    #     print("True data (final state).")
    #     final_state = x
    # else:
    #     final_state = x[-1]

    # investigate why the dimension might be different for the first run
    
    final_state = x[-1]

    final_infected = final_state[1]
    final_recovered = final_state[2]

    sx = np.array([
        final_infected,
        final_recovered
    ])
    return sx


def fire_summariser(x):
    print("X shape to compute stats: ", x.shape)
    if x.ndim == 2:
        x = x[0]

    # In the forest fire ABM, we do not register any time-series data for the agents. We only
    # record the reporter for final percentage of burned trees and the seed. Therefore, we only
    # need to return x[-1] where x is the list of reporters and the percentage is the last element.
    burned_percentage = x[-1]
    print("Burned percentage: ", burned_percentage) 
    sx = np.array([
        burned_percentage
    ])
    return sx


def segregation_summariser(x):
    print("X shape to compute stats: ", x.shape)
    if x.ndim == 2:
        x = x[0]

    # In the segregation ABM, we do not register any time-series data for the agents. We only
    # record the reporter for final percentage of similar neighbors and the seed. Therefore, we only
    # need to return x[-1] where x is the list of reporters and the percentage is the last element.
    segregation_percentage = x[-1]
    sx = np.array([
        segregation_percentage
    ])
    return sx


def acf(x, lag):
	"""
	Computes autocorrelation at specified lag for 1D time series
	"""
	return np.dot(x[:-lag], x[lag:]) / (x.shape[0] - 1)



def hopfield_summariser(x):
    print("X shape to compute stats: ", x.shape)
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    iu1 = np.triu_indices(50)
    final = x[-1]
    w, s = final[:, :-2], final[:, -2:]
    A = np.sign(w)
    n_neg_triad = np.sum(np.diag(A.dot(A).dot(A)) + 1) / 2
    A = (w >= 0).astype(int)
    k = np.sum(A, axis=1)
    glob_clust_coeff = np.trace((A.dot(A)).dot(A)) / np.sum(k*(k-1))
    overlaps = np.dot(s, s.T)[iu1]/2
    weights = w[iu1]
    corr_weight_op = np.corrcoef(weights, overlaps)

    sx = np.concatenate([
        n_neg_triad.flatten(), 
        glob_clust_coeff.flatten(), 
        corr_weight_op.flatten()
    ])

    return sx



# def hopfield_summariser(x):
#     print("X shape to compute stats: ", x.shape)
    
#     if isinstance(x, torch.Tensor):
#         x = x.detach().cpu().numpy()

#     final_state = x[-1]

#     # summing all states
#     sum_states = np.sum(final_state, axis=0)

#     # computing the magnitude of the sum vector
#     magnitude = np.linalg.norm(sum_states)

#     # normalising by the number of agents to get the coherence measure
#     coherence = magnitude / final_state.shape[0]

#     # sx = np.array([
#     #     coherence
#     # ])

#     # Calculating the mean of opinions across each topic
#     mean_opinions = np.mean(final_state, axis=0)  # Mean over agents for each topic
    
#     # Calculating the variance in opinions across each topic to capture diversity
#     variance_opinions = np.var(final_state, axis=0)  # Variance over agents for each topic
    
#     # Compile the statistics into a single array
#     # Note: To append 'mean_opinions' and 'variance_opinions' to 'sx', we flatten them 
#     # because 'sx' is expected to be a 1D array. Adjust accordingly if different format is needed.
#     # print(mean_opinions.shape)
#     sx = np.concatenate([
#         np.array([coherence]), 
#         mean_opinions.flatten(), 
#         variance_opinions.flatten()
#     ])

#     return sx


# def hopfield_summariser(x):
#     print("X shape to compute stats: ", x.shape)

#     if isinstance(x, torch.Tensor):
#         x = x.detach().cpu().numpy()

#     # final_s = x

#     final_state = x[-1]
#     final_s = final_state[:, -2:]

#     # Compute mean opinions per topic
#     mean_opinions = np.mean(final_s, axis=0)

#     # Compute variance in opinions per topic
#     variance_opinions = np.var(final_s, axis=0)

#     # Compute overall coherence
#     # Assuming binary opinions for simplicity (-1, 1)
#     coherence_per_topic = np.abs(np.sum(final_s, axis=0)) / final_s.shape[0]
#     overall_coherence = np.linalg.norm(np.sum(final_s, axis=0)) / final_s.shape[0]

#     sx = np.concatenate([
#         np.array([overall_coherence]),  # Make scalar a 1D array
#         mean_opinions.flatten(),  # Already 1D, flatten() is optional
#         variance_opinions.flatten()  # Already 1D, flatten() is optional
#     ])

#     print("Statistics: ", sx)
#     return sx


def brockhommes_summariser(x):
    print("X shape to compute stats: ", x.shape)

    if x.ndim == 1:
        return np.array([
            1.81295542e-05, 
            6.47890787e-01,
            2.00598284e-02,
            3.32031255e-01
        ])

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    final_state = x[-1]

    sx = np.array([
        final_state[0],
        final_state[1],
        final_state[2],
        final_state[3],
    ])
    return sx


def frankewesterhoff_summariser(x):
    print("X to compute stats: ", x)

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # if x[0].ndim == 1:
    #     return np.array([
    #         -1.25042709e-02,
    #         -2.35443573e-04,
    #         -1.18352260e-04,
    #         0.00000000e+00,
    #         1.17387535e+00,
    #         2.29191274e+00
    #     ])

    final_price = x[0]
    final_wealth_fundamentalists = x[1]
    final_wealth_chartists = x[2]
    final_proportion_fundamentalists = x[3]
    final_demand_fundamentalists = x[4]
    final_demand_chartists = x[5]

    sx = np.array([
        final_price,
        final_wealth_fundamentalists,
        final_wealth_chartists,
        final_proportion_fundamentalists,
        final_demand_fundamentalists,
        final_demand_chartists
    ])
    return sx


def socialcare_summariser(x):
    final_tax = x[0]
    final_per_capita_cost = x[1]

    sx = np.array([
        final_tax,
        final_per_capita_cost
    ])
    return sx


# def frankewesterhoff_summariser(x):
#     print("X shape to compute stats: ", x.shape)

#     x = x.reshape(-1)
#     sx = np.array([
#         np.min(x),
#         np.quantile(x, 0.25),
#         np.median(x),
#         np.quantile(x, 0.75),
#         np.max(x),
#         np.mean(x),
#         np.var(x),
#         acf(x, 1),
#         acf(x, 2),
#         acf(x, 3)
#     ])
#     return sx



