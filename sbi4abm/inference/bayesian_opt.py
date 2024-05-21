import numpy as np
import random

from sbi4abm.sbi.simulators.simutils import simulate_in_batches

from emukit.core import ContinuousParameter, ParameterSpace, DiscreteParameter
from emukit.core.initial_designs import RandomDesign
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.loop import UserFunctionResult
from emukit.core.loop import OuterLoop
from emukit.core.optimization import GradientAcquisitionOptimizer, LocalSearchAcquisitionOptimizer, MultiSourceAcquisitionOptimizer
from emukit.core.optimization.optimizer import Optimizer
from emukit.core.optimization.random_search_acquisition_optimizer import RandomSearchAcquisitionOptimizer

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from skopt.space import Real
from sklearn.metrics import mean_squared_error

class BayesianOpt:
    def __init__(self, simulator, target):
        self.simulator = simulator
        self.target = target
        self.parameter_space = None

    def f(self, thetas):
        mse_arr = []
        for theta in thetas:
            x = self.simulator(theta)
            mse_arr.append(mean_squared_error(self.target, x))
        return np.array(mse_arr).reshape(-1, 1)


    def set_parameter_space(self, prior):
        parameters_list = []
        for i in range(prior.low.shape[0]):
            low = prior.low[i].item()  # Convert torch.Tensor to Python scalar
            high = prior.high[i].item()
            parameter = ContinuousParameter(f"theta_{i}", low, high)
            parameters_list.append(parameter)
        
        # create continuous parameter space
        parameter_space = ParameterSpace(parameters_list)
        self.parameter_space = parameter_space


    def collect_random_points(self):
        design = RandomDesign(self.parameter_space) # Collect random points
        num_data_points = 20
        X = design.get_samples(num_data_points)
        Y = self.f(X)
        return X, Y


    def run_bo_loop(self, X, Y, max_iterations=10):
        model_gpy = GPRegression(X,Y) # Train and wrap the model in Emukit
        model_emukit = GPyModelWrapper(model_gpy)
        expected_improvement = ExpectedImprovement(model = model_emukit)

        acquisition_optimizerers = [GradientAcquisitionOptimizer(self.parameter_space),
                                    LocalSearchAcquisitionOptimizer(self.parameter_space),
                                    RandomSearchAcquisitionOptimizer(self.parameter_space)]

        bayesopt_loop = BayesianOptimizationLoop(model = model_emukit,
                                                space = self.parameter_space,
                                                acquisition = expected_improvement,
                                                batch_size = 1,
                                                acquisition_optimizer=acquisition_optimizerers[0])
        
        bayesopt_loop.run_loop(self.f, max_iterations)
        results = bayesopt_loop.get_results()
        print("\n")
        print("Target: ", self.target)
        print("MSE Loss: ", self.f([results.minimum_location]))
        print("RESULTS: ", results.minimum_location)
        return results