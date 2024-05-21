import argparse
from argparse import RawTextHelpFormatter
import numpy as np

from sbi4abm.inference import kde, neural, bayesian_opt
from sbi4abm.utils import io
from sbi4abm.validation import sbc
from sbi4abm.utils.stats import virus_summariser, flock_summariser, \
								fire_summariser, hopfield_summariser, \
								brockhommes_summariser, segregation_summariser, \
								frankewesterhoff_summariser

class Summariser1D:

	def __init__(self, simulator):

		self.simulator = simulator

	def __call__(self, pars):

		"""
		Summarises 1D time series x with: quantiles 0, 25, 50, 75, 100; mean;
		variance; autocorrelations at lags 1, 2, 3
		"""

		x = self.simulator(pars)
		return self.summarise(x)

	def summarise(self, x):
		if args.task == "flocking":
			return flock_summariser(x)
		elif args.task == "virus":
			return virus_summariser(x)
		elif args.task == "fire":
			return fire_summariser(x)
		elif args.task == "hop":
			return hopfield_summariser(x)
		elif args.task == "bh_smooth":
			return brockhommes_summariser(x)
		elif args.task == "segregation":
			return segregation_summariser(x)
		elif args.task == "fw_hpm" or args.task == "fw_wp":
			return frankewesterhoff_summariser(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for single simulation",
									 formatter_class=RawTextHelpFormatter)
    parser.add_argument("--task", type=str, help="Task name")
    parser.add_argument("--params", nargs='+')
    args = parser.parse_args()

    simulator, _, _, _ = io.load_task(args.task)

    simulator = Summariser1D(simulator)
	
    x = simulator(args.params)
	
    print("Resulting statistics: ", x)
	
