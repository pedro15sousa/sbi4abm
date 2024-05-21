from numba import njit
import numpy as np
import agentpy as ap
import networkx as nx
import random

class Person(ap.Agent):

    def setup(self):
        """ Initialize a new variable at agent creation. """
        self.condition = 0  # Susceptible = 0, Infected = 1, Recovered = 2

    def being_sick(self):
        """ Spread disease to peers in the network. """
        rng = self.model.random
        for n in self.network.neighbors(self):
            if n.condition == 0 and self.p.infection_chance > rng.random():
                n.condition = 1  # Infect susceptible peer
        if self.p.recovery_chance > rng.random():
            self.condition = 2  # Recover from infection


class VirusModel(ap.Model):

    def setup(self):
        """ Initialize the agents and network of the model. """

        # Prepare a small-world network
        graph = nx.watts_strogatz_graph(
            self.p.population,
            self.p.number_of_neighbors,
            self.p.network_randomness)

        # Create agents and network
        self.agents = ap.AgentList(self, self.p.population, Person)
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

        # Infect a random share of the population
        I0 = int(self.p.initial_infection_share * self.p.population)
        self.agents.random(I0).condition = 1

    def update(self):
        """ Record variables after setup and each step. """

        # Record share of agents with each condition
        for i, c in enumerate(('S', 'I', 'R')):
            n_agents = len(self.agents.select(self.agents.condition == i))
            self[c] = n_agents / self.p.population
            self.record(c)

        # Stop simulation if disease is gone
        # if self.I == 0:
        #     self.stop()

    def step(self):
        """ Define the models' events per simulation step. """

        # Call 'being_sick' for infected agents
        self.agents.select(self.agents.condition == 1).being_sick()

    def end(self):
        """ Record evaluation measures at the end of the simulation. """

        # Record final evaluation measures
        self.report('Total share infected', self.I + self.R)
        self.report('Peak share infected', max(self.log['I']))


class Model:
    def __init__(self):
        self.parameters = {
            'population': 4000,
            'infection_chance': 0.6,
            'recovery_chance': 0.07,
            'initial_infection_share': 0.15,
            'number_of_neighbors': 4,
            'network_randomness': 0.8,
            'steps': 100
        }
        self.model = VirusModel

    def convert_to_tensor(self, results):
        numpy_array = np.array(results.variables.VirusModel.to_numpy().tolist())
        # Convert numpy array to PyTorch tensor
        # tensor = torch.from_numpy(numpy_array).float()
        return numpy_array
    
    def simulate(self, pars, T=None, seed=None):
        if not (pars is None):
            infection_chance, recovery_chance = [float(p) for p in pars]
        else:
            infection_chance, recovery_chance = 0.6, 0.07

        if not (T is None):
            assert isinstance(T, int) and (T > 0), "T must be positive int"
        else:
            T = 100

        comb_params = {
            'steps': T,
            'infection_chance': infection_chance,
            'recovery_chance': recovery_chance
        }

        print("THETAs: ", comb_params)

        parameters_multi = dict(self.parameters)
        parameters_multi.update(comb_params)

        model = self.model(parameters_multi)
        results = model.run()

        y = self.convert_to_tensor(results)
        print(y.shape)
        return y
