from numba import njit
import numpy as np
import agentpy as ap
from scipy import stats
import torch

def normalise(v):
    """Normalise a vector to length 1."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Boid(ap.Agent):
    """
    An agent with a position and velocity in a continuous space that
    follows Craig Reynolds' flocking rules (3 rules) plus a fourth rule
    to avoid the edges of the simulation space.
    """
    
    def setup(self):
        self.velocity = normalise(
            self.model.nprandom.random(self.p.ndim) - 0.5)
        
    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):
        pos = self.pos
        ndim = self.p.ndim

        # Rule 1 - Cohesion
        nbs = self.neighbors(self, distance=self.p.outer_radius)
        nbs_len = len(nbs)
        nbs_pos_array = np.array(nbs.pos)
        nbs_vec_array = np.array(nbs.velocity)
        if nbs_len > 0:
            center = np.sum(nbs_pos_array, 0) / nbs_len
            v1 = (center - pos) * self.p.cohesion_strength
        else:
            v1 = np.zeros(ndim)

        # Rule 2 - Seperation
        v2 = np.zeros(ndim)
        for nb in self.neighbors(self, distance=self.p.inner_radius):
            v2 -= nb.pos - pos
        v2 *= self.p.seperation_strength

        # Rule 3 - Alignment
        if nbs_len > 0:
            average_v = np.sum(nbs_vec_array, 0) / nbs_len
            v3 = (average_v - self.velocity) * self.p.alignment_strength
        else:
            v3 = np.zeros(ndim)

        # Rule 4 - Borders
        v4 = np.zeros(ndim)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(ndim):
            if pos[i] < d:
                v4[i] += s
            elif pos[i] > self.space.shape[i] - d:
                v4[i] -= s

        # Update velocity
        self.velocity += v1 + v2 + v3 + v4
        self.velocity = normalise(self.velocity)

    def update_position(self):
        self.space.move_by(self, self.velocity)


class BoidsModel(ap.Model):

    def setup(self):
        self.space = ap.Space(self, shape=[self.p.size]*self.p.ndim)
        self.agents = ap.AgentList(self, self.p.population, Boid)
        self.space.add_agents(self.agents, random=True)
        self.agents.setup_pos(self.space)

    def step(self):
        """ Defines the models' events per simulation step. """

        self.agents.update_velocity()  # Adjust direction
        self.agents.update_position()  # Move into new direction

    def update(self):
        """ Record agents' positions after setup and each step."""
        # self.record('positions', self.agents.pos)
        for i, agent in enumerate(self.agents):
            self.record(f'agent_{i}', list(agent.pos))

class Model:

    def __init__(self):
        self.parameters = {
            'size': 50,
            'seed': 123,
            'steps': 200,
            'ndim': 2,
            'population': 200,
            'inner_radius': 3,
            'outer_radius': 10,
            'border_distance': 15,
            'cohesion_strength': 0.25,
            'seperation_strength': 0.15,
            'alignment_strength': 0.45,
            'border_strength': 0.5
        }
        self.model = BoidsModel

    def convert_to_tensor(self, results):
        numpy_array = np.array(results.variables.BoidsModel.to_numpy().tolist())
        # Convert numpy array to PyTorch tensor
        # tensor = torch.from_numpy(numpy_array).float()
        return numpy_array

    def simulate(self, pars, T=None, seed=None):

        if not (pars is None):
            cohesion_stren, seperation_stren, alignment_stren, border_stren = [float(p) for p in pars]
        else:
            cohesion_stren, seperation_stren, alignment_stren, border_stren = 0.25, 0.15, 0.45, 0.5

        if not (T is None):
            assert isinstance(T, int) and (T > 0), "T must be positive int"
        else:
            T = 200

        comb_params = {
            'steps': T,
            'cohesion_strength': cohesion_stren,
            'seperation_strength': seperation_stren,
            'alignment_strength': alignment_stren,
            'border_strength': border_stren
        }
        
        print("Thetas: ", comb_params)

        parameters_multi = dict(self.parameters)
        parameters_multi.update(comb_params)

        model = self.model(parameters_multi)
        results = model.run()

        # if comb_params is not None:
        #     parameters_multi = dict(self.parameters)
        #     parameters_multi.update(comb_params)
        #     sample = ap.Sample(parameters_multi)

        #     exp = ap.Experiment(self.model, sample, iterations=iters)
        #     results = exp.run(n_jobs, verbose=5)
        # else:
        #     model = self.model(self.parameters)
        #     results = model.run()

        results.save()
        y = self.convert_to_tensor(results)
        return y

