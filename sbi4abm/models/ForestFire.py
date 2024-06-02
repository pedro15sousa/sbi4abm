# Model design
import agentpy as ap
import numpy as np

class ForestModel(ap.Model):

    def setup(self):
        n_trees = int(self.p['Tree density'] * (self.p.size**2))
        trees = self.agents = ap.AgentList(self, n_trees)

        self.forest = ap.Grid(self, [self.p.size]*2, track_empty=True)
        self.forest.add_agents(trees, random=True, empty=True)

        # Condition 0: Alive, 1: Burning, 2: Burned
        self.agents.condition = 0

        unfortunate_trees = self.forest.agents[0:self.p.size, 0:2]
        unfortunate_trees.condition = 1

    def step(self):
        burning_trees = self.agents.select(self.agents.condition == 1)
        for tree in burning_trees:
            for neighbor in self.forest.neighbors(tree):
                if neighbor.condition == 0:
                    neighbor.condition = 1 # Neighbor starts burning
            tree.condition = 2 # Tree burns out

        # Stop simulation if no fire is left
        # if len(burning_trees) == 0:
        #     self.stop()

    def end(self):

        # Document a measure at the end of the simulation
        burned_trees = len(self.agents.select(self.agents.condition == 2))
        self.report('Percentage of burned trees',
                    burned_trees / len(self.agents))
        

class Model:
    def __init__(self):
        # self.parameters = {
        #     'Tree density': 0.5, # Percentage of grid covered by trees
        #     'size': 50, # Height and length of the grid
        #     'steps': 200,
        # }
        self.parameters = {
            'Tree density': 0.4, # Percentage of grid covered by trees
            'size': 100, # Height and length of the grid
            'steps': 200,
        }
        self.model = ForestModel

    def convert_data(self, results):
        numpy_array = np.array(results.variables.ForestModel.to_numpy().tolist())
        return numpy_array
    
    def convert_reporters(self, results):
        numpy_array = np.array(results.reporters.to_numpy().tolist())
        return numpy_array
    
    def simulate(self, pars, T=None, seed=None):
        if not (pars is None):
            tree_density, grid_size = [float(p) for p in pars]
            # tree_density = float(pars[0])
        else:
            tree_density, grid_size = 0.5, 50
            # tree_density = 0.4

        if not (T is None):
            assert isinstance(T, int) and (T > 0), "T must be positive int"
        else:
            T = 200

        comb_params = {
            'Tree density': tree_density,
            'size': int(grid_size),
            'steps': T
        }
        # comb_params = {
        #     'Tree density': tree_density,
        #     'steps': T
        # }

        print("THETAs: ", comb_params)

        parameters_multi = dict(self.parameters)
        parameters_multi.update(comb_params)

        model = self.model(parameters_multi)
        results = model.run()
        y = self.convert_reporters(results)
        return y

