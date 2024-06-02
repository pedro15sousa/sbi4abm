import agentpy as ap
import numpy as np

class Person(ap.Agent):

    def setup(self):
        self.grid = self.model.grid
        self.random = self.model.random
        self.group = self.random.choice(range(self.p.n_groups))
        self.share_similar = 0
        self.happy = False

    def update_happiness(self):
        neighbors = self.grid.neighbors(self)
        similar = len([n for n in neighbors if n.group == self.group])
        ln = len(neighbors)
        self.share_similar = similar / ln if ln > 0 else 0
        self.happy = self.share_similar >= self.p.want_similar

    def find_new_home(self):
        new_spot = self.random.choice(self.model.grid.empty)
        self.grid.move_to(self, new_spot)


class SegregationModel(ap.Model):

    def setup(self):
        s = self.p.size
        n = self.n = int(self.p.density * (s ** 2))
        self.grid = ap.Grid(self, (s, s), track_empty=True)
        self.agents = ap.AgentList(self, n, Person)
        self.grid.add_agents(self.agents, random=True, empty=True)

    def update(self):
        self.agents.update_happiness()
        self.unhappy = self.agents.select(self.agents.happy == False)

        if len(self.unhappy) == 0:
            self.stop()

    def step(self):
        self.unhappy.find_new_home()

    def get_segregation(self):
        return round(sum(self.agents.share_similar) / self.n, 2)

    def end(self):
        self.report('segregation', self.get_segregation())


class Model:
    def __init__(self):
        self.parameters = {
            'want_similar': 0.3, # For agents to be happy
            'n_groups': 6, # Number of groups
            'density': 0.80, # Density of population
            'size': 50, # Height and length of the grid
            'steps': 100  # Maximum number of steps
        }
        # self.parameters = {
        #     'want_similar': 0.3, # For agents to be happy
        #     'n_groups': 3, # Number of groups
        #     'density': 0.80, # Density of population
        #     'size': 100, # Height and length of the grid
        #     'steps': 200  # Maximum number of steps
        # }
        self.model = SegregationModel

    def convert_to_tensor(self, results):
        numpy_array = np.array(results.variables.SegregationModel.to_numpy().tolist())
        # Convert numpy array to PyTorch tensor
        # tensor = torch.from_numpy(numpy_array).float()
        return numpy_array
    
    def convert_reporters(self, results):
        numpy_array = np.array(results.reporters.to_numpy().tolist())
        return numpy_array
    
    def simulate(self, pars, T=None, seed=None):
        if not (pars is None):
            want_similar, density = [float(p) for p in pars]
        else:
            want_similar, density = 0.3, 0.80

        if not (T is None):
            assert isinstance(T, int) and (T > 0), "T must be positive int"
        else:
            T = 100

        comb_params = {
            'steps': T,
            'want_similar': want_similar,
            'density': density
        }

        print("THETAs: ", comb_params)

        parameters_multi = dict(self.parameters)
        parameters_multi.update(comb_params)

        # model = self.model(parameters_multi)
        # results = model.run()

        model = self.model(parameters_multi)
        results = model.run()
        y = self.convert_reporters(results)
        return y

