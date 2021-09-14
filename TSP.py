import numpy as np
from functools import lru_cache
from itertools import accumulate
import matplotlib.pyplot as plt
class TSP:
    def __init__(self, n_travelers=3, n_cities=50, n_dimensions=2, trade_cities=False, shortest_travel=3):
        self.n_cities = n_cities
        self.n_dimensions = n_dimensions
        self.n_travelers = n_travelers
        self.trade_cities = trade_cities
        self.shortest_travel = shortest_travel
        self.cities = np.random.random_sample(size=(n_cities, n_dimensions))

    @lru_cache(maxsize=1000)
    def dist(self, origin, destination):
        o = self.cities[origin]
        d = self.cities[destination]
        distance = euclidian_distance(o, d)
        return distance
    
    def random_solution(self):
        sol = np.random.choice(self.n_cities, self.n_cities, replace=False)
        step_size = self.n_cities//self.n_travelers
        return np.split(sol, range(step_size, step_size*self.n_travelers, step_size))

    def fitness(self, solution):
        fitness = 0
        for travel in solution:
            travel = np.asarray(travel)
            for i in range(np.size(travel)):
                fitness += self.dist(travel[i-1], travel[i])
        return fitness
    
    def mutate(self, solution):
        split_indices = list(accumulate([np.size(x) for x in solution]))[:-1]
        conc_solution = np.concatenate(solution)

        # Shift
        shift = np.random.randint(self.n_cities)
        mutation = np.roll(conc_solution,shift)

        # Mutate
        i = np.random.randint(self.n_cities-2)
        l = np.random.randint(low=i + 2, high=i+2+self.n_cities//2)
        l = l if l < self.n_cities else self.n_cities
        mutation[i:l] = np.random.choice(mutation[i:l], l-i, replace=False)

        # Unshift
        mutation = np.roll(mutation, -shift)

        # trade cities between travelers
        if self.trade_cities:
            j = np.random.randint(1,len(split_indices)+1)
            delta = np.random.randint(-1, 1)
            split_indices = [0] + split_indices + [self.n_cities]
            if split_indices[j-1] + self.shortest_travel-1 < split_indices[j] + delta < split_indices[j+1] - self.shortest_travel-1:
                split_indices[j] += delta
            split_indices = split_indices[1:-1]

        # resplit
        mutation = np.split(mutation, split_indices)
        return mutation

    def plot_solution(self, solution):
        positions = self.cities
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)         # Prepare 2 plots
        ax[0].set_title('Raw nodes')
        ax[1].set_title('Optimized tour')
        ax[0].scatter(positions[:, 0], positions[:, 1])             # plot A
        ax[1].scatter(positions[:, 0], positions[:, 1])             # plot B
        
        # Draw arrows
        for travel in solution:
            travel = np.asarray(travel)
            c = list(np.random.rand(3))
            for i in range(np.size(travel)):
                start_pos = positions[travel[i-1]]
                end_pos = positions[travel[i]]
                ax[1].annotate("",
                        xy=start_pos, xycoords='data',
                        xytext=end_pos, textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3",
                                        color=c))
        # Info box
        textstr = "N nodes: %d\nN travelers: %d\nTotal length: %.3f" % (self.n_cities,self.n_travelers, self.fitness(solution))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=9, # Textbox
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.show()

def euclidian_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a-b)









    # TESTS
def test_euclidian():
    assert euclidian_distance(1, 5) == 4
    assert euclidian_distance(0,0) == 0
    assert euclidian_distance((0,0), (3,4)) == 5