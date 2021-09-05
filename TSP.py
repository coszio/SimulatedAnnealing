import numpy as np
from functools import lru_cache

class TSP:
    def __init__(self, n_cities=10, n_dimensions=2):
        self.n_cities = n_cities
        self.n_dimensions = n_dimensions
        self.cities = np.random.random_sample(size=(n_cities, n_dimensions))

    @lru_cache(maxsize=1000)
    def dist(self, origin, destination):
        o = self.cities[origin]
        d = self.cities[destination]
        distance = euclidian_distance(o, d)
        return distance
    

    def fitness(self, solution):
        fitness = 0
        for i in range(self.n_cities):
            fitness += self.dist(solution[i-1], solution[i])
        return fitness
    
    def mutate(self, solution):
        # Shift
        shift = np.random.randint(self.n_cities)
        mutation = np.roll(solution,shift)

        # Mutate
        i = np.random.randint(self.n_cities)
        l = i + np.random.randint(i, self.n_cities)
        mutation[i:l] = np.random.choice(mutation[i:l], l-i, replace=False)

        # Unshift
        mutation = np.roll(mutation, -shift)
        return mutation


def euclidian_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a-b)









    # TESTS
def test_euclidian():
    assert euclidian_distance(1, 5) == 4
    assert euclidian_distance(0,0) == 0
    assert euclidian_distance((0,0), (3,4)) == 5