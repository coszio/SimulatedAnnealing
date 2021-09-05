from math import exp
import numpy as np
from numpy.random import random
from TSP import TSP

KB = 1.380649e-23

class SimAnnealing:
    '''
    @author: Luis Cossio
    '''
    def __init__(self, n_iterations=500, init_temp=0.1, min_temp=0.000000001, alpha=0.1, beta=1.5, n_cities=10, n_dims=2):
        if not(0 < alpha < 1):
            raise AttributeError(f'alpha must be in within the range (0,1). Currently alpha={alpha}')
        if beta <= 1:
            raise AttributeError(f'beta must be greater than 1. Currently beta={beta}')
        self.alpha = alpha
        self.beta = beta
        self.n_cities = n_cities
        self.n_dims = n_dims
        self.n_iter = n_iterations
        self.temperature = init_temp
        self.min_temp = min_temp
        self.tsp = TSP(n_cities=n_cities, n_dimensions=n_dims)
        self.solution = self.random_solution()
        self.fitness = self.tsp.fitness(self.solution)


    def random_solution(self):
        return np.random.choice(self.n_cities, self.n_cities, replace=False)

    def p_acceptance(self, candidate_fitness):
        return exp(-abs(self.fitness - candidate_fitness) / self.temperature)

    def metropolis(self, candidate):
        cand_fit = self.tsp.fitness(candidate)
        if cand_fit < self.fitness or random.random() < self.p_acceptance(cand_fit):
            return candidate, cand_fit
        return self.solution, self.fitness

    def run(self):
        while self.temperature > self.min_temp:
            candidate = self.tsp.mutate(self.solution)
            self.solution, self.fitness = self.metropolis(candidate)
            self.best_fitnesses.append(self.fitness)
            self.temperature *= self.alpha

            
    def get_temp(self):
        self.temperature *= self.alpha
        return self.temperature