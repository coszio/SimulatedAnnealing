from math import exp

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random

from TSP import TSP

KB = 1.380649e-23

class SimAnnealing:
    '''
    @author: Luis Cossio
    '''
    def __init__(self, 
                 n_iterations=50000, 
                 first_temp=0.1, 
                 min_temp=0.0001, 
                 alpha=0.0001, 
                 beta=1.01, 
                 n_cities=30, 
                 n_travelers=3, 
                 trade_cities=False, 
                 shortest_travel=4):
        if not(0 < alpha < 1):
            raise AttributeError(f'alpha must be in within the range (0,1). Currently alpha={alpha}')
        if beta <= 1:
            raise AttributeError(f'beta must be greater than 1. Currently beta={beta}')
        self.alpha = alpha
        self.beta = beta
        self.n_cities = n_cities
        self.n_iter = n_iterations
        self.min_temp = min_temp
        self.tsp = TSP(n_cities=n_cities, 
                        n_travelers=n_travelers, 
                        trade_cities=trade_cities, 
                        shortest_travel=shortest_travel)
        self.solution = self.tsp.random_solution()
        self.fitness = self.tsp.fitness(self.solution)
        self.temperature = self.init_temp(first_temp=first_temp)
        self.best_solutions = []
        self.best_fitnesses = []

    def cadena_markov(self, L0=30):
        accepted = 0
        for _ in range(L0):
            candidate = self.tsp.mutate(self.solution)
            cand_fit = self.tsp.fitness(candidate)
            if cand_fit < self.fitness or random() < self.p_acceptance(cand_fit):
                self.solution, self.fitness = candidate, cand_fit
                accepted += 1
        return accepted / L0

    def init_temp(self, first_temp=0.1, L0=30, expected_p=0.8):
        p = 0
        self.temperature = first_temp
        while p < expected_p:
            p = self.cadena_markov(L0)
            self.temperature *= self.beta
        return self.temperature

    def p_acceptance(self, candidate_fitness):
        return np.exp(-abs(self.fitness - candidate_fitness) / self.temperature)

    def metropolis(self, candidate):
        cand_fit = self.tsp.fitness(candidate)
        if cand_fit < self.fitness or random() < self.p_acceptance(cand_fit):
            return candidate, cand_fit
        return self.solution, self.fitness

    def update_temp(self):
        self.temperature *= 1-self.alpha #(self.best_fitnesses[-2] / self.best_fitnesses[-1])

    def run(self):
        i = 0
        while self.temperature > self.min_temp and i < self.n_iter:
            self.cadena_markov(L0=1)
            self.best_fitnesses.append(self.fitness)
            self.best_solutions.append(self.solution)
            self.update_temp()
            i += 1

    def plot_solution(self):
        self.tsp.plot_solution(self.solution)

if __name__ == '__main__':
    sa = SimAnnealing()
    sa.run()

    plt.xlabel('iteraciones')
    plt.ylabel('menor encontrado')
    plt.plot(range(len(sa.best_fitnesses)), sa.best_fitnesses)

    plt.show()

    sa.plot_solution()
