from random import randint, shuffle
import random
import numpy as np
from copy import deepcopy
from BinPacking import bin_packing

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.1
GA_MUTATIONRATE = 0.25
UNIFORM_PR = 0.5

WEIGHT_BOUND = 100


class Genome:
    """ Genome class """

    def __init__(self, gene, fitness=0):
        """ Constructor """
        self.gene = gene  # the string
        self.fitness = fitness  # its fitness
        self.age = 0  # for aging

    def __gt__(self, other):
        """ Override comparison method (for sorting) """
        return self.fitness > other.fitness

    def calc_individual_fitness(self):
        """ calculation of our gene score with chosen heuristic and aging score """
        temp_fitness = self.gene.get_fitness()
        # Adding extra aging score
        temp_fitness += self.aging()
        return temp_fitness

    def swap_mutation(self, num_of_mutations=4):
        """ Swap mutation - randomly pick 2 indexes and swap their values"""
        tsize = len(self.gene)
        for i in range(num_of_mutations):
            ipos_1 = randint(0, tsize - 2)
            ipos_2 = randint(ipos_1 + 1, tsize - 1)

            self.gene[ipos_1], self.gene[ipos_2] = self.gene[ipos_2], self.gene[ipos_1]

    def aging(self):
        """ Adding age cost calculated by MSE from the optimal age. """
        mean_age = 5
        age_score = abs(self.age - mean_age)
        return age_score

    def get_fitness(self):
        return self.fitness


class Population:

    def __init__(self):
        self.genomes = []
        self.buffer = []

    def init_population(self, N, C, WEIGHTS):
        """ Population initialize - if is_queen is True we will generate random permutation of integers with the given
            range. """

        items = np.arange(N)
        for i in range(GA_POPSIZE):
            np.random.shuffle(items)
            bins = bin_packing(items.copy(), N, C, WEIGHTS)
            self.genomes.append(Genome(gene=bins))
            self.buffer.append(Genome(gene=bin_packing([], N, C, WEIGHTS)))

    def calc_fitness(self):
        for i in range(GA_POPSIZE):
            temp_fitness = self.genomes[i].calc_individual_fitness()
            self.genomes[i].fitness = temp_fitness

    def sort_by_fitness(self):
        self.genomes.sort()  # Sort population using its fitness

    def elitism(self, esize):
        for i in range(esize):
            self.buffer[i] = deepcopy(self.genomes[i])
            self.buffer[i].age += 1  # Kept for another generation

    def mate(self, cross_method=1, selection_method=0):
        esize = int(GA_POPSIZE * GA_ELITRATE)

        # Taking the best citizens
        self.elitism(esize)

        # If we choose parent selection then we will choose parents as the number of psize-esize
        if selection_method == 1:
            num_of_parents = GA_POPSIZE - esize
            # Parents selection method - SUS
            self.genomes[esize:] = self.SUS(num_of_parents=num_of_parents)

        elif selection_method == 2:
            num_of_parents = GA_POPSIZE - esize
            self.genomes[esize:] = self.tournament_selection(num_of_parents=num_of_parents)

        self.two_point_crossover(esize=esize)

    def get_best_fitness(self):
        return self.genomes[0].fitness

    def print_best(self):
        y = set(self.genomes[0].gene.bins)
        print("Number of bins in use: ", len(y))
        print('Best: ', self.genomes[0].gene, '(', self.genomes[0].fitness, ')')

    def calc_avg_std(self):
        """ 1.1. Calculate and report avg fitness value of each generation in population
            + Calculate the std from mean"""
        arr_size = len(self.genomes)
        fitness_array = np.empty(arr_size)

        for idx, citizen in enumerate(self.genomes):
            fitness_array[idx] = citizen.fitness

        avg_fitness = fitness_array.mean()
        std_fitness = fitness_array.std()

        print('Average fitness:', avg_fitness, '\nStandard deviation: ', std_fitness, '\n')

    def swap(self):
        self.genomes, self.buffer = self.buffer, self.genomes

    def two_point_crossover(self, esize, num_of_crossovers=2):
        tsize = len(self.genomes[0].gene)

        for i in range(esize, GA_POPSIZE, 2):

            for j in range(num_of_crossovers):
                # picking 2 genomes
                i1 = randint(0, GA_POPSIZE - 2)
                i2 = randint(i1 + 1, GA_POPSIZE - 1)

                child_1 = self.genomes[i1].gene.bins.copy()
                child_2 = self.genomes[i2].gene.bins.copy()

                # picking 2 points for each parent
                indexes = np.random.randint(tsize, size=2)
                # sort indexes
                indexes = np.sort(indexes)

                # Crossover
                child_1[indexes[0]:indexes[1]], child_2[indexes[0]:indexes[1]] = \
                    child_2[indexes[0]:indexes[1]], child_1[indexes[0]:indexes[1]]

                # Now we need to handle duplicates or missing id's
                #child_1 = self.handle_crossover(i1, child_1)
                #child_2 = self.handle_crossover(i2, child_2)

                self.buffer[i].gene.set_obj(obj=child_1)
                self.buffer[i + 1].gene.set_obj(obj=child_2)

            if random.random() < GA_MUTATIONRATE:
                Genome.swap_mutation(self.buffer[i])
            if random.random() < GA_MUTATIONRATE:
                Genome.swap_mutation(self.buffer[i + 1])

    def handle_crossover(self, idx, child):
        mask = np.isin(self.genomes[idx].gene.bins, child, invert=True)
        missing = self.genomes[idx].gene[mask].tolist()
        unique, count = np.unique(child, return_counts=True)
        duplicated = unique[count > 1].tolist()
        for idx, val in np.ndenumerate(child):
            if val in duplicated:
                child[idx] = missing.pop()
                duplicated.remove(val)

        return child

    def SUS(self, num_of_parents):
        """ Parent selection method - Stochastic Universal Sampling (SUS)"""
        # Summing all fitness's for each citizen in population
        F = sum(genome.fitness for genome in self.genomes)
        # Step size
        dist = F / num_of_parents
        # Our starting point - small int between [0, step_size] chose uniformly
        start_point = np.random.uniform(0, dist)

        points = [start_point + i * dist for i in range(num_of_parents)]

        selected = set()
        while len(selected) < num_of_parents:
            cum_fitness = 0
            inv = 0
            shuffle(self.genomes)
            for i in range(len(points)):
                while cum_fitness <= points[i]:
                    cum_fitness += self.genomes[inv].get_fitness()
                    inv += 1
                if len(selected) > num_of_parents or inv >= GA_POPSIZE:
                    break
                selected.add(self.genomes[inv])

        return selected

    def tournament_selection(self, num_of_parents):
        """ choosing k individuals for a tournament, winner pass for crossover """
        selected = []
        k = 6
        pr = 0.3
        while len(selected) < num_of_parents:
            best_fitness = np.inf
            best_inv = None

            # Choose k random indexes from the population
            inv_to_check = set(np.random.randint(GA_POPSIZE, size=k))
            if random.random() > pr * (1 - pr) ** len(selected):
                # Getting the best gene
                for inv in inv_to_check:
                    if self.genomes[inv].fitness < best_fitness:
                        best_fitness = self.genomes[inv].fitness
                        best_inv = self.genomes[inv]
                selected.append(best_inv)

        return selected
