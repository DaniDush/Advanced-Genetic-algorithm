from random import randint, shuffle
import random
import numpy as np
from copy import deepcopy
from NQueens import n_queens
from KnapSack import knap_sack
from BoolPegia import bool_pgia
from BinPacking import bin_packing, Items

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.1
GA_MUTATIONRATE = 0.25
GA_TARGET = "Hello world!"
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

    def mutate(self):
        """ Basic mutation implementation """
        tsize = len(self.gene)
        ipos = randint(0, tsize - 1)
        delta = randint(32, 122)
        self.gene[ipos] = chr((ord(self.gene[ipos]) + delta) % 122)

    def swap_mutation(self):
        """ Swap mutation - randomly pick 2 indexes and swap their values"""
        tsize = len(self.gene)
        ipos_1 = randint(0, tsize - 1)
        ipos_2 = randint(0, tsize - 1)
        while ipos_1 == ipos_2:
            ipos_2 = randint(0, tsize - 1)

        self.gene[ipos_1], self.gene[ipos_2] = self.gene[ipos_2], self.gene[ipos_1]

    def scramble_mutation(self):
        """ Scramble mutation - Choose 2 random indexes and shuffle the string within that range """
        tsize = len(self.gene)
        ipos_1 = randint(0, int(tsize / 2) - 1)
        ipos_2 = randint(ipos_1 + 1, tsize - 1)

        self.gene.shuffle(ipos_1, ipos_2)

    def aging(self):
        """ Adding age cost calculated by MSE from the optimal age. """
        mean_age = 5
        age_score = abs(self.age - mean_age)
        return age_score

    def get_fitness(self):
        return self.fitness


class Population:

    def __init__(self, problem=0):
        self.genomes = []
        self.buffer = []
        self.problem = problem

    def init_population(self, N=None):
        """ Population initialize - if is_queen is True we will generate random permutation of integers with the given
            range. """

        tsize = len(GA_TARGET)
        #   Solving N Queens problem
        if self.problem == 0:
            for i in range(GA_POPSIZE):
                game = n_queens(N=N)
                self.genomes.append(Genome(game))
                self.buffer.append(Genome(gene=game))  # We will append empty Genome instead of doing resize
        #   Solving Knap Sack problem
        elif self.problem == 1:
            for i in range(GA_POPSIZE):
                sack = knap_sack(N=N)
                self.genomes.append(Genome(gene=sack))
                self.buffer.append(Genome(gene=knap_sack()))  # We will append empty Genome instead of doing resize
        #   Solving String problem
        elif self.problem == 2:
            for i in range(GA_POPSIZE):
                bool = bool_pgia(GA_TARGET, tsize)
                self.genomes.append(Genome(gene=bool))
                self.buffer.append(Genome(gene=bool_pgia(GA_TARGET)))
        else:
            for i in range(GA_POPSIZE):
                bins = bin_packing(N, False)
                self.genomes.append(Genome(gene=bins))
                self.buffer.append(Genome(gene=bin_packing(N, True)))

    def calc_fitness(self):
        for i in range(GA_POPSIZE):
            temp_fitness = self.genomes[i].calc_individual_fitness()
            self.genomes[i].fitness = temp_fitness

    def sort_by_fitness(self):
        if self.problem == 1:
            self.genomes.sort(reverse=True)  # Sort population using its fitness
        else:
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

        if cross_method == 1:
            self.one_point_crossover(esize=esize)

        elif cross_method == 2:
            self.two_point_crossover(esize=esize)

        elif cross_method == 3:
            self.uniform_crossover(esize=esize)

        elif cross_method == 4:
            self.ordered_crossover(esize=esize)

        elif cross_method == 5:
            self.CX_crossover(esize=esize)

    def get_best_fitness(self):
        return self.genomes[0].fitness

    def print_best(self):
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

    def one_point_crossover(self, esize):
        tsize = len(GA_TARGET)

        for i in range(esize, GA_POPSIZE):
            i1 = randint(0, int(GA_POPSIZE / 2) - 1)
            i2 = randint(0, int(GA_POPSIZE / 2) - 1)
            spos = randint(0, tsize - 2)

            obj = self.genomes[i1].gene[:spos] + self.genomes[i2].gene[spos:]

            self.buffer[i].gene.set_obj(obj=obj)
            if random.random() < GA_MUTATIONRATE:
                if self.problem == 2:
                    Genome.mutate(self.buffer[i])
                else:
                    Genome.swap_mutation(self.buffer[i])

    def two_point_crossover(self, esize):
        tsize = len(GA_TARGET)

        for i in range(esize, GA_POPSIZE):
            # picking 2 genomes
            i1 = randint(0, int(GA_POPSIZE / 2) - 1)
            i2 = randint(0, int(GA_POPSIZE / 2) - 1)

            # picking 2 points
            spos_1 = randint(0, tsize - 2)
            spos_2 = randint(spos_1, tsize - 1)

            obj = self.genomes[i1].gene[:spos_1] + self.genomes[i2].gene[spos_1:spos_2] + self.genomes[i1].gene[spos_2:]

            self.buffer[i].gene.set_obj(obj=obj)

            if random.random() < GA_MUTATIONRATE:
                Genome.scramble_mutation(self.buffer[i])

    def uniform_crossover(self, esize):
        tsize = len(self.genomes[0].gene)
        for i in range(esize, GA_POPSIZE):
            obj = [None] * tsize
            # picking 2 genomes
            i1 = randint(0, int(GA_POPSIZE / 2) - 1)
            i2 = randint(0, int(GA_POPSIZE / 2) - 1)

            for j in range(tsize):
                if random.random() <= UNIFORM_PR:
                    obj[j] = self.genomes[i1].gene[j]
                else:
                    obj[j] = self.genomes[i2].gene[j]

            self.buffer[i].gene.set_obj(obj=obj)

            if random.random() < GA_MUTATIONRATE:
                self.buffer[i].mutate()

    def ordered_crossover(self, esize):
        tsize = self.genomes[0].gene.N
        for i in range(esize, GA_POPSIZE):
            obj = []
            obj_1 = []
            obj_2 = []
            remains = []

            # picking 2 genomes
            i1 = randint(0, GA_POPSIZE - 2)
            i2 = randint(i1, GA_POPSIZE - 1)

            # picking 2 points
            spos_1 = randint(0, tsize - 2)
            spos_2 = randint(spos_1, tsize - 1)

            # Solving for bin packing
            if self.problem == 3:
                sum_of_bins = [0]*tsize
                for j in range(spos_1, spos_2):
                    obj_1.append(self.genomes[i1].gene.bins[j])
                    sum_of_bins[self.genomes[i1].gene.bins[j]] += bin_packing.W[j]

                for j in range(tsize):
                    if spos_1 <= j < spos_2:
                        continue

                    _bin = self.genomes[i1].gene.bins[j]
                    if sum_of_bins[_bin] + bin_packing.W[j] <= bin_packing.C and sum_of_bins[_bin] != 0:
                        sum_of_bins[_bin] += bin_packing.W[j]
                        obj_2.append(_bin)

                    else:
                        while True:
                            rand_bin = randint(0, tsize - 1)
                            if sum_of_bins[rand_bin] + bin_packing.W[j] <= bin_packing.C:
                                obj_2.append(rand_bin)
                                sum_of_bins[rand_bin] += bin_packing.W[j]
                                break

                obj = obj_2[:spos_1] + obj_1 + obj_2[spos_1:]

            # Solving for other problems
            else:
                for j in range(spos_1, spos_2):
                    obj_1.append(self.genomes[i1].gene[j])

                for val in self.genomes[i2].gene:
                    if val not in obj_1:
                        obj_2.append(val)
                    else:
                        remains.append(val)

                # Insert missing values
                if len(obj) < tsize:
                    for j in range(tsize):
                        if j not in obj:
                            obj.append(j)
                        if len(obj) == tsize:
                            break

            self.buffer[i].gene.set_obj(obj=obj)

            if random.random() < GA_MUTATIONRATE:
                Genome.swap_mutation(self.buffer[i])

    def CX_crossover(self, esize):
        tsize = len(self.genomes[0].gene)
        for j in range(esize, GA_POPSIZE, 2):
            # first we will create the cycle
            cycles = [-10] * tsize
            cycle_count = 1

            # picking 2 genomes
            i1 = randint(0, int(GA_POPSIZE / 2) - 1)
            i2 = randint(0, int(GA_POPSIZE / 2) - 1)

            point_generator = (i for i, v in enumerate(cycles) if v < 0)

            # Iterate over point_generator which tell us in which point the cycle continue
            for i in point_generator:

                while cycles[i] < 0:
                    cycles[i] = cycle_count
                    i = self.genomes[i1].gene[self.genomes[i2].gene[i]]

                cycle_count += 1

            obj = [self.genomes[i1].gene[i] if n % 2 else self.genomes[i2].gene[i] for i, n in
                   enumerate(cycles)]

            self.buffer[j].gene.set_obj(obj=obj)

            obj = [self.genomes[i1].gene[i] if n % 2 else self.genomes[i2].gene[i] for i, n in
                   enumerate(cycles)]

            self.buffer[j + 1].gene.set_obj(obj=obj)

            if random.random() < GA_MUTATIONRATE:
                Genome.scramble_mutation(self.buffer[j])
            if random.random() < GA_MUTATIONRATE:
                Genome.scramble_mutation(self.buffer[j + 1])

    def handle_crossover(self, buffer, bin_weights, bin_items):
        """ Moving items from over-weighted bins """
        for idx, weight in enumerate(bin_weights):
            if weight > buffer.gene.C:
                # sort items by weight
                items = bin_items[idx]
                items.sort()
                for item in items:
                    if weight - item.weight <= buffer.gene.C:
                        t = 0
                        while True:
                            if bin_weights[t] + item.weight <= buffer.gene.C:
                                buffer.gene.bins[item.get_id()] = Items(item.id, item.weight, t)
                                bin_weights[t] += item.weight
                                weight -= item.weight
                                break
                            t += 1
                    break

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
