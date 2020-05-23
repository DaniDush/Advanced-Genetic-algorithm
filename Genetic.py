import time
from random import randint, shuffle
import random
import numpy as np
from copy import deepcopy
from NQueens import n_queens
from KnapSack import knap_sack
from BoolPegia import bool_pgia
from BinPacking import bin_packing

GA_POPSIZE = 100
GA_MAXITER = 16384
GA_ELITRATE = 0.2
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

    def __init__(self, problem=0, speciation_threshold=4):
        self.genomes = []
        self.buffer = []
        self.problem = problem
        self.species_list = []
        self.speciation_threshold = speciation_threshold
        self.optimal_species = 30
        self.distance_matrix = []

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
                _bool = bool_pgia(GA_TARGET, tsize)
                self.genomes.append(Genome(gene=_bool))
                self.buffer.append(Genome(gene=bool_pgia(GA_TARGET)))

        #   Solving Bin packing problem
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
        # If its maximize problem
        if self.problem == 1 or self.problem == 3:
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

        # Fitness share method
        # self.fitness_share()

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

        return avg_fitness, std_fitness

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
                sum_of_bins = [0] * tsize
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

            # Calling probabilistic crowding
            self.buffer[i] = self.probabilistic_crowding(self.genomes[i1], self.genomes[i2], self.buffer[i])

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

    @staticmethod
    def sharing_function(distance, sigma_share, alpha):
        return 1 - (distance / sigma_share) ** alpha if distance < sigma_share else 0

    def fitness_share(self):
        """ Implementation of share fitness method for scaling fitness's"""

        start = time.time()

        alpha = 1
        sigma_share = 15500
        tsize = self.genomes[0].gene.N
        sharing_matrix = np.zeros((tsize, tsize))
        print(tsize)

        for i in range(0, tsize):
            for j in range(i + 1, tsize):
                sharing_matrix[i, j] = self.sharing_function(self.genomes[i].gene.calc_distance(self.genomes[j].gene),
                                                             sigma_share=sigma_share, alpha=alpha)
                sharing_matrix[j, i] = sharing_matrix[i, j]
        print("Finish scaling")
        if self.problem == 1 or self.problem == 3:
            for i in range(0, tsize):
                self.genomes[i].fitness /= sum(sharing_matrix[i])
        else:
            for i in range(0, tsize):
                self.genomes[i].fitness *= sum(sharing_matrix[i])

        print("Share fitness time: ", time.time() - start)

    @staticmethod
    def probabilistic_crowding(parent_1, parent_2, children=None, is_crossover=True):
        """ Probabilistic Crowding, We will run a tournament and save the winner to the next generation """
        if is_crossover:
            # First we will run a competition between parents
            parent_1_chance = parent_1.fitness / (parent_1.fitness + parent_2.fitness)
            random_roll = random.random()
            # Set Winner
            if random_roll <= parent_1_chance: winner = parent_1
            else: winner = parent_2

            # Now we will run it between the parent and the children
            children_fitness = children.gene.get_fitness()

            children_chance = children_fitness / (children_fitness + parent_2.fitness)
            random_roll = random.random()
            if random_roll <= children_chance: winner = children
            else: winner = winner

            return winner

    def handle_local_optima(self):
        self.add_random_immigrants()

    def add_random_immigrants(self):
        """ Adding random immigrants for the population by iterating over all of the population and replace citizen
            by probability of 0.1 """

        immigrants_indices = []
        for i in range(GA_POPSIZE):
            if self.genomes[0].gene.calc_distance(self.genomes[i].gene):
                random_roll = random.random()

                if random_roll <= 0.15:
                    immigrants_indices.append(i)
                    # Creating new citizen according to the problem were trying to solve
                    if self.problem == 0:
                        N = self.genomes[0].gene.N
                        game = n_queens(N=N)
                        self.genomes[i] = Genome(game)

                    elif self.problem == 1:
                        N = self.genomes[0].gene.N
                        sack = knap_sack(N=N)
                        self.genomes[i] = Genome(gene=sack)

                    elif self.problem == 2:
                        string_size = self.genomes[0].gene.size
                        _bool = bool_pgia(GA_TARGET, string_size)
                        self.genomes[i] = Genome(gene=_bool)

                    else:
                        N = self.genomes[0].gene.N
                        bins = bin_packing(N, False)
                        self.genomes[i] = Genome(gene=bins)

        # Calc fitness for the new citizens
        # for idx in immigrants_indices:
        #     print(idx)
        #     self.genomes[idx].gene.fitness = self.genomes[idx].gene.get_fitness()

        self.calc_fitness()
        # Sorting the new population
        self.sort_by_fitness()

    def make_species(self, phenotypes, adjust_threshold=True):
        """ Update species_list """

        self.species_list = []
        for p in phenotypes:
            selected_species = self.find_species(p)
            if len(selected_species) == 0:  # If we didnt find matching species we will ad one
                self.species_list.append(selected_species)  # initialize a new population

            selected_species.append(p)

        if adjust_threshold:
            self.adjust_speciation_threshold(phenotypes)
        return self.species_list

    def find_species(self, i):
        """searching for matching species from species_list"""

        for species in self.species_list:
            if self.genomes[i].gene.calc_distance(species[0]) < self.speciation_threshold:
                return species
        # If we didnt found matching species
        return None

    def adjust_speciation_threshold(self, max_iterations=1):
        """Adjust the number of species to be around max_species"""
        iterations = 0

        while len(
                self.species_list) != self.optimal_species and self.speciation_threshold > 0 and iterations < max_iterations:
            iterations += 1
            self.speciation_threshold += 1 * np.sign(len(self.species_list) - self.optimal_species)
            self.speciation_threshold = max(1, self.speciation_threshold)
            self.species_list = self.make_species(self.genomes, adjust_threshold=False)

    def calc_similarity(self, citizen_one, citizen_two):
        distance = 0
        tsize = self.genomes[0].gene.N

        for i in range(tsize):
            if citizen_one.board[col] != citizen_two.board[col]:
                similarity = similarity + 1
        return round((similarity / self.data.queens_num), 3)