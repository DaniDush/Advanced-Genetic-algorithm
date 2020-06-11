from random import randint, shuffle
import random
import numpy as np
from copy import deepcopy
from NQueens import n_queens
from KnapSack import knap_sack
from BoolPegia import bool_pgia
from BinPacking import bin_packing
import Baldwin
from GeneticProgramming import GP_tree

GA_POPSIZE = 1000
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
        #temp_fitness += self.aging()
        return temp_fitness

    def mutate(self, num_of_mutations):
        """ Basic mutation implementation """
        for i in range(num_of_mutations):
            tsize = len(self.gene)
            ipos = randint(0, tsize - 1)
            delta = randint(32, 122)
            self.gene[ipos] = chr((ord(self.gene[ipos]) + delta) % 122)

    def swap_mutation(self, num_of_mutations):
        """ Swap mutation - randomly pick 2 indexes and swap their values"""
        for i in range(num_of_mutations):
            tsize = len(self.gene)
            ipos_1 = randint(0, tsize - 1)
            ipos_2 = randint(0, tsize - 1)
            while ipos_1 == ipos_2:
                ipos_2 = randint(0, tsize - 1)
            self.gene[ipos_1], self.gene[ipos_2] = self.gene[ipos_2], self.gene[ipos_1]

    def scramble_mutation(self, num_of_mutations):
        """ Scramble mutation - Choose 2 random indexes and shuffle the string within that range """
        for i in range(num_of_mutations):
            tsize = len(self.gene)
            ipos_1 = randint(0, int(tsize / 2) - 1)
            ipos_2 = randint(ipos_1 + 1, tsize - 1)

            self.gene.shuffle(ipos_1, ipos_2)

    def bit_flip(self):
        tsize = len(self.gene)
        ipos_1 = randint(0, tsize - 2)
        ipos_2 = randint(0, tsize - 1)

        while self.gene.solution[ipos_1] == '?':
            ipos_1 = randint(0, tsize - 2)

        while self.gene.solution[ipos_2] == '?':
            ipos_2 = randint(0, tsize - 2)

        if self.gene.solution[ipos_2] == '1':
            self.gene.solution[ipos_2] = '0'
        elif self.gene.solution[ipos_2] == '0':
            self.gene.solution[ipos_2] = '1'

        if self.gene.solution[ipos_1] == '1':
            self.gene.solution[ipos_1] = '0'
        elif self.gene.solution[ipos_1] == '0':
            self.gene.solution[ipos_1] = '1'

    def branch_mutation(self):
        self.gene.branch_generator()

    def aging(self):
        """ Adding age cost calculated by MSE from the optimal age. """
        mean_age = 5
        age_score = abs(self.age - mean_age)
        return age_score

    def get_fitness(self):
        return self.fitness


class Population:

    def __init__(self, problem=0, speciation_threshold=6.3):
        self.genomes = []
        self.buffer = []
        self.mutation_rate = 0.25
        self.problem = problem
        self.species_list = []
        self.speciation_threshold = speciation_threshold
        self.optimal_species = 30
        self.distance_matrix = []
        self.num_of_mutations = 1

    def init_population(self, N=None):
        """ Population initialize - if is_queen is True we will generate random permutation of integers with the given
            range. """
        global GA_POPSIZE

        tsize = len(GA_TARGET)
        #   Solving N Queens problem
        if self.problem == 0:
            GA_POPSIZE = 1000
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
            GA_POPSIZE = 5000
            for i in range(GA_POPSIZE):
                _bool = bool_pgia(GA_TARGET, tsize)
                self.genomes.append(Genome(gene=_bool))
                self.buffer.append(Genome(gene=bool_pgia(GA_TARGET)))

        #   Solving Bin packing problem
        elif self.problem == 3:
            GA_POPSIZE = 400
            for i in range(GA_POPSIZE):
                bins = bin_packing(N, False)
                self.genomes.append(Genome(gene=bins))
                self.buffer.append(Genome(gene=bin_packing(N, True)))

        #   Solving for Baldwin
        elif self.problem == 4:
            # Create random target
            random_target = ''.join(random.choices(Baldwin.Alphabet, k=20))
            Baldwin.TARGET = list(random_target)

            for i in range(1000):
                rand_solution = Baldwin.BaldwinEffectProblem.initialize_citizen()
                self.genomes.append(Genome(Baldwin.BaldwinEffectProblem(rand_solution=rand_solution), fitness=1))
                self.buffer.append(Genome(gene=Baldwin.BaldwinEffectProblem(), fitness=1))

        #   Solving for genetic programming
        elif self.problem == 6:
            GA_POPSIZE = 8000
            half_pop = int(GA_POPSIZE / 2)
            # Ramp half and half
            for i in range(half_pop):
                grow_tree = GP_tree()
                full_tree = GP_tree()
                GP_tree.generate_tree(current_node=grow_tree.root, GROW=True, current_depth=0)
                GP_tree.generate_tree(current_node=full_tree.root, GROW=False, current_depth=0)
                self.genomes.append(Genome(gene=grow_tree))
                self.genomes.append(Genome(gene=full_tree))
                self.buffer.append(Genome(gene=[]))
                self.buffer.append(Genome(gene=[]))

    def calc_fitness(self):
        for i in range(GA_POPSIZE):
            temp_fitness = self.genomes[i].calc_individual_fitness()
            self.genomes[i].fitness = temp_fitness

    def sort_by_fitness(self):
        # If its maximize problem
        if self.problem == 1 or self.problem == 3 or self.problem == 4 or self.problem == 6:
            self.genomes.sort(reverse=True)  # Sort population using its fitness
        else:
            self.genomes.sort()  # Sort population using its fitness

    def elitism(self, esize):
        for i in range(esize):
            self.buffer[i] = deepcopy(self.genomes[i])
            self.buffer[i].age += 1  # Kept for another generation

    def mate(self, cross_method=1, selection_method=0):
        global GA_ELITRATE
        esize = int(GA_POPSIZE * GA_ELITRATE)

        if self.problem != 6:
            # Taking the best citizens
            self.elitism(esize)
        else:
            esize = 0

        # If we choose parent selection then we will choose parents as the number of psize-esize
        if selection_method == 1:
            num_of_parents = GA_POPSIZE - esize
            # Parents selection method - SUS
            self.genomes[esize:] = self.SUS(num_of_parents=num_of_parents)

        elif selection_method == 2:
            num_of_parents = GA_POPSIZE - esize
            self.genomes[esize:] = self.tournament_selection(num_of_parents=num_of_parents, esize=esize)

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

        elif cross_method == 6:
            self.trees_crossover(esize=esize)

    def get_best_fitness(self):
        i = 0
        size = len(self.genomes)
        if self.problem == 3:
            while True:
                empty_bins = 0
                flag = True
                sum_of_bins = self.genomes[i].gene.create_sum_of_bins()

                for _bin in sum_of_bins:
                    if _bin == 0:
                        empty_bins += 1
                    if _bin > bin_packing.C:
                        flag = False
                        break

                if flag is True:
                    print(f"Number of empty bins for best individual is: {empty_bins}")
                    return self.genomes[i]
                elif i == size - 1:
                    return None
                else:
                    i += 1

        return self.genomes[i]

    def print_best(self):
        best_inv = self.get_best_fitness()
        print('Best: ', best_inv.gene, '(', best_inv.fitness, ')')

        return best_inv

    def calc_avg_std(self):
        """ 1.1. Calculate and report avg fitness value of each generation in population
            + Calculate the std from mean"""
        arr_size = len(self.genomes)
        fitness_array = np.empty(arr_size)

        for idx, citizen in enumerate(self.genomes):
            fitness_array[idx] = citizen.fitness

        avg_fitness = fitness_array.mean()
        std_fitness = fitness_array.std()

        if self.problem == 4:
            true_array = np.empty(arr_size)
            false_array = np.empty(arr_size)
            learned_array = np.empty(arr_size)

            for idx, citizen in enumerate(self.genomes):
                true_array[idx] = (citizen.gene.true_count / 20)
                false_array[idx] = (citizen.gene.false_count / 20)
                learned_array[idx] = citizen.gene.learn

            avg_learned = learned_array.mean()
            avg_true = true_array.mean()
            avg_false = false_array.mean()
            print(avg_true, avg_false, avg_learned)

            return avg_fitness, std_fitness, avg_true, avg_false, avg_learned

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
            if random.random() < self.mutation_rate:
                if self.problem == 2:
                    Genome.mutate(self.buffer[i], self.num_of_mutations)
                else:
                    Genome.swap_mutation(self.buffer[i], self.num_of_mutations)

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

            if random.random() < self.mutation_rate:
                if self.problem == 4:
                    Genome.bit_flip(self.buffer[i])
                else:
                    Genome.scramble_mutation(self.buffer[i], self.num_of_mutations)

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

            if random.random() < self.mutation_rate:
                self.buffer[i].mutate(self.num_of_mutations)

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

                obj = obj_2[:spos_1] + obj_1 + obj_2[spos_1:]

                # Insert missing values
                if len(obj) < tsize:
                    for j in range(tsize):
                        if j not in obj:
                            obj.append(j)
                        if len(obj) == tsize:
                            break

            self.buffer[i].gene.set_obj(obj=obj)

            if random.random() < self.mutation_rate:
                self.buffer[i].swap_mutation(self.num_of_mutations)

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

            if random.random() < self.mutation_rate:
                Genome.scramble_mutation(self.buffer[j], self.num_of_mutations)
            if random.random() < self.mutation_rate:
                Genome.scramble_mutation(self.buffer[j + 1], self.num_of_mutations)

    def trees_crossover(self, esize):
        for i in range(esize, GA_POPSIZE):
            # picking 2 genomes
            i1 = randint(0, int(GA_POPSIZE / 2) - 2)
            i2 = randint(i1 + 1, int(GA_POPSIZE / 2) - 1)

            # Recombination vs mutation
            if random.random() > 0.05:
                self.genomes[i1].gene.reservoir_sampling(pr=0.9)
                self.genomes[i2].gene.reservoir_sampling(pr=0.9)
                child = deepcopy(self.genomes[i1].gene)
                child.branch_swap(self.genomes[i2].gene.reservoir)

                child_fitness = child.get_fitness()
                if child_fitness < self.genomes[i1].fitness:
                    self.buffer[i].gene = child
                else:
                    if random.random() > 0.5:
                        self.buffer[i].gene = child
                    else:
                        self.buffer[i] = self.genomes[i1]
            else:
                child = deepcopy(self.genomes[i1])
                child.gene.reservoir_sampling(pr=0.5)
                child.branch_mutation()
                self.buffer[i] = child

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

    def over_selection(self, esize):
        pop_size = GA_POPSIZE-esize+1
        """ Implementation of over selection method """
        x = int((pop_size*4)/100)     # Taking 4% of the population
        first_range = int((pop_size*80)/100)
        choose_from = self.genomes[0:x]
        new_population = random.choices(choose_from, k=first_range)
        choose_from = self.genomes[x+1:]
        new_population.extend(random.choices(choose_from, k=pop_size-first_range))
        self.genomes[esize:] = new_population

    def tournament_selection(self, num_of_parents, esize):
        """ choosing k individuals for a tournament, winner pass for crossover """
        selected = []
        k = 6
        pr = 0.2

        maximize = False

        if self.problem == 6 or self.problem == 4 or self.problem == 3 or self.problem == 1:
            maximize = True

        while len(selected) < num_of_parents:
            if maximize:
                best_fitness = -np.inf
            else:
                best_fitness = np.inf
            best_inv = None

            # Choose k random indexes from the population
            inv_to_check = set(np.random.randint(GA_POPSIZE, size=k))
            if random.random() > pr * (1 - pr) ** len(selected):
                # Getting the best gene
                for inv in inv_to_check:
                    if maximize:
                        if self.genomes[inv].fitness > best_fitness:
                            best_fitness = self.genomes[inv].fitness
                            best_inv = self.genomes[inv]
                    else:
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
        alpha = 1
        sigma_share = 4.5
        tsize = self.genomes[0].gene.N

        sharing_matrix = np.zeros((GA_POPSIZE, GA_POPSIZE))
        distance_matrix = []
        for i in range(0, GA_POPSIZE):
            for j in range(i + 1, GA_POPSIZE):
                distance = self.genomes[i].gene.calc_distance(self.genomes[j].gene) / tsize
                distance_matrix.append(distance)
                sharing_matrix[i, j] = self.sharing_function(distance, sigma_share=sigma_share, alpha=alpha)
                sharing_matrix[j, i] = sharing_matrix[i, j]

        if self.problem == 1 or self.problem == 3 or self.problem == 4:
            for i in range(0, tsize):
                scaling_sum = sum(sharing_matrix[i])
                if scaling_sum == 0: scaling_sum = 1
                self.genomes[i].fitness /= scaling_sum
        else:
            for i in range(0, tsize):
                scaling_sum = sum(sharing_matrix[i])
                if scaling_sum == 0: scaling_sum = 1
                self.genomes[i].fitness *= scaling_sum

    def hyper_mutation(self, new_rate, new_num_of_mutations):
        """ Changing the mutation rate due to the quality of solutions """
        self.num_of_mutations = int(new_num_of_mutations)
        self.mutation_rate = new_rate

    def make_species(self, adjust_threshold=True):
        """ Make species_list """

        self.species_list = [[0]]
        for i in range(1, GA_POPSIZE):
            selected_species, flag = self.find_species(i)
            if flag is False and i != 0:  # If we didnt find matching species we will add one
                self.species_list.append([])  # initialize a new species

            # Adding the citizen to the matching species
            self.species_list[selected_species].append(i)

        safe_range = 4
        # Adjust the threshold:
        if adjust_threshold and abs(len(self.species_list) - self.optimal_species) > safe_range:
            self.adjust_speciation_threshold()

        return self.species_list

    def find_species(self, i):
        """searching for matching species from species_list"""
        species_counter = 0
        for j, species in enumerate(self.species_list):
            distance = self.genomes[species[0]].gene.calc_distance(self.genomes[i].gene) / self.genomes[i].gene.N

            if distance < self.speciation_threshold:
                self.genomes[i].species = j
                return j, True

            species_counter += 1

        self.genomes[i].species = species_counter
        return species_counter, False

    def adjust_speciation_threshold(self, max_iterations=3):
        """Adjust the number of species to be around max_species"""
        iterations = 0

        while len(
                self.species_list) != self.optimal_species and self.speciation_threshold > 0 and iterations < max_iterations:
            iterations += 1
            self.speciation_threshold += 1 * np.sign(len(self.species_list) - self.optimal_species) / (iterations + 2)
            self.speciation_threshold = max(1, self.speciation_threshold)
            self.species_list = self.make_species(adjust_threshold=False)

    def calc_similarity(self):
        # We will sample 10% of the population to measure distances
        sampled_list = random.sample(range(GA_POPSIZE), round(GA_POPSIZE * 10 / 100))
        max_distance = 0
        distance = 0

        iterations = 0
        # Iterating over the sample list to compute distances
        for i in sampled_list:
            sampled_list.remove(i)
            for j in sampled_list:
                iterations += 1
                new_distance = self.genomes[i].gene.calc_distance(self.genomes[j].gene)
                distance += new_distance

                if new_distance > max_distance:
                    max_distance = new_distance

        if distance != 0:
            # Scaling the distance value to [0,2]
            distance /= iterations
            distance /= max_distance
            distance += 1

        return distance
