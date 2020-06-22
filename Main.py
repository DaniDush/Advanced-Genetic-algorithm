from copy import deepcopy
from time import time, sleep
import Genetic
from Const import probs
from MinimalConflicts import minimal_conflicts
from BinPacking import bin_packing
import NSGA_2
import matplotlib.pyplot as plt
import GeneticProgramming
import threading
from Utils import print_nqueens_board, get_args

GA_MAXITER = 150
KS_MAXITER = 30
spec_thres_3 = None
GLOBAL_BEST = None
ABSOLUTE_TIME = [None]*7
threadLock = threading.Lock()
ISLANDS = []
NUM_OF_THREADS = 0


class myThread(threading.Thread):
    def __init__(self, threadID, name, N, question, problem):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.N = N
        self.question = question
        self.problem = problem
        self.island = None

    def run(self):
        run_genetic_algo(self.problem, self.N, self.question)

    def set_population(self, population):
        self.island = population

    def receive_migrants(self, migrants):
        while self.island is None:  # Waiting for island creation
            pass
        self.island.receive_migrants(migrants)


# TODO what we do if spos_1 == spos_2

def run_minimal_conflicts(N):
    print("Minimal conflict solution: \n")
    start_time = time()

    conflicts_solver = minimal_conflicts(size=N)
    solution = conflicts_solver.solve()

    print('Minimal conflict solving time: ', time() - start_time)
    print_nqueens_board(solution, N)


def run_genetic_algo(problem, N, question):
    global GLOBAL_BEST, ABSOLUTE_TIME
    pop_size = 1000
    current_island = threading.currentThread().getName()

    print(f"{current_island} Activated \n")
    ######################################################################
    # Initialize local optima vars
    similarity_threshold = 1
    std_threshold = 0.2
    local_optima_range = 5
    local_optima_matrix = [0] * local_optima_range
    generation_std = []
    generation_avg_fitness = []
    generation_similarity = []

    ######################################################################
    generation_true_count = []
    generation_false_count = []
    generation_learned = []
    generation_species = []
    # Initialize algorithm vars
    cross_method = 2
    selection_method = 2
    max_iter = GA_MAXITER

    # If its N Queens problem
    if problem == 0:
        pop_size = 2000
        if current_island == 'Island-5':
            pop_size = 800
            selection_method = 2
            cross_method = 5
        elif current_island == 'Island-4':
            pop_size = 1000
            selection_method = 0
            cross_method = 5
        elif current_island == 'Island-1':
            selection_method = 2
            cross_method = 5
        else:
            selection_method = 2
            cross_method = 4

    # If its Knap Sack problem
    elif problem == 1:
        spec_threshold = 0.5
        max_iter = KS_MAXITER
        OP = probs[N][3]

        if current_island == 'Island-1':
            pop_size = 500
            selection_method = 0
            cross_method = 1

        elif current_island == 'Island-2':
            pop_size = 500
            selection_method = 2
            cross_method = 2

        else:
            selection_method = 2
            cross_method = 2
            pop_size = 1000

    # If its String problem
    elif problem == 2:
        pop_size = 600

        if current_island == 'Island-1':
            pop_size = 1000
            selection_method = 0  # No selection
        elif current_island == 'Island-2':
            pop_size = 500
            selection_method = 1  # SUS
        else:
            selection_method = 2  # Tournament Selection

    # If its Bin packing problem
    elif problem == 3:
        cross_method = 4
        pop_size = 400

        if current_island == 'Island-5':
            pop_size = 200

        max_iter = 150
        std_threshold = 0.1
        spec_threshold = spec_thres_3

    # If its Baldwin effect
    elif problem == 4:
        max_iter = 50
        cross_method = 2
        selection_method = 2
        pop_size = 400

    # GP - XOR
    elif problem == 6:
        max_iter = 20
        cross_method = 6
        selection_method = 2
        pop_size = 3000

    # GP - MATH
    elif problem == 7:
        max_iter = 30000  # Will terminate before max_iter
        cross_method = 6
        selection_method = 2
        pop_size = 1000

    ######################################################################
    start_time = time()

    if problem == 3 or problem == 1:
        current_population = Genetic.Population(problem=problem, pop_size=pop_size, speciation_threshold=spec_threshold)
    else:
        current_population = Genetic.Population(problem=problem, pop_size=pop_size)

    threading.currentThread().set_population(current_population)
    current_population.init_population(N=N)

    for i in range(max_iter):

        generation_start_time = time()
        current_population.calc_fitness()

        if problem == 3:
            # current_population.fitness_share()
            species = current_population.make_species()
            number_of_species = len(current_population.species_list)
            generation_species.append(number_of_species)
            # print(f"Current number of species: {number_of_species} ")

        current_population.sort_by_fitness()
        if NUM_OF_THREADS > 1:  # If we have more then 1 island we will perform migration
            current_population.spread_migrants()
            current_population.perform_migration()

        best_inv = current_population.get_best()

        ################ Synchronize Area ################
        # Setting global best
        threadLock.acquire()

        if GLOBAL_BEST is None:
            GLOBAL_BEST = best_inv
        else:
            # If its maximize problem
            if problem == 1 or problem == 4 or problem == 6:
                if best_inv.fitness > GLOBAL_BEST.fitness:
                    GLOBAL_BEST = best_inv
                    print('Best: ', GLOBAL_BEST.gene, '(', GLOBAL_BEST.fitness, ')')

            elif problem == 3 and best_inv is not None:  # Check by number of valid empty bins
                if best_inv.gene.empty_bins > GLOBAL_BEST.gene.empty_bins:
                    print(best_inv.gene.empty_bins, GLOBAL_BEST.gene.empty_bins)
                    GLOBAL_BEST = deepcopy(best_inv)
                    print('Best: ', GLOBAL_BEST.gene, '(', GLOBAL_BEST.fitness, ')')
                    print('New number of empty bins: ', GLOBAL_BEST.gene.empty_bins)

            else:
                if best_inv.fitness < GLOBAL_BEST.fitness:
                    GLOBAL_BEST = best_inv
                    print('Best: ', GLOBAL_BEST.gene, '(', GLOBAL_BEST.fitness, ')')

        threadLock.release()
        ################ Synchronize Area ################

        if problem != 4:
            avg_fitness, std = current_population.calc_avg_std()
        else:
            avg_fitness, std, true_counter, false_counter, avg_learned = current_population.calc_avg_std()
            generation_true_count.append(true_counter)
            generation_false_count.append(false_counter)
            generation_learned.append(avg_learned)

        generation_avg_fitness.append(avg_fitness)
        generation_std.append(std)

        if problem == 1:
            ####################################################################
            # Checking if were converging to local optima
            if i > local_optima_range:  # If were after 10 generation we will start checking
                std_signal = 0
                similarity_signal = 0

                # Checking std sign for local optima
                if generation_std[-1] < std_threshold and generation_std[-1] < generation_std[-2] < generation_std[-3]:
                    std_signal = 1

                # Checking similarity sign for local optima
                distance = current_population.calc_similarity()
                generation_similarity.append(distance)
                if distance <= similarity_threshold:
                    similarity_signal = 1

                if similarity_signal == 1 or std_signal == 1:
                    print("Were on local Optima")
                    local_optima_matrix.append(1)

                    if i > local_optima_range + 1:
                        if local_optima_matrix[i - 1 - local_optima_range] == 1:
                            current_population.hyper_mutation(0.5, 3)
                            if local_optima_matrix[i - 1 - local_optima_range] == 1 and local_optima_matrix[
                                i - 2 - local_optima_range] == 1:
                                new_rate = max(5, current_population.genomes[0].gene.N / 20)
                                print("Strong mutation")
                                current_population.hyper_mutation(0.7, new_rate)

                else:
                    local_optima_matrix.append(0)
                    current_population.hyper_mutation(0.25, 1)

            #####################################################################
        if problem == 1:
            if OP == GLOBAL_BEST.gene.sack:
                print(f'{current_island} terminated')
                break

        if problem == 7:
            if GeneticProgramming.IS_TERMINATE:
                print(f'{current_island} terminated')
                break

        if GLOBAL_BEST.fitness is 0 and (problem == 0 or problem == 2):
            print(f'{current_island} terminated')
            break

        current_population.mate(cross_method=cross_method, selection_method=selection_method)
        current_population.swap()

        print(f'For thread {current_island}:\nGeneration running time for iteration {i}: ',
              time() - generation_start_time)

    print('Absolute running time: ', time() - start_time, '\n')

    # elif problem == 3:
    #     plt.plot(list(range(max_iter)), generation_species, color='blue', label='Number of species')
    #     plt.show()

    if problem == 4:
        xs = list(range(max_iter))
        plt.figure(1)
        plt.plot(xs, generation_true_count, color='blue', label='True positions')
        plt.figure(2)
        plt.plot(xs, generation_false_count, color='green', label='False positions')
        plt.figure(3)
        plt.plot(xs, generation_learned, color='black', label='False positions')
        plt.show()

    # if problem == 6:
    #     xs = list(range(max_iter))
    #     plt.figure(1)
    #     plt.plot(xs, num_of_operators, color='blue', label='Change in the number of operators for best citizen')
    #     plt.xlabel("Number of generation")
    #     plt.ylabel("Number of operators")
    #     plt.show()


def multi_threading_ga(problem, N, question, num_threads=5):
    global ISLANDS
    # running_times = []
    # speed_up = []
    # efficiency = []

    ISLANDS = []
    # If its Bin packing problem
    if problem == 3:
        global spec_thres_3
        N, C, WEIGHTS, spec_threshold = get_args(question)
        bin_packing.W = WEIGHTS
        bin_packing.C = C
        spec_thres_3 = spec_threshold

    # GP - XOR
    elif problem == 6:
        GeneticProgramming.OPERATORS = GeneticProgramming.XOR_OPERATORS
        GeneticProgramming.OPERANDS = GeneticProgramming.XOR_OPERANDS

    # GP - MATH
    elif problem == 7:
        GeneticProgramming.OPERATORS = GeneticProgramming.MATH_OPERATORS
        GeneticProgramming.OPERANDS = GeneticProgramming.MATH_OPERANDS
        GeneticProgramming.init_math_args()
        GeneticProgramming.PROBLEM = 'M'

    # Create new threads and add to islands
    for i in range(NUM_OF_THREADS):
        thread = myThread(1, f"Island-{i + 1}", N, question, problem)
        ISLANDS.append(thread)

    Genetic.ISLANDS = ISLANDS
    # Start new Threads
    for island in ISLANDS:
        island.start()

    # Waiting for threads to terminate
    for island in ISLANDS:
        island.join()

    if problem == 3:
        print(f"Number of Empty bins: {GLOBAL_BEST.gene.empty_bins}")

    if problem == 7:
        print(f"Number of hits: {GLOBAL_BEST.gene.hits}")

    print("Fitness", GLOBAL_BEST.fitness)

    # import numpy as np
    # running_times = ABSOLUTE_TIME
    # for i in range(7):
    #     speed_up.append(np.true_divide(running_times[0], running_times[i]))
    # print(running_times)
    # for i in range(7):
    #     efficiency.append(np.true_divide(running_times[0], (i+1)*running_times[i]))

    # plt.figure(2)
    # plt.plot(xs, speed_up, color='blue', label='T1/Tp')
    # plt.xlabel("Number of threads")
    # plt.ylabel("Speed Up")
    #
    # plt.figure(3)
    # plt.plot(xs, efficiency, color='blue', label='T1/Tp')
    # plt.xlabel("Number of threads")
    # plt.ylabel("Efficiency")
    # plt.show()


def main():
    global NUM_OF_THREADS
    inp = None
    question = None

    while True:
        num_threads = int(input("Choose number of threads (>= 1) : "))
        if num_threads >= 1:
            NUM_OF_THREADS = num_threads
            break
        print("Invalid number, try again")

    problem = int(input("Insert 0 for N Queens, 1 for Knap sack, 2 for String problem, 3 for Bin packing problem,"
                        "4 for Baldwin effect simulation, 5 for NSGA-2, 6 for XOR using GP, 7 for Univariate using "
                        "GP:"))
    if problem == 0:
        inp = int(input("Choose N: "))
    elif problem == 1:
        while True:
            inp = int(input("Choose Problem number from dataset (0-7): "))
            if 0 <= inp <= 7:
                break
            print("Invalid number, try again")
    elif problem == 3:
        while True:
            question = int(input("Choose Problem number from dataset (1-4): "))
            if 1 <= question <= 4:
                break
            print("Invalid number, try again")

    if problem == 5:
        NSGA_2.NSGA_2_Solver()

    else:
        multi_threading_ga(problem=problem, N=inp, question=question, num_threads=num_threads)

    # run_minimal_conflicts(N=N)

    sleep(14)


if __name__ == '__main__':
    main()
