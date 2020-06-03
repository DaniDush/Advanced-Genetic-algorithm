from pathlib import Path
import numpy as np
from time import time, sleep
import Genetic
from Const import probs
from MinimalConflicts import minimal_conflicts
from BinPacking import bin_packing
import NSGA_2
import matplotlib.pyplot as plt
import GeneticProgramming

GA_MAXITER = 150
KS_MAXITER = 30
species_thres = [4.3, 6.3, 8.91, 12.7]


def get_args(question):
    problem_args = []
    path = Path(f'{question}.txt')
    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        problem_args.append(int(lines[i][:-1]))

    N = int(problem_args[0])
    C = int(problem_args[1])
    WEIGHTS = problem_args[2:]

    return N, C, WEIGHTS, species_thres[question-1]


# TODO what we do if spos_1 == spos_2

def run_minimal_conflicts(N):
    print("Minimal conflict solution: \n")
    start_time = time()

    conflicts_solver = minimal_conflicts(size=N)
    solution = conflicts_solver.solve()

    print('Minimal conflict solving time: ', time() - start_time)
    print_nqueens_board(solution, N)


def run_genetic_algo(problem, N, question):
    print("Genetic Algorithm solution: \n")
    ######################################################################
    # Initialize local optima vars
    similarity_threshold = 1
    std_threshold = 0.2
    local_optima_range = 5
    local_optima_matrix = [0]*local_optima_range
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
        selection_method = 2
        cross_method = 4

    # If its Knap Sack problem
    elif problem == 1:
        spec_threshold = 1
        max_iter = KS_MAXITER
        selection_method = 0
        cross_method = 2
        OP = probs[N][3]

    # If its Bin packing problem
    elif problem == 3:
        std_threshold = 0.1
        N, C, WEIGHTS, spec_threshold = get_args(question)
        bin_packing.W = WEIGHTS
        bin_packing.C = C
        cross_method = 4

    # If its Baldwin effect
    elif problem == 4:
        max_iter = 50
        cross_method = 2
        selection_method = 2

    ######################################################################
    start_time = time()
    if problem == 3 or problem == 1:
        current_population = Genetic.Population(problem=problem, speciation_threshold=spec_threshold)
    else:
        current_population = Genetic.Population(problem=problem)

    current_population.init_population(N=N)

    for i in range(max_iter):
        generation_start_time = time()
        current_population.calc_fitness()

        # if problem != 4 and problem != 2:
        #     # current_population.fitness_share()
        #     species = current_population.make_species()
        #     number_of_species = len(current_population.species_list)
        #     generation_species.append(number_of_species)
        #     print(f"Current number of species: {number_of_species} ")

        current_population.sort_by_fitness()
        best_inv = current_population.print_best()

        if problem != 4:
            avg_fitness, std = current_population.calc_avg_std()
        else:
            avg_fitness, std, true_counter, false_counter, avg_learned = current_population.calc_avg_std()
            generation_true_count.append(true_counter)
            generation_false_count.append(false_counter)
            generation_learned.append(avg_learned)

        generation_avg_fitness.append(avg_fitness)
        generation_std.append(std)

        if problem != 4 and problem != 2:
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
                                new_rate = max(5, current_population.genomes[0].gene.N/20)
                                print("Strong mutation")
                                current_population.hyper_mutation(0.7, new_rate)

                else:
                    local_optima_matrix.append(0)
                    current_population.hyper_mutation(0.25, 1)

            #####################################################################
        if problem == 1:
            if OP == best_inv.gene.sack:
                print(f'Generation running time for iteration {i}: ', time() - generation_start_time)
                break

        if best_inv.fitness is 0:
            print("Fitness", current_population.get_best_fitness())
            print(f'Generation running time for iteration {i}: ', time() - generation_start_time)
            break

        current_population.mate(cross_method=cross_method, selection_method=selection_method)
        current_population.swap()
        print(f'Generation running time for iteration {i}: ', time() - generation_start_time)

    print('Absolute running time: ', time() - start_time)

    # print final board if its N Queens
    if problem == 0:
        print_nqueens_board(current_population.genomes[0].gene, N)

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


def print_nqueens_board(final_board, N):
    # print final board
    board = ["-"] * N ** 2
    for i in range(0, N):
        board[i * N + final_board[i]] = "Q"

    for row in range(0, N):
        print(board[row * N:(row + 1) * N])

    print("")


def main():
    GP = GeneticProgramming.GP_tree()
    GP.generate_tree()
    return
    inp = None
    question = None
    problem = int(input("Insert 0 for N Queens, 1 for Knap sack, 2 for String problem, 3 for Bin packing problem,"
                        " 4 for Baldwing effect simulation, 5 for NSGA-2: "))
    if problem == 0:
        inp = int(input("Choose N: "))
    elif problem == 1:
        inp = int(input("Choose Problem number from dataset (0-7): "))
    elif problem == 3:
        question = int(input("Choose Problem number from dataset (1-4): "))
        while question >= 5 or question <= 0:
            print("Problem number is not valid please choose again: ")
            question = int(input("Choose Problem number from dataset (1-4): "))

    if problem == 5:
        NSGA_2.NSGA_2_Solver()

    else:
        run_genetic_algo(problem=problem, N=inp, question=question)

    # run_minimal_conflicts(N=N)

    sleep(14)


if __name__ == '__main__':
    main()
