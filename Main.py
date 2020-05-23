from pathlib import Path
from time import time, sleep
import Genetic
from BinPacking import bin_packing
from Const import probs
from MinimalConflicts import minimal_conflicts
import numpy as np
import NSGA_2
import Baldwin

GA_MAXITER = 100
KS_MAXITER = 30
species_thres = [4.3, 6.3, 8.91, 12.7]


def get_args():
    problem_args = []
    path = Path('Problems', '4.txt')
    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        problem_args.append(int(lines[i][:-1]))

    N = int(problem_args[0])
    C = int(problem_args[1])
    WEIGHTS = problem_args[2:]

    return N, C, WEIGHTS


def run_minimal_conflicts(N):
    print("Minimal conflict solution: \n")
    start_time = time()

    conflicts_solver = minimal_conflicts(size=N)
    solution = conflicts_solver.solve()

    print('Minimal conflict solving time: ', time() - start_time)
    print_nqueens_board(solution, N)


def run_genetic_algo(problem, N):
    print("Genetic Algorithm solution: \n")
    ######################################################################
    # Initialize local optima vars
    similarity_threshold = 1.2
    std_threshold = 0.2
    local_optima_matrix = []
    local_optima_range = 5
    generation_std = []
    generation_avg_fitness = []
    generation_similarity = []

    ######################################################################
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
        max_iter = KS_MAXITER
        selection_method = 0
        cross_method = 2
        OP = probs[N][3]

    # If its Bin packing problem
    elif problem == 3:
        N, C, WEIGHTS = get_args()
        bin_packing.W = WEIGHTS
        bin_packing.C = C
        cross_method = 4

    # If its Baldwin effect
    elif problem == 4:
        cross_method = 2
        selection_method = 2

    ######################################################################
    start_time = time()
    current_population = Genetic.Population(problem=problem)
    current_population.init_population(N=N)

    for i in range(max_iter):
        generation_start_time = time()
        if i > 1 and problem != 4:
            current_population.calc_fitness(i)
        # current_population.fitness_share()
        species = current_population.make_species()
        print(len(current_population.species_list))
        current_population.sort_by_fitness()
        current_population.print_best()
        avg_fitness, std = current_population.calc_avg_std()
        generation_avg_fitness.append(avg_fitness)
        generation_std.append(std)

        #####################################################################
        # Checking if were converging to local optima
        if i > local_optima_range:  # If were after 10 generation we will start checking
            std_signal = False
            similarity_signal = False

            # Checking std sign for local optima
            if np.mean(generation_std[-local_optima_range:-1]) <= std_threshold:
                std_signal = True

            # Checking similarity sign for local optima
            distance = current_population.calc_similarity()
            generation_similarity.append(distance)
            if distance <= similarity_threshold:
                similarity_signal = True

            print(std_signal, similarity_signal)

            if similarity_signal or std_signal < std_threshold:
                print("Were on local Optima")
                local_optima_matrix.append(True)
                size = len(current_population.genomes[0].gene)
                current_population.hyper_mutation(0.7, round(size/10))
            else:
                local_optima_matrix.append(False)
            if i > local_optima_range + 1:

                if local_optima_matrix[i-1-local_optima_range] is True:
                    current_population.hyper_mutation(0.25, 1)
                    if local_optima_matrix[i-1-local_optima_range] is True and local_optima_matrix[i-2-local_optima_range] is True:
                        current_population.hyper_mutation(0.5, 3)

        ######################################################################

        if problem == 1:
            if OP == current_population.get_best_fitness():
                print(f'Generation running time for iteration {i}: ', time() - generation_start_time)
                break

        elif problem == 3:
            print("Number of empty bins", current_population.genomes[0].gene.get_empty_bins(), "\n")

        elif current_population.get_best_fitness() is 0:
            print("Fitness", current_population.get_best_fitness())
            print(f'Generation running time for iteration {i}: ', time() - generation_start_time)
            break

        current_population.mate(cross_method=cross_method, selection_method=selection_method)
        current_population.swap()
        print(f'Generation running time for iteration {i}: ', time() - generation_start_time)

    ######################################################################
    print('Absolute running time: ', time() - start_time)
    import matplotlib.pyplot as plt
    plt.plot(generation_std, marker='', color='olive', linewidth=2)
    plt.plot(generation_similarity, marker='', color='blue', linewidth=2)

    plt.show()

    # print final board if its N Queens
    if problem == 0:
        print_nqueens_board(current_population.genomes[0].gene, N)

    elif problem == 3:
        bins_weight = [0] * N
        for i, _bin in enumerate(current_population.genomes[0].gene.bins):
            bins_weight[_bin] += WEIGHTS[i]

        for i, weight in enumerate(bins_weight):
            print(f"The weight of bin {i} is: {weight}")


def print_nqueens_board(final_board, N):
    # print final board
    board = ["-"] * N ** 2
    for i in range(0, N):
        board[i * N + final_board[i]] = "Q"

    for row in range(0, N):
        print(board[row * N:(row + 1) * N])

    print("")


def main():
    NSGA_2.NSGA_2_Solver(-0.99, 0.99, 100)
    return
    inp = None
    problem = int(input("Insert 0 for N Queens, 1 for Knap sack, 2 for String problem, 3 for Bin packing problem,"
                        " 4 for Running Baldwin effect experiment  : "))
    if problem == 0:
        inp = int(input("Choose N: "))
    elif problem == 1:
        inp = int(input("Choose Problem number from dataset (0-7): "))

    run_genetic_algo(problem=problem, N=inp)

    # run_minimal_conflicts(N=N)

    sleep(14)


if __name__ == '__main__':
    main()
