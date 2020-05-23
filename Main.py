from pathlib import Path
from time import time, sleep
import Genetic
from Const import probs
from MinimalConflicts import minimal_conflicts
from BinPacking import bin_packing
import NSGA_2

GA_MAXITER = 150
KS_MAXITER = 30


def get_args():
    problem_args = []
    path = Path('Problems', '1.txt')
    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        problem_args.append(int(lines[i][:-1]))

    N = int(problem_args[0])
    C = int(problem_args[1])
    WEIGHTS = problem_args[2:]

    return N, C, WEIGHTS


# TODO what we do if spos_1 == spos_2

def run_minimal_conflicts(N):
    print("Minimal conflict solution: \n")
    start_time = time()

    conflicts_solver = minimal_conflicts(size=N)
    solution = conflicts_solver.solve()

    print('Minimal conflict solving time: ', time() - start_time)
    print_nqueens_board(solution, N)


def run_genetic_algo(problem, N):
    print("Genetic Algorithm solution: \n")
    local_optima_range = 5
    generation_std = []
    generation_avg_fitness = []
    generation_similarity = []

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

    start_time = time()
    current_population = Genetic.Population(problem=problem)
    current_population.init_population(N=N)
    epsilon = 0.2
    for i in range(max_iter):
        generation_start_time = time()
        current_population.calc_fitness()
        current_population.sort_by_fitness()
        current_population.print_best()
        avg_fitness, std = current_population.calc_avg_std()
        generation_avg_fitness.append(avg_fitness)
        generation_std.append(std)

        # Checking if were converging to local optima
        if i > local_optima_range:  # If were after 10 generation we will start checking
            fitness_signal = 0
            std_signal = 0
            for j in range(local_optima_range):
                if generation_avg_fitness[-j - 2] - epsilon <= generation_avg_fitness[-j - 1] <= generation_avg_fitness[-j - 2] + epsilon:
                    fitness_signal += 1
                if generation_std[-j - 1] <= [-j - 2]:  # if std is converging to 0
                    std_signal += 1
            print(fitness_signal, std_signal)
            if fitness_signal + std_signal == 10:
                print("Were on local Optima")
                current_population.handle_local_optima()

        if problem == 1:
            if OP == current_population.get_best_fitness():
                print(f'Generation running time for iteration {i}: ', time() - generation_start_time)
                break

        if problem == 3:
            print("Number of empty bins", current_population.genomes[0].gene.get_empty_bins(), "\n")

        if current_population.get_best_fitness() is 0:
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

    elif problem == 3:
        bins_weight = [0]*N
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


""" Problem 0 = N Queens
    Problem 1 = Knap sack
    Problem 2 = String problem
    """


def main():
    NSGA_2.NSGA_2_Solver()
    return
    inp = None
    problem = int(input("Insert 0 for N Queens, 1 for Knap sack, 2 for String problem, 3 for Bin packing problem : "))
    if problem == 0:
        inp = int(input("Choose N: "))
    elif problem == 1:
        inp = int(input("Choose Problem number from dataset (0-7): "))

    run_genetic_algo(problem=problem, N=inp)

    # run_minimal_conflicts(N=N)

    sleep(14)


if __name__ == '__main__':
    main()
