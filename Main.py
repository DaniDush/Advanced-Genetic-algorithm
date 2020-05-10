from pathlib import Path
from time import time
import GA

GA_MAXITER = 16384


def get_args():
    problem_args = []
    path = Path('Problems', 'Falkenauer_u120_00.txt')
    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        problem_args.append(int(lines[i][:-1]))

    N = int(problem_args[0])
    C = int(problem_args[1])
    WEIGHTS = problem_args[2:]

    return N, C, WEIGHTS


def run_genetic_algo():
    print("Genetic Algorithm solution: \n")

    selection_method = 3
    cross_method = 2

    N, C, WEIGHTS = get_args()

    best_citizen = []

    start_time = time()
    current_population = GA.Population()
    current_population.init_population(N, C, WEIGHTS)

    for i in range(GA_MAXITER):
        generation_start_time = time()
        current_population.calc_fitness()
        current_population.sort_by_fitness()
        current_population.print_best()
        current_population.calc_avg_std()

        if current_population.get_best_fitness() is 0:
            print('Generation running time: ', time() - generation_start_time)
            break

        current_population.mate(cross_method=cross_method, selection_method=selection_method)
        current_population.swap()
        print('Generation running time: ', time() - generation_start_time)

    print('Absolute running time: ', time() - start_time)


def main():
    run_genetic_algo()


if __name__ == '__main__':
    main()
