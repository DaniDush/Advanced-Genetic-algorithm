import math
import random
import matplotlib.pyplot as plt


class nsga_2:
    x_1 = 0
    x_2 = 0

    @staticmethod
    def function_1(x):
        """ implementation of function 1 """
        y = -(x ** 2)
        return y

    @staticmethod
    def function_2(x):
        """ implementation of function 2 """
        y = -((x - 10) ** 2)
        return y

    @staticmethod
    def crowding_distance(values1, values2, front):
        # Function to calculate crowding distance
        distance = [0 for i in range(0, len(front))]
        sorted1 = sort_by_values(front, values1[:])
        sorted2 = sort_by_values(front, values2[:])
        distance[0] = 4444444444444444
        distance[len(front) - 1] = 4444444444444444
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (
                    max(values1) - min(values1))
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (
                    max(values2) - min(values2))
        return distance

    @staticmethod
    # Function to carry out the crossover
    def crossover(a, b):
        r = random.random()
        if r > 0.5:
            return nsga_2.mutation((a + b) / 2)
        else:
            return nsga_2.mutation((a - b) / 2)

    @staticmethod
    # Function to carry out the mutation operator
    def mutation(solution):
        mutation_prob = random.random()
        if mutation_prob < 1:
            solution = nsga_2.x_1 + (nsga_2.x_2 - nsga_2.x_1) * random.random()
        return solution

    @staticmethod
    # Function to sort by values
    def sort_by_values(list1, values):
        sorted_list = []
        while len(sorted_list) != len(list1):
            if index_of(min(values), values) in list1:
                sorted_list.append(index_of(min(values), values))
            values[index_of(min(values), values)] = math.inf
        return sorted_list


# Function to find index of list
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Function to carry out NSGA-II's fast non dominated sort
def non_dominated_sorting(values1, values2, population_size):
    S = [[] for i in range(population_size)]
    n = [0] * population_size
    rank = [0] * population_size

    front = [[]]

    for p in range(population_size):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (
                    values1[p] >= values1[q] and values2[p] > values2[q]) or (
                    values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (
                    values1[q] >= values1[p] and values2[q] > values2[p]) or (
                    values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


def nsga_2_solver(x_1, x_2, population_size):
    max_iter = 5000
    solutions = []
    nsga_2.x_1 = x_1
    nsga_2.x_2 = x_2

    # First we will initialize the population
    for i in range(population_size):
        solutions.append(x_1 + (x_2 - x_1) * random.random())

    # Start iterating
    for i in range(max_iter):
        # Calculate functions values
        function_1_values = [nsga_2.function_1(solutions[i]) for i in range(0, population_size)]
        function_2_values = [nsga_2.function_2(solutions[i]) for i in range(0, population_size)]

        # Sorting using non dominated sort
        sorted_population = non_dominated_sorting(function_1_values, function_2_values, population_size)

        # Printing best pareto so far
        print(f"The best front for Generation number {i} is: \n")
        for valuez in sorted_population[0]:
            print(round(solutions[valuez], 3), end=" ")

        # Sorting using crowding distance
        crowding_distance_values = []
        for j in range(len(sorted_population)):
            crowding_distance_values.append(
                nsga_2.crowding_distance(function_1_values[:], function_2_values[:], sorted_population[i][:]))

        solutions_2 = solutions[:]

        # Generating offsprings
        while len(solutions_2) != 2 * population_size:
            i1 = random.randint(0, population_size - 1)
            i2 = random.randint(0, population_size - 1)
            solutions_2.append(nsga_2.crossover(solutions[i1], solutions[i2]))

        function1_values_2 = [nsga_2.function_1(solutions_2[i]) for i in range(0, 2 * population_size)]
        function2_values_2 = [nsga_2.function_2(solutions_2[i]) for i in range(0, 2 * population_size)]

        sortd_solutions_2 = non_dominated_sorting(function1_values_2[:], function2_values_2[:], population_size)

        crowding_distance_values2 = []
        for j in range(0, len(sortd_solutions_2)):
            crowding_distance_values2.append(
                nsga_2.crowding_distance(function1_values_2[:], function2_values_2[:], sortd_solutions_2[i][:]))
        new_solution = []
        for j in range(0, len(sortd_solutions_2)):
            non_dominated_sorted_solution2_1 = [
                index_of(sortd_solutions_2[i][j], sortd_solutions_2[i]) for j in
                range(0, len(sortd_solutions_2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [sortd_solutions_2[i][front22[j]] for j in
                     range(0, len(sortd_solutions_2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if len(new_solution) == population_size:
                    break
            if len(new_solution) == population_size:
                break
        solution = [solutions_2[j] for j in new_solution]

    # Lets plot the final front now
    function1 = [i * -1 for i in function_1_values]
    function2 = [j * -1 for j in function_2_values]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()
