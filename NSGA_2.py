import math
import random
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

MAX_ITER = 100
x1 = 0
x2 = 0
n = 0


# First function to optimize
def function_1(x_vector):
    y = 0

    for x in x_vector:
        y -= x ** 2

    return y


# Second function to optimize
def function_2(x_vector):
    y = 0

    for x in x_vector:
        y -= (x - 2) ** 2

    return y


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


def non_dominated_sorting(objective_1, objective_2):
    # Initialize lists
    domination_list = [[] for i in range(0, len(objective_1))]
    dominated_counter = [0] * len(objective_1)
    rank_list = [0] * len(objective_1)
    front = [[]]

    for p in range(0, len(objective_1)):
        for q in range(0, len(objective_1)):
            # Check if p dominates q
            if (objective_1[p] < objective_1[q] and objective_2[p] < objective_2[q]) or (
                    objective_1[p] <= objective_1[q] and objective_2[p] < objective_2[q]) or (
                    objective_1[p] < objective_1[q] and objective_2[p] <= objective_2[q]):
                if q not in domination_list[p]:
                    domination_list[p].append(q)

            # check if q dominates p
            elif (objective_1[q] < objective_1[p] and objective_2[q] < objective_2[p]) or (
                    objective_1[q] <= objective_1[p] and objective_2[q] < objective_2[p]) or (
                    objective_1[q] < objective_1[p] and objective_2[q] <= objective_2[p]):
                dominated_counter[p] = dominated_counter[p] + 1
        if dominated_counter[p] == 0:
            rank_list[p] = 0

            if p not in front[0]:
                front[0].append(p)

    # Find rest of fronts
    i = 0
    while front[i]:
        temp_list = []
        for p in front[i]:
            for q in domination_list[p]:
                dominated_counter[q] = dominated_counter[q] - 1
                if dominated_counter[q] == 0:
                    rank_list[q] = i + 1
                    if q not in temp_list:
                        temp_list.append(q)
        i = i + 1
        front.append(temp_list)

    del front[len(front) - 1]
    return front


# Function to calculate crowding distance
def crowding_distance(objective_1, objective_2, front):
    len_front = len(front)
    epsilon = 1e-20  # Epsilon for division in 0
    distance = [0] * len_front

    # Sorting the fronts by objectives
    sorted_front_1 = sort_by_values(front, objective_1[:])
    sorted_front_2 = sort_by_values(front, objective_2[:])

    distance[0] = np.inf
    distance[len(front) - 1] = np.inf

    norm_1 = (max(objective_1) - min(objective_1))
    norm_2 = (max(objective_2) - min(objective_2))

    # Handling division by 0
    if norm_1 == 0:
        norm_1 = epsilon
    if norm_2 == 0:
        norm_2 = epsilon

    # Calculating the distance
    for k in range(1, len_front - 1):
        # for objective 1
        distance[k] += (objective_1[sorted_front_1[k + 1]] - objective_1[sorted_front_1[k - 1]]) / norm_1
        # for objective 2
        distance[k] += (objective_2[sorted_front_2[k + 1]] - objective_2[sorted_front_2[k - 1]]) / norm_2

    return distance


# Function to carry out the crossover
def two_point_crossover(parent_1, parent_2):
    spos_1 = random.randint(0, n - 1)
    spos_2 = random.randint(0, n)

    while spos_2 <= spos_1:
        spos_2 = random.randint(0, n)

    children = parent_1[:spos_1] + parent_2[spos_1:spos_2] + parent_2[spos_2:]

    if random.random() < 0.35:
        return mutation(children)
    else:
        return children


def index_of(a, list_to_serach):
    for i in range(0, len(list_to_serach)):
        if list_to_serach[i] == a:
            return i
    return -1


# Function to carry out the mutation operator
def mutation(children):
    rand_idx = random.randint(0, n)
    children[rand_idx] = x1 + (x2 - x1) * random.random()
    return children


def NSGA_2_Solver(x_1=-50, x_2=50, vector_len=5, population_size=10):
    global x1, x2, n

    # Main program starts here
    pop_size = population_size

    # Initialization
    x1 = x_1
    x2 = x_2
    n = vector_len

    max_iter = MAX_ITER
    population = []

    print("Solving NSGA-2")

    # Initialize population
    for citizen in range(pop_size):
        vector = [x1 + (x2 - x1) * random.random() for i in range(0, n + 1)]
        population.append(vector)

    iteration = 0
    while iteration < max_iter:
        objective_1_list = []
        objective_2_list = []

        # Calculate the objectives values
        for i in range(pop_size):
            objective_1_list.append(function_1(population[i]))
            objective_2_list.append(function_2(population[i]))

        # Calculate min-pareto front using non dominated sotring
        NSGA_solutions = non_dominated_sorting(objective_1_list, objective_2_list)

        # Print list of best fronts for the current iteration
        print(f"Front for iteration number {iteration} is : \n")
        for point in NSGA_solutions[0]:
            print(population[point], 3, end=" ")

        print("\n")

        # Calculate crowding distance
        crowding_distance_list = []
        for i in range(0, len(NSGA_solutions)):
            crowding_distance_list.append(
                crowding_distance(objective_1_list[:], objective_2_list[:], NSGA_solutions[i][:]))

        temp_population = population[:]
        # Generating offsprings

        while len(temp_population) != pop_size * 2:
            # Choosing 2 random parents for crossover
            spos_1 = random.randint(0, pop_size - 1)
            spos_2 = random.randint(0, pop_size - 1)

            while spos_1 == spos_2:
                spos_2 = random.randint(0, pop_size - 1)

            # Two point crossover + mutation
            temp_population.append(two_point_crossover(population[spos_1], population[spos_2]))

        # Calculate objectives values for the new population
        function1_values2 = [function_1(temp_population[i]) for i in range(0, 2 * pop_size)]
        function2_values2 = [function_2(temp_population[i]) for i in range(0, 2 * pop_size)]
        non_dominated_sorted_solution2 = non_dominated_sorting(function1_values2, function2_values2)

        # Calculate crowding distance for the new population
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))

        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                range(0, len(non_dominated_sorted_solution2[i]))]

            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])

            front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                     range(len(non_dominated_sorted_solution2[i]))]

            for value in front:
                new_solution.append(value)

                if len(new_solution) == pop_size:
                    break

            population = [temp_population[i] for i in new_solution]
            iteration += 1

    plt.xlabel('Objective 1', fontsize=15)
    plt.ylabel('Objective 2', fontsize=15)
    plt.scatter(objective_1_list, objective_2_list)
    plt.show()
