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


def is_dominates(p_1_1, p_1_2, p_2_1, p_2_2):
    if (p_1_1 < p_2_1 and p_1_2 < p_2_2) or (p_1_1 <= p_2_1 and p_1_2 < p_2_2) or (p_1_1 < p_2_1 and p_1_2 <= p_2_2):
        return True
    else:
        return False


def non_dominated_sorting(objective_1, objective_2):
    # Initialize lists
    domination_list = [[] for i in range(0, len(objective_1))]
    dominated_counter = [0] * len(objective_1)
    rank_list = [0] * len(objective_1)
    front = [[]]

    for p in range(0, len(objective_1)):
        for q in range(0, len(objective_1)):
            # Check if p dominates q
            if is_dominates(objective_1[p], objective_2[p], objective_1[q], objective_2[q]):
                if q not in domination_list[p]:
                    domination_list[p].append(q)

            # check if q dominates p
            elif is_dominates(objective_1[q], objective_2[q], objective_1[p], objective_2[p]):
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
                dominated_counter[q] -= 1
                if dominated_counter[q] == 0:
                    rank_list[q] = i + 1
                    if q not in temp_list:
                        temp_list.append(q)
        i = i + 1
        front.append(temp_list)

    del front[len(front) - 1]
    return front, rank_list


def tournament(ranks, distances):
    sample_space = list(range(len(ranks)))
    participants = random.sample(sample_space, 5)

    # Choosing first parent
    best_idx = None
    for participant in participants:
        if best_idx is None or ranks[participant] > ranks[best_idx] or\
                (ranks[participant] == ranks[best_idx] and distances[participant > distances[best_idx]]):
            best_idx = participant

    parent_1 = best_idx

    # Choosing parent_2
    participants = random.sample(sample_space, 5)

    # Choosing first parent
    best_idx = None
    for participant in participants:
        if best_idx is None or ranks[participant] > ranks[best_idx] or \
                (ranks[participant] == ranks[best_idx] and distances[participant > distances[best_idx]]):
            best_idx = participant

    parent_2 = best_idx

    return parent_1, parent_2


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

    if len(front) == 1:
        return [0]
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

    children = parent_1[:spos_1] + parent_2[spos_1:spos_2] + parent_1[spos_2:]

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


def NSGA_2_Solver(x_1=-50, x_2=50, vector_len=5, population_size=100):
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
        fronts, ranks = non_dominated_sorting(objective_1_list, objective_2_list)

        # Print list of best fronts for the current iteration
        print(f"Best Front for iteration number {iteration} is : \n")
        for point in fronts[0]:
            print(population[point], 3, end=" ")
        print("\n")

        # Calculate crowding distance
        crowding_distance_list = []
        for i in range(0, len(fronts)):
            crowding_distance_list.extend(
                crowding_distance(objective_1_list[:], objective_2_list[:], fronts[i][:]))

        print("Length of crowding distance is:", len(crowding_distance_list))
        temp_population = []
        # Generating offsprings
        while len(temp_population) != pop_size:
            # Running a tournament
            idx_1, idx_2 = tournament(ranks, crowding_distance_list)

            # Two point crossover + mutation
            temp_population.append(two_point_crossover(population[idx_1], population[idx_2]))

        population = temp_population

        print("Finish new_fronts")

        print("Finish iterating")
        iteration += 1

    plt.xlabel('Objective 1', fontsize=15)
    plt.ylabel('Objective 2', fontsize=15)
    plt.scatter(objective_1_list, objective_2_list)
    plt.show()
