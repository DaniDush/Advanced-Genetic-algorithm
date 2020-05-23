import random
from copy import deepcopy

N = 20
POPSIZE = 1000
Alphabet = ['1', '0']
TARGET = []


class baldwinEffect:
    def __init__(self, random_object):
        self.random_object = random_object
        self.learning_iterations = 1000
        self.N = N

    def get_fitness(self):
        """ Calc the fitness using the given equation """
        unused_iterations = 1000

        # Local search
        if self.random_object != TARGET:  # If the current object is not TARGET
            for i in range(self.learning_iterations):
                if self.random_object != TARGET:  # If the current object is not TARGET
                    unused_iterations -= 1     # Decrease unused iterations counter
                    # Convert the object to a new one
                    new_object = self.change_object()
                    # If new object is TARGET we will stop
                    if new_object == TARGET:
                        break

        fitness = 1 + (19 * (unused_iterations / 1000))

        if fitness != 1:
            print(fitness)

        return fitness

    def change_object(self):
        """ Changing the object for local search """

        # First we will find indices of all '?' in our random object
        question_idx = [i for i, j in enumerate(self.random_object) if j == '?']
        new_object = deepcopy(self.random_object)

        for idx in question_idx:
            # Choosing random bit
            bit = str(random.randint(0, 1))
            # Assign the new random bit instead of '?'
            new_object[idx] = bit

        return new_object

    def set_obj(self, obj):
        self.random_object = obj

    def shuffle(self, start, end):
        shuff = self.random_object[start:end]
        random.shuffle(shuff)
        self.random_object[start:end] = shuff

    def __str__(self):
        return str(self.random_object)

    def __getitem__(self, item):
        return self.random_object[item]

    def __len__(self):
        return N


def initialize_rand_object():
    random_object = ['-1'] * N

    # Creating random indices for each option
    idx_list = list(range(N))

    # Shuffle idx_list to generate random indices
    random.shuffle(idx_list)

    unknown_ind = idx_list[0:int(N / 2)]  # Half indices for '?'
    true_ind = idx_list[int(N / 2):int(N / 2) + int(N / 4)]  # Quarter will be wrong
    false_ind = idx_list[int(N / 2) + int(N / 4):]  # Quarter will be right

    # Assign the matching value to each indices
    for idx in unknown_ind:
        random_object[idx] = '?'

    for true_idx, false_idx in zip(true_ind, false_ind):
        alphabet_choices = ['0', '1']

        random_object[true_idx] = alphabet_choices[1]
        random_object[false_idx] = alphabet_choices[0]

    return random_object


