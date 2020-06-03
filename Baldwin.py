import random
from copy import deepcopy

N = 20
GA_POPSIZE = 1000
Alphabet = ['1', '0']
local_iterations = 1000
TARGET = []


class BaldwinEffectProblem:
    def __init__(self, rand_solution=None):
        self.solution = rand_solution
        self.true_count = 0
        self.false_count = 0
        self.unused = 0
        self.learn = 0

    @staticmethod
    def initialize_citizen():
        random_object = [None] * N

        # Creating random indices for each option by shuffle indices list
        idx_list = list(range(N))
        random.shuffle(idx_list)

        unknown_idx = idx_list[:int(N/2)]
        true_idx = idx_list[int(N/2):int(N/2)+int(N/4)]
        false_idx = idx_list[int(N/2)+int(N/4):]

        for idx in unknown_idx:
            random_object[idx] = '?'

        for true, false in zip(true_idx, false_idx):
            random_object[true] = TARGET[true]
            if TARGET[false] == '0':
                random_object[false] = '1'
            else:
                random_object[false] = '0'

        # Creating the new citizen
        return random_object

    def get_fitness(self):
        self.true_count = 0
        self.false_count = 0
        # Counting the false and right positions
        for i, char in enumerate(self.solution):
            if (char == '1' and TARGET[i] == '1') or (char == '0' and TARGET[i] == '0'):
                self.true_count += 1
            elif (char == '1' and TARGET[i] == '0') or (char == '0' and TARGET[i] == '1'):
                self.false_count += 1

        if self.true_count - self.false_count > 2:
            self.learn = 1
            # Running local search to adapt the fitness value
            unused_tries = 1000
            if self.solution != TARGET:
                question_idx = [i for i, j in enumerate(self.solution) if j == '?']
                new_object = deepcopy(self.solution)

                for j in range(local_iterations):
                    unused_tries -= 1
                    temp_gene = self.change_solution(question_idx, new_object)
                    # If its a match
                    if temp_gene == TARGET:
                        self.unused = unused_tries
                        return 1 + (19 * (unused_tries / 1000))

                # if we didnt find solution in the local search
                self.unused = 0
                if self.true_count-self.false_count > 1:
                    return self.true_count-self.false_count
                else:
                    return 1
            else:
                return 20
        else:
            return 1

    def change_solution(self, question_idx, new_object):
        """ Creating new solution for the local search, we will replace '?' with '0' or '1' """
        # First we will find indices of all '?' in our random object
        for i in question_idx:
            # Choosing bit from ['0','1']
            bit = random.choice(Alphabet)
            new_object[i] = bit

        return new_object

    def shuffle(self, start, end):
        shuff = self.solution[start:end]
        random.shuffle(shuff)
        self.solution[start:end] = shuff

    def set_obj(self, obj):
        self.solution = obj

    def __getitem__(self, item):
        return self.solution[item]

    def __setitem__(self, key, value):
        self.solution[key] = value

    def __len__(self):
        return len(self.solution)

    def __repr__(self):
        return str(self.solution)

    def __str__(self):
        return str(self.solution)