import random
from copy import deepcopy

N = 20
GA_POPSIZE = 1000
Alphabet = ['1', '0']
local_iterations = 1000
target_object = ''.join(random.choices(Alphabet, k=N))


class BaldwinEffectProblem:
    def __init__(self, rand_solution=None):
        self.solution = rand_solution

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
            random_object[true] = '1'
            random_object[false] = '0'

        # Creating the new citizen
        return random_object

    def get_fitness(self):
        # Running local search to adapt the fitness value
        unused_tries = 1000
        if self.solution != target_object:
            unused_tries -= 1
            for j in range(local_iterations):
                temp_gene = self.change_solution()
                # If its a match
                if temp_gene == target_object:
                    return 1 + ((19 * unused_tries) / 1000)

            # if we didnt find solution in the local search
            return 1

    def change_solution(self):
        """ Creating new solution for the local search, we will replace '?' with '0' or '1' """
        new_solution = deepcopy(self.solution)
        # Find indices of question marks
        question_indices = [i for i, x in enumerate(self.solution) if x == '?']

        for i in question_indices:
            # Choosing bit from ['0','1']
            bit = random.choice(Alphabet)
            new_solution[i] = bit

        return new_solution

    def crossover(self, first_parent, second_parent):
        tsize = N
        spos = int(random.randint(0, 32767) % tsize)
        return Genome(first_parent.str[0:spos] + second_parent.str[spos:tsize])

    def mutate(self, citizen):
        tsize = N
        ipos = int(random.randint(0, 32767) % tsize)
        bit = random.randint(0, 2)
        if bit == 2:
            bit = '?'
        bit = str(bit)
        string_list = list(citizen.str)
        string_list[ipos] = bit
        citizen.str = ''.join(string_list)
