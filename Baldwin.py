from Genetic import Genome
import random

# import utils

N = 20
GA_POPSIZE = 1000
Alphabet = ['', '1', '0']


class BaldwinEffectProblem:
    def __init__(self, data):
        self.data = data
        self.learning_steps = 1000
        self.target_object = ''.join(random.choices(Alphabet, k=N))

    @staticmethod
    def initialize_citizen():
        random_object = [None] * N

        # Creating random indices for each option
        idx_list = [range(N)]
        unknown_ind = random.choices(idx_list, k=N/2)   # Half indices for '?'
        idx_list.remove(unknown_ind)

        true_ind = random.choices(idx_list, k=N/4)      # Quarter of '1'
        idx_list.remove(unknown_ind)

        false_ind = random.choices(idx_list, k=N/4)     # Quarter of '0'

        # Assign the matching value to each indices
        for idx in unknown_ind:
            random_object[idx] = '?'

        for true_idx, false_idx in zip(true_ind, false_ind):
            random_object[true_idx] = '1'
            random_object[false_idx] = '0'

        # Creating the new citizen
        return Genome(random_object)

    def calc_fitness(self, population):
        for i in range(self.data.ga_popsize):
            tries_left = 0
            if self.is_solution_cadidate(population[i].str):
                for j in range(self.learning_steps):
                    updated_gene = self.update_string(population[i].str)
                    if updated_gene == self.target_object:
                        tries_left = self.learning_steps - j
                        break
            population[i].fitness = 1 + ((19 * tries_left) / 1000)

    def is_solution_cadidate(self, citizen_gene):
        for i in range(N):
            if citizen_gene[i] != '?' and citizen_gene[i] != self.target_object[i]:
                return False
        return True

    def update_string(self, citizen_gene):
        assert (N == len(citizen_gene))
        ques_marks = []
        for i, entry in enumerate(citizen_gene):
            if entry == '?':
                ques_marks.append(i)
        while len(ques_marks) > 0:
            entry = random.choice(ques_marks)
            ques_marks.remove(entry)
            bit = str(random.randint(0, 1))
            string_list = list(citizen_gene)
            string_list[entry] = bit
            citizen_gene = ''.join(string_list)
        return citizen_gene

    def print_best(self, gav, iter_num):
        print("Best: " + gav[0].str + " (" + str(gav[0].fitness) + ")")
        # with open("output.txt", 'a') as file:
        #     file.write("Best: " + gav[0].str + " (" + str(gav[0].fitness) + ")\n")
        #     file.write("    Iteration number: " + str(iter_num) + "\n")
        #     file.write("    Fitness average: " + str(np.mean(gav, self.data.ga_popsize), 3)) + "\n")
        #     file.write("    Fitness deviation: " + str(round(utils.deviation(gav, self.data.ga_popsize), 3)) + "\n")

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

    def is_done(self, best):
        return False
