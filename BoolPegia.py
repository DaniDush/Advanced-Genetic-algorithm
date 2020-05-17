from random import randint, shuffle
import numpy as np


class bool_pgia:
    def __init__(self, target, tsize=None):
        if tsize is None:
            self.string = []
        else:
            new_string = "".join([chr(randint(32, 122)) for x in range(tsize)])
            self.string = [char for char in new_string]

        self.N = tsize
        self.size = tsize

        self.target = [char for char in target]

    def calc_distance(self, other):
        """ implementation of Levenshtein distance between 2 genes """
        # Create and initializing the distance matrix (fill with 0's)
        dist_matrix = np.zeros((self.size + 1, self.size + 1))

        dist_matrix[:, 0] = np.arange(self.size + 1)
        dist_matrix[0, :] = np.arange(self.size + 1)

        # Calculate distances
        for x in range(1, self.size):
            for y in range(1, self.size):
                # if there is a match
                if self.string[x - 1] == other.string[y - 1]:
                    dist_matrix[x, y] = min(
                        dist_matrix[x - 1, y] + 1,
                        dist_matrix[x - 1, y - 1],
                        dist_matrix[x, y - 1] + 1
                    )
                else:
                    dist_matrix[x, y] = min(
                        dist_matrix[x - 1, y] + 1,
                        dist_matrix[x - 1, y - 1] + 1,
                        dist_matrix[x, y - 1] + 1
                    )

        return dist_matrix[self.size, self.size]

    def set_obj(self, obj):
        self.string = obj
        self.size = len(obj)

    def get_fitness(self, method=0):
        if method == 0:
            return self.pgia()

        else:
            return self.char_difference()

    def shuffle(self, start, end):
        shuff = self.string[start:end]
        shuffle(shuff)
        self.string[start:end] = shuff

    def char_difference(self):
        fitness = 0
        # iterate over the string to calculate differences
        for j in range(self.size):
            # ord represent each char as int (Unicode code)
            fitness += abs(ord(self.string[j]) - ord(self.target[j]))

        return fitness

    def pgia(self):
        """ Bool pgia heuristic.
            If the guess is in the right place, we will add points 0 to fitness (best bonus).
            If the guess is in target but not in the right place, we will add points 10 to fitness (medium bonus).
            Else, we will add 30 points to fitness (worst bonus). """
        fitness = 0
        # iterate over the string to calculate differences
        for j in range(self.size):
            # ord represent each char as int (Unicode code)
            char_to_check = self.string[j]
            if char_to_check == self.target[j]:
                continue
            else:
                fitness += 10
        return fitness

    def __getitem__(self, item):
        return self.string[item]

    def __setitem__(self, key, value):
        self.string[key] = value

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.string)

    def __str__(self):
        return str(self.string)

