""" implementation of minimal conflicts algorithm """

import numpy as np
import random
from NQueens import n_queens

REPLACE_PROB = 0.3


class minimal_conflicts:

    def __init__(self, size=8):
        self.obj = list(np.random.randint(size, size=size))
        self.size = size

    def attacker_queen(self):
        """ Attacker Queen heuristic.
            for each Queen we will check how many Queens she attacks (row, column, diagonal)"""
        fitness = 0
        for i in range(self.size):
            fitness += self.queen_conflicts(start=i, rand_queen=i)

        return fitness

    def queen_conflicts(self, rand_queen, start=-1):
        # Calculate queen conflicts using individual_attack heuristic
        inv_conflicts = 0
        for j in range(start + 1, self.size):
            # Queens are in the same row
            if j != rand_queen:
                if self.obj[rand_queen] == self.obj[j]:
                    inv_conflicts += 1
                # Get the difference between the current column
                # and the check column
                offset = j - rand_queen
                # To be a diagonal, the check column value has to be
                # equal to the current column value +/- the offset
                if 0 <= self.obj[j] - offset == self.obj[rand_queen]:
                    inv_conflicts += 1

                if self.obj[rand_queen] == self.obj[j] + offset:
                    inv_conflicts += 1
        return inv_conflicts
        # conflicts = Heuristics.individual_attack(inv=self.obj, i=rand_queen, tsize=self.size)

    def solve(self):
        while True:
            #   First we will choose random queen from the board
            rand_queen = random.randint(0, self.size - 1)
            min_conflicts = self.size + 1  # We cant attack more the N+1 queens

            if self.attacker_queen() == 0:
                return self.obj

            min_pos = -1  # Saving current position

            for i in range(0, self.size):
                self.obj[rand_queen] = i
                queen_conflicts = self.queen_conflicts(rand_queen)

                if queen_conflicts == min_conflicts:
                    if random.random() < REPLACE_PROB:
                        min_conflicts = queen_conflicts
                        min_pos = i

                elif queen_conflicts < min_conflicts:
                    min_conflicts = queen_conflicts
                    min_pos = i

            self.obj[rand_queen] = min_pos
            if self.attacker_queen() == 0:
                return self.obj
