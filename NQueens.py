import itertools
from random import shuffle
import numpy as np


class n_queens:

    def __init__(self, N=None):
        # Create array of N queens in range [0, N-1] - each index represent the row of the queen in the board
        # Each value represent the column of the queen in the board
        if N is None:
            self.board = []
        else:
            self.board = list(np.random.randint(N, size=N))
        self.N = N

    def calc_distance(self, other):
        """ implementation of Kendall-tau distance between 2 genes """
        distance = 0

        # Create pairs
        pairs = itertools.combinations(range(0, self.N), 2)

        for elem_1, elem_2 in pairs:
            dist_1 = self.board.index(elem_1) - self.board.index(elem_2)
            dist_2 = other.board.index(elem_1) - other.board.index(elem_2)
            if dist_1 * dist_2 < 0:
                distance += 1

        return distance

    def set_obj(self, obj):
        self.board = obj
        self.N = len(obj)

    def get_fitness(self):
        """ Attacker Queen heuristic.
                for each Queen we will check how many Queens she attacks (row, column, diagonal)"""
        fitness = 0
        tsize = self.N
        for i in range(tsize):
            fitness += self.individual_attack(i=i, tsize=tsize)

        return fitness

    def individual_attack(self, i, tsize):
        inv_fitness = 0
        for j in range(i + 1, tsize):
            # Queens are in the same row
            if self.board[i] == self.board[j]:
                inv_fitness += 1
            # Get the difference between the current column
            # and the check column
            offset = j - i
            # To be a diagonal, the check column value has to be
            # equal to the current column value +/- the offset
            if 0 <= self.board[j] - offset == self.board[i]:
                inv_fitness += 1

            if self.board[i] == self.board[j] + offset:
                inv_fitness += 1
        return inv_fitness

    def shuffle(self, start, end):
        shuff = self.board[start:end]
        shuffle(shuff)
        self.board[start:end] = shuff

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        return self.board[item]

    def __setitem__(self, key, value):
        self.board[key] = value

    def __repr__(self):
        return str(self.board)

    def __str__(self):
        return str(self.board)

    def __index__(self, value):
        for i, val in enumerate(self.board):
            if val == value:
                return i
