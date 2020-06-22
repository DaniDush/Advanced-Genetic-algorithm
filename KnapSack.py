from random import shuffle, randint
from Const import probs
from scipy.spatial import distance as scipy_dist

WL = 0
W = []
V = []


class knap_sack:
    def __init__(self, N=None):
        global WL, W, V
        self.sack = []
        sack_size = 0
        if N is not None:
            WL = probs[N][0]
            W = probs[N][1]
            V = probs[N][2]
            sack_size = len(W)
            for j in range(sack_size):
                # Adding random solution
                self.sack.append(randint(0, 1))
        self.N = sack_size

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        return self.sack[item]

    def __setitem__(self, key, value):
        self.sack[key] = value

    def set_obj(self, obj):
        self.sack = obj
        self.N = len(obj)

    def get_fitness(self):
        sum_weights = 0
        sum_values = 0

        # get weights and profits
        for i, item in enumerate(self.sack):
            if item == 0:
                continue
            else:
                sum_weights += W[i]
                sum_values += V[i]

        # if greater than the optimal return -1 or the number otherwise
        if sum_weights > WL:
            return -10
        else:
            return sum_values

    def shuffle(self, start, end):
        shuff = self.sack[start:end]
        shuffle(shuff)
        self.sack[start:end] = shuff

    def calc_distance(self, other):
        """ Calculating the euclidean distance between sacks """

        distance = round(scipy_dist.euclidean(self.sack, other.sack))*self.N

        return distance

    def __repr__(self):
        return str(self.sack)

    def __str__(self):
        return str(self.sack)
