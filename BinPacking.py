from random import randint, shuffle
from scipy.spatial import distance as scipy_dist


class bin_packing:
    W = []
    C = 0

    def __init__(self, N, is_buffer):
        self.bins = []
        if not is_buffer:
            bins = list(range(0, N))
            shuffle(bins)
            self.bins = bins

        self.N = N
        self.sum_of_bins = []

    def get_sum_of_bin(self, idx):
        sum_of_bin = 0
        items = []
        for item in self.bins:
            if idx == item.bin:
                items.append(item)
                sum_of_bin += item.weight

        return sum_of_bin, items

    def set_obj(self, obj):
        self.bins = obj

    def get_fitness(self):
        total_sum = 0
        K = 2
        bin_in_used = 0
        sum_of_bins = [0] * self.N
        # get weights and profits
        for item, _bin in enumerate(self.bins):
            if sum_of_bins[_bin] == 0:
                bin_in_used += 1
            sum_of_bins[_bin] += bin_packing.W[item]

        self.sum_of_bins = sum_of_bins

        for s in sum_of_bins:
            total_sum += (s / bin_packing.C) ** K

        fitness = total_sum / bin_in_used

        return fitness

    def get_empty_bins(self):
        unique_bins = set(self.bins)
        return self.N - len(unique_bins)

    def shuffle(self, start, end):
        shuff = self.bins[start:end]
        shuffle(shuff)
        self.bins[start:end] = shuff

    def calc_distance(self, other):
        """ Calculating the euclidean distance between bins """

        distance = round(scipy_dist.euclidean(self.bins, other.bins))

        return distance

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        return self.bins[item]

    def __setitem__(self, key, value):
        self.bins[key] = value

    def __repr__(self):
        return str(self.bins)

    def __str__(self):
        return str(self.bins)
