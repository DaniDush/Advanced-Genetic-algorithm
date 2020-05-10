
class bin_packing:
    def __init__(self, bins, N, C, WEIGHTS):
        self.bins = bins
        self.C = C
        self.N = N
        self.W = WEIGHTS

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        return self.bins[item]

    def __setitem__(self, key, value):
        self.bins[key] = value

    def set_obj(self, obj):
        self.bins = obj

    def get_fitness(self):
        total_sum = 0

        K = 1
        sum_bins = [0] * self.N
        # get weights and profits
        for i, _bin in enumerate(self.bins):
            sum_bins[_bin] += self.W[i]

        for s in sum_bins:
            total_sum += (s / self.C) ** K

        fitness = total_sum / self.N

        return fitness

    def __repr__(self):
        return str(self.bins)

    def __str__(self):
        return str(self.bins)
