import itertools
import time
from random import randint, shuffle
from collections import OrderedDict


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

        fitness = total_sum / self.N

        return fitness

    def get_empty_bins(self):
        unique_bins = set(self.bins)
        return self.N - len(unique_bins)

    def shuffle(self, start, end):
        shuff = self.bins[start:end]
        shuffle(shuff)
        self.bins[start:end] = shuff

    def calc_distance(self, other):

        # Invert self.bins
        identity = sorted(self.bins)
        ui = []
        for x in identity:
            index = self.bins.index(x)
            ui.append(identity[index])

        ##################################################
        _id = identity
        x = other.bins
        y = ui

        id_x_Map = OrderedDict(zip(_id, x))
        id_y_Map = OrderedDict(zip(_id, y))
        r = []
        for x_index, x_value in id_x_Map.items():
            for y_index, y_value in id_y_Map.items():
                if x_value == y_index:
                    r.append(y_value)

        ##################################################
        x = r

        values_checked = []
        unorderd_xr = []
        ordered_xr = []

        for value in x:
            values_to_right = []
            for n in x[x.index(value) + 1:]:
                values_to_right.append(n)
            result = [i for i in values_to_right if i < value]
            if len(result) != 0:
                values_checked.append(value)
                unorderd_xr.append(len(result))

        value_ltValuePair = OrderedDict(zip(values_checked, unorderd_xr))

        for key in sorted(value_ltValuePair):
            # print key,value_ltValuePair[key]
            ordered_xr.append(value_ltValuePair[key])

        distance = sum(ordered_xr)

        # print("Kendal Tau distance = ", distance)

        # print("Calc distance method time: ", time.time() - start)

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
