from copy import deepcopy
from random import randint, random
import numpy as np

PROBLEM = 'X'

MAX_PARSE_TREE_DEPTH = 3
MAX_OPERATORS = 8
XOR_OPERATORS = ['AND', 'OR', 'NOT']
XOR_OPERANDS = ['A', 'B']
XOR_TABLE = [True, True, False, False]  # XOR function values from the table
A_INPUTS = np.array([True, False, False, True])
B_INPUTS = np.array([False, True, False, True])

######################### Testing #########################
# Test sequence 1 - ((A OR B) AND NOT((NOT A) OR (NOT B)))
TEST_TABLE_1 = [False, False, False, True]

# Test sequence 2 - ((A AND B) OR NOT((NOT A) AND (NOT B)))
TEST_TABLE_2 = [True, True, False, True]

######################### Univariate #########################
MATH_OPERATORS = ['+', '-', '*', '/']
MATH_OPERANDS = ['X']
X_INPUTS = []
Y_TABLE = []
X_RANGE = [-1, 1]
N_HITS = 20
MAX_EVAL = 50000
FITNESS_EVAL = 0
IS_TERMINATE = False
OPERATORS = []
OPERANDS = []


class Node:
    def __init__(self, value=None, right=None, left=None):
        self.value = value
        self.right = right
        self.left = left
        self.index = None

    def __str__(self):
        return self.value


class GP_tree:

    def __init__(self):
        self.root = Node()
        self.reservoir = None
        self.hits = None

    @staticmethod
    def generate_tree(current_node, GROW, current_depth):
        """Function to generate citizen as a Tree:
            param:
            GROW = True will activate the Grow method,
            GROW = False will activate the full method.
        """
        if current_depth == 0:
            current_node.value = np.random.choice(OPERATORS)

        elif current_depth < MAX_PARSE_TREE_DEPTH and GROW is False:
            action = np.random.choice(OPERATORS)
            current_node.value = action

        elif current_depth < MAX_PARSE_TREE_DEPTH and GROW is True:
            if random() < 0.5:
                action = np.random.choice(OPERANDS)
            else:
                action = np.random.choice(OPERATORS)
            current_node.value = action

        if current_depth == MAX_PARSE_TREE_DEPTH:
            action = np.random.choice(OPERANDS)
            current_node.value = action

        if current_node.value in OPERATORS:
            if current_node.value == 'NOT':
                current_node.left = Node()
                GP_tree.generate_tree(current_node=current_node.left, GROW=GROW, current_depth=current_depth + 1)
            else:
                current_node.left = Node()
                GP_tree.generate_tree(current_node=current_node.left, GROW=GROW, current_depth=current_depth + 1)
                current_node.right = Node()
                GP_tree.generate_tree(current_node=current_node.right, GROW=GROW, current_depth=current_depth + 1)

    # TODO make the fitness normalize (now its closer to 4)
    def get_fitness(self):
        """ Function for calculating the fitness of the program """
        global IS_TERMINATE, FITNESS_EVAL
        fitness = 0

        # Running the program
        return_values = run_gp_program(self.root)

        # XOR
        if PROBLEM == 'X':
            for idx, value in enumerate(return_values):
                if value == XOR_TABLE[idx]:
                    fitness += 2

        # MATH
        else:
            hits = 0
            for idx, value in enumerate(return_values):
                # print(value, Y_TABLE[idx])
                fitness += (value-Y_TABLE[idx])**2

                if Y_TABLE[idx] - 0.1 <= value <= Y_TABLE[idx] + 0.1:
                    hits += 1

            FITNESS_EVAL += 1
            self.hits = hits
            # Termination criteria
            if hits == N_HITS or FITNESS_EVAL >= MAX_EVAL:
                IS_TERMINATE = True

            return fitness

        operators_size = GP_tree.get_properties(self.root)

        # Anti-Bloating

        # Adding penalty for long trees
        fitness -= (0.1 * operators_size)

        if operators_size > MAX_OPERATORS or fitness < 0:
            fitness = 0

        return fitness

    def reservoir_sampling(self, pr=0.9):
        """ function to run reservoir sampling from the root """

        # For Cross-over pr=0.9, for mutation pr=0.5
        if random() < pr:
            node_type = OPERANDS
        else:
            node_type = OPERANDS

        index = 0
        self.reservoir = None
        while self.reservoir is None:
            self.sampling(current_node=self.root, index=index, node_type=node_type)
        return self.reservoir

    def sampling(self, current_node, index, node_type):
        """ Implementation of reservoir sampling """
        current_node.index = index
        if current_node.value in node_type:
            index += 1
            # determine if we replace the reservoir
            rnd_idx = randint(0, current_node.index + 1)
            if rnd_idx == 0:
                self.reservoir = current_node

        if current_node.left is not None:
            self.sampling(current_node=current_node.left, index=index, node_type=node_type)
        if current_node.right is not None:
            self.sampling(current_node.right, index, node_type)

    def branch_swap(self, branch, copy=True):
        """
        :param copy: True = making new copy of the branch
        :param branch: the new branch from the other parent
        :return:
        """

        self.reservoir.value = branch.value

        if copy:
            self.reservoir.left = deepcopy(branch.left)
            self.reservoir.right = deepcopy(branch.right)

        else:
            self.reservoir.left = branch.left
            self.reservoir.right = branch.right

    # TODO make the choice random !
    def branch_generator(self):
        GP_tree.generate_tree(current_node=self.reservoir, GROW=True, current_depth=MAX_PARSE_TREE_DEPTH - 3)

    @staticmethod
    def get_properties(root):
        """ function to get some properties of the tree such as height, number of operators, number of operands etc...)
        :param root:
        :return: return value by choice
        """
        size = 0
        leaf_count = 0
        min_leaf_depth = 0
        max_leaf_depth = -1
        is_strict = True
        current_level = [root]
        non_full_node_seen = False
        while len(current_level) > 0:
            max_leaf_depth += 1
            next_level = []

            for node in current_level:
                size += 1
                value = node.value

                # Node is a leaf.
                if node.left is None and node.right is None:
                    if min_leaf_depth == 0:
                        min_leaf_depth = max_leaf_depth
                    leaf_count += 1

                if node.left is not None:
                    next_level.append(node.left)
                    is_complete = not non_full_node_seen
                else:
                    non_full_node_seen = True

                if node.right is not None:
                    next_level.append(node.right)
                    is_complete = not non_full_node_seen
                else:
                    non_full_node_seen = True

                # If we see a node with only one child, it is not strict
                is_strict &= (node.left is None) == (node.right is None)

            current_level = next_level

        return size - leaf_count

    def __str__(self):
        lines = _build_tree_string(self.root, curr_index=0)[0]
        print('\n' + '\n'.join((line.rstrip() for line in lines)))
        return ""


def run_gp_program(node):
    """ Running the program with given A and B inputs """
    # Stop condition
    if node.value == 'A':
        return A_INPUTS
    elif node.value == 'B':
        return B_INPUTS
    elif node.value == 'X':
        return X_INPUTS

    ##########################################
    # If its an operator
    if node.value == 'AND':
        return np.logical_and(run_gp_program(node.left), run_gp_program(node.right))

    elif node.value == 'OR':
        return np.logical_or(run_gp_program(node.left), run_gp_program(node.right))

    elif node.value == 'NOT':
        return np.logical_not(run_gp_program(node.left))

    elif node.value == '+':
        return np.add(run_gp_program(node.left), run_gp_program(node.right))

    elif node.value == '-':
        return np.subtract(run_gp_program(node.left), run_gp_program(node.right))

    elif node.value == '/':
        value_array = np.true_divide(run_gp_program(node.left), run_gp_program(node.right))
        value_array[~ np.isfinite(value_array)] = 1  # -inf inf NaN
        return value_array

    elif node.value == '*':
        return np.multiply(run_gp_program(node.left), run_gp_program(node.right))

    # Should not get here
    return None


def function(X):
    # X^2+X+1
    return (X ** 2) + X + 1


def init_math_args(n=40):
    """ Function to initialize problem parameters """
    global N_HITS
    N_HITS = n

    for i in range(n):
        value = X_RANGE[0] + (i / (n - 1)) * (X_RANGE[1] - X_RANGE[0])
        X_INPUTS.append(value)
        Y_TABLE.append(function(value))


def _build_tree_string(root, curr_index, index=False, delimiter='-'):
    """ Printing the tree - taken from binarytree package
    :param root: Root node of the binary tree.
    :type root: binarytree.Node
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if index:
        node_repr = '{}{}{}'.format(curr_index, delimiter, root.value)
    else:
        node_repr = str(root.value)

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = \
        _build_tree_string(root.left, 2 * curr_index + 1, index, delimiter)
    r_box, r_box_width, r_root_start, r_root_end = \
        _build_tree_string(root.right, 2 * curr_index + 2, index, delimiter)

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(' ' * (l_root + 1))
        line1.append('_' * (l_box_width - l_root))
        line2.append(' ' * l_root + '/')
        line2.append(' ' * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(' ' * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append('_' * r_root)
        line1.append(' ' * (r_box_width - r_root + 1))
        line2.append(' ' * r_root + '\\')
        line2.append(' ' * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = ' ' * gap_size
    new_box = [''.join(line1), ''.join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
        r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end
