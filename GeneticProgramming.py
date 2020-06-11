from copy import deepcopy
from random import randint, random
import numpy as np

MAX_PARSE_TREE_DEPTH = 3
OPERATORS = ['AND', 'OR', 'NOT']
NEGATIVE_OPERATORS = ['NAND', 'NOR']
OPERANDS = ['A', 'B']
NEGATIVE_OPERANDS = ['NotA', 'NotB']
OPTIONS = [OPERATORS, OPERANDS]
INPUTS = [[True, False], [False, True], [False, False], [True, True]]  # Inputs for A and B
TABLE = [True, True, False, False]  # XOR function values from the table


class Node:
    def __init__(self, value=None, right=None, left=None):
        self.value = value
        self.right = right
        self.left = left
        self.index = None

    def __str__(self):
        return self.value


class GP_tree:
    operators_counter = 0

    def __init__(self, node=None, father=None):
        self.root = Node()
        self.reservoir = None

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

    @staticmethod
    def operator_not_convergence(node):
        """ function to converge the not operator with its child """
        action = np.random.choice(OPTIONS)
        rand_choice = np.random.choice(action)

        while rand_choice == 'NOT':
            action = np.random.choice(OPTIONS)
            rand_choice = np.random.choice(action)

        if rand_choice == 'A':
            node = 'NotA'
        elif rand_choice == 'B':
            node = 'NotB'
        elif rand_choice == 'AND':
            node = 'NAND'
        elif rand_choice == 'OR':
            node = 'NOR'

        return node

    # TODO make the fitness normalize (now its closer to 4)
    def get_fitness(self):
        """ Function for calculating the fitness of the program """
        fitness = 0
        num_of_inputs = len(INPUTS)
        for i in range(num_of_inputs):
            return_value = run_gp_program(self.root, A=INPUTS[i][0], B=INPUTS[i][1])

            if return_value == TABLE[i]:
                fitness += 2

        operators_size = GP_tree.get_properties(self.root)

        # Anti-Bloating

        # Adding penalty for long trees
        fitness -= (0.1 * operators_size)

        if operators_size > 8 or fitness < 0:
            fitness = 0

        # Initialize num of operators for the next tree
        GP_tree.operators_counter = 0

        return fitness

    def reservoir_sampling(self, pr=0.9):
        if random() < pr:
            node_type = OPERANDS
        else:
            node_type = OPERANDS

        """ function to run reservoir sampling from the root """
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

    def branch_swap(self, branch):
        """
        :param branch: the new branch from the second parent
        :return:
        """

        self.reservoir.value = branch.value
        self.reservoir.left = deepcopy(branch.left)
        self.reservoir.right = deepcopy(branch.right)

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


def run_gp_program(node, A=True, B=False):
    """ Running the program with given A and B inputs """
    # Stop condition
    if node.value == 'A':
        return A
    elif node.value == 'B':
        return B

    ##########################################
    # If its an operator
    if node.value == 'AND':
        return run_gp_program(node.left, A=A, B=B) and run_gp_program(node.right, A=A, B=B)

    elif node.value == 'OR':
        return run_gp_program(node.left, A=A, B=B) or run_gp_program(node.right, A=A, B=B)

    elif node.value == 'NOT':
        return not (run_gp_program(node.left, A=A, B=B))

    # Should not get here
    return None


def _build_tree_string(root, curr_index, index=True, delimiter='-'):
    """Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
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
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
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
