from random import randint, random
import numpy as np

MAX_PARSE_TREE_DEPTH = 8
OPERATORS = ['AND', 'OR', 'NOT']
NEGATIVE_OPERATORS = ['NAND', 'NOR']
OPERANDS = ['A', 'B']
NEGATIVE_OPERANDS = ['NotA', 'NotB']
OPTIONS = [OPERATORS, OPERANDS]
INPUTS = [[True, False], [False, True], [False, False], [True, True]]   # Inputs for A and B
TABLE = [True, True, False, False]  # XOR function values from the table


class GP_tree:

    operators_counter = 0

    def __init__(self, node=None, father=None):
        self.node = node
        self.father = father
        self.children = []
        self.reservoir = None

    def generate_tree(self, GROW=True, current_depth=0):
        """Function to generate citizen as a Tree:
            param:
            GROW = True will activate the Grow method,
            GROW = False will activate the full method.
        """
        if current_depth == 0:
            self.node = np.random.choice(OPERATORS)

        # If its Grow method and out current depth isn't the maximum
        elif current_depth < MAX_PARSE_TREE_DEPTH and GROW:
            action = np.random.choice(OPTIONS)
            self.node = np.random.choice(action)

        # If its Full method and out current depth isn't the maximum
        # Or the current node is the root
        elif current_depth < MAX_PARSE_TREE_DEPTH and not GROW:
            self.node = np.random.choice(OPERATORS)

        # If we are in the maximum depth
        elif current_depth == MAX_PARSE_TREE_DEPTH:
            self.node = np.random.choice(OPERANDS)

        # If the node contains NOT we will create an operand as left child
        if self.node == 'NOT':
            self.node = GP_tree.operator_not_convergence(self.node)

        # If the node contains AND | OR we will create 2 sub tress
        if self.node in OPERATORS or self.node in NEGATIVE_OPERATORS:
            self.children.append(GP_tree(father=self.node))
            self.children[0].generate_tree(GROW=GROW, current_depth=current_depth + 1)
            self.children.append(GP_tree(father=self.node))
            self.children[1].generate_tree(GROW=GROW, current_depth=current_depth + 1)

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
        operators = 0
        num_of_inputs = len(INPUTS)
        for i in range(num_of_inputs):
            return_value = run_gp_program(self, A=INPUTS[i][0], B=INPUTS[i][1])

            if return_value == TABLE[i]:
                fitness += 1

            if i == 0:
                operators = GP_tree.operators_counter

        # Adding penalty for long trees
        fitness += (len(OPERATORS) - (0.1 * operators))

        # Initialize num of operators for the next tree
        GP_tree.operators_counter = 0

        return fitness

    def reservoir_sampling(self):
        if random() < 0.9:
            node_type = OPERATORS
        else:
            node_type = OPERANDS

        """ function to run reservoir sampling from the root """
        index = 0
        self.reservoir = None
        while self.reservoir is not None:
            GP_tree.sampling(tree=self, current_node=self, index=index, node_type=node_type)
        return self.reservoir

    @staticmethod
    def sampling(tree, current_node, index, node_type):
        """ Implementation of reservoir sampling """
        index += 1
        # determine if we replace the reservoir
        rnd_idx = randint(0, index)
        if rnd_idx == 0:
            if current_node.node in node_type:
                tree.reservoir = current_node
            for child in current_node.children:
                GP_tree.sampling(tree, child, index, node_type)

    @staticmethod
    def display_tree(current_node):
        for i, child in enumerate(current_node.children):
            if i == 0:
                print("/\n /")
            else:
                print("\ \n \ ")

            GP_tree.display_tree(child)


def run_gp_program(node, A=True, B=False):
    """ Running the program with given A and B inputs """
    # Stop condition
    if node.node == 'A':
        return A
    elif node.node == 'NotA':
        return not A
    elif node.node == 'B':
        return B
    elif node.node == 'NotB':
        return not B

    # If its operator
    GP_tree.operators_counter += 1

    if node.node == 'AND':
        return run_gp_program(node.children[0], A=A, B=B) and run_gp_program(node.children[1], A=A, B=B)

    elif node.node == 'OR':
        return run_gp_program(node.children[0], A=A, B=B) or run_gp_program(node.children[1], A=A, B=B)

    elif node.node == 'NAND':
        return not (run_gp_program(node.children[0], A=A, B=B) and run_gp_program(node.children[1], A=A, B=B))

    elif node.node == 'NOR':
        return not (run_gp_program(node.children[0], A=A, B=B) or run_gp_program(node.children[1], A=A, B=B))

    # Should not get here
    return None
