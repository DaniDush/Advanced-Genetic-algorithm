import operator
import numpy as np

MAX_PARSE_TREE_DEPTH = 5
OPERATORS = [operator.not_, operator.and_, operator.or_]
OPERANDS = ['A', 'B']
OPTIONS = [OPERATORS, OPERANDS]


class GP_tree:

    def __init__(self, node=None, left_child=None, right_child=None, height=0):
        self.node = node
        self.left_child = left_child
        self.right_child = right_child
        self.height = height

    def generate_tree(self, GROW=True, current_depth=0):
        """Function to generate citizen as a Tree: GROW = True will activate the Grow method,
            GROW = False will activate the full method.
        """

        # If its Grow method and out current depth isn't the maximum
        if current_depth < MAX_PARSE_TREE_DEPTH and GROW:
            action = np.random.choice(OPTIONS)
            self.node = np.random.choice(action)
            print(self.node)

        # If its Full method and out current depth isn't the maximum
        # Or the current node is the root
        elif current_depth == 0 or current_depth < MAX_PARSE_TREE_DEPTH and not GROW:
            self.node = np.random.choice(OPERATORS)

        # If we are in the maximum depth
        elif current_depth == MAX_PARSE_TREE_DEPTH:
            self.node = np.random.choice(OPERANDS)

        # If the node contains AND | OR we will create 2 sub tress
        if self.node != operator.not_ and self.node in OPERATORS:
            self.height += 1
            self.left_child = GP_tree()
            self.left_child.generate_tree(GROW=GROW, current_depth=current_depth+1)
            self.right_child = GP_tree()
            self.right_child.generate_tree(GROW=GROW, current_depth=current_depth+1)

        # If the node contains NOT we will create an operand as left child
        elif self.node == operator.not_:
            self.height += 1
            self.left_child = np.random.choice(OPERANDS)

        print(self.height)

    #
    # def calc_fitness(self):
