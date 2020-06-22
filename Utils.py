from pathlib import Path

species_thres = [4, 6.3, 8.91, 12.7]


def print_nqueens_board(final_board, N):
    # print final board
    board = ["-"] * N ** 2
    for i in range(0, N):
        board[i * N + final_board[i]] = "Q"

    for row in range(0, N):
        print(board[row * N:(row + 1) * N])

    print("")


def get_args(question):
    problem_args = []

    path = Path(f'{question}.txt')
    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        problem_args.append(int(lines[i][:-1]))

    N = int(problem_args[0])
    C = int(problem_args[1])
    WEIGHTS = problem_args[2:]

    return N, C, WEIGHTS, species_thres[question - 1]
