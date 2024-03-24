"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    o_count = 0
    x_count = 0
    for row in range(3):
        for col in range(3):
            if board[row][col] == X:
                x_count += 1
            elif board[row][col] == O:
                o_count += 1
    
    if x_count > o_count:
        return O
    else:
        return X

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for row in range(3):
        for col in range(3):
            if board[row][col] is EMPTY:
                actions.add((row, col))

    return actions

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    row, col = action
    
    if board[row][col] is not EMPTY:
        raise Exception("Invalid action")
    
    new_board = [row[:] for row in board] 
    new_board[row][col] = player(board)  
    return new_board 


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] == X:
            return X
        elif board[row][0] == board[row][1] == board[row][2] == O:
            return O    

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] == X:
            return X
        elif board[0][col] == board[1][col] == board[2][col] == O:
            return O

    if board[0][0] == board[1][1] == board[2][2] == X:
        return X
    elif board[0][0] == board[1][1] == board[2][2] == O:
        return O
    if board[0][2] == board[1][1] == board[2][0] == X:
        return X
    elif board[0][2] == board[1][1] == board[2][0] == O:
        return O
    
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    for row in range(3):
        for col in range(3):
            if board[row][col] is EMPTY:
                return False

    return True



def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0

def max_value(board):
    if terminal(board):
        return utility(board)
    value = float("-inf")
    for action in actions(board):
        value = max(value, min_value(result(board, action)))

    return value


def min_value(board):
    if terminal(board):
        return utility(board)
    value = float("inf")
    for action in actions(board):
        value = min(value, max_value(result(board, action)))
    return value


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    print(terminal(board))
    if terminal(board):
        return None

    current_player = player(board)
    best_action = None
    if current_player == X:
        best_value = float("-inf")
        for action in actions(board):
            value = min_value(result(board, action))
            if value > best_value:
                best_value = value
                best_action = action
    else:
        best_value = float("inf")
        for action in actions(board):
            value = max_value(result(board, action))
            if value < best_value:
                best_value = value
                best_action = action

    return best_action