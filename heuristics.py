import numpy as np
import game

# TO FIX - Add static evaluation of position

def minimax(board, player, depth):
    # goal is for P1 to maximize score and for P2 to minimize score
    # valid scores can be between -1 and 1
    if depth == 0:
        return  # add static evaluation
    elif board.state == 1:
        # player 1 has won, score of position is 1
        return (1, -1)
    elif board.state == 2:
        # player 2 has won, score of position is -1
        return (-1, -1)
    elif player == 1:
        # player 1 wants to do MUCH better than -1 (maximize score)
        maxEval = -1
        best_move = -1
        for i in range(7):
            mod_board = board
            mod_board.move(i)
            # use new board, switch player's turn, and go deeper
            eval = minimax(mod_board, 2, depth - 1)[0]
            if eval >= maxEval:
                maxEval = eval
                best_move = i
        return (maxEval, best_move)
    elif player == 2:
        # player 2 wants to do MUCH better than 1 (minimize score)
        minEval = 1
        best_move = -1
        for i in range(7):
            mod_board = board
            mod_board.move(i)
            # use new board, switch player's turn, and go deeper
            eval = minimax(mod_board, 1, depth - 1)[0]
            if eval <= minEval:
                minEval = eval
                best_move = i
        return (minEval, best_move)
    else:
        # error checking for invalid player number
        print("Invaild player!")
        return