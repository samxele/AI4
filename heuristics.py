import numpy as np
import game

def scaled_sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1

def minimax(board, player, depth):
    # goal is for P1 to maximize score and for P2 to minimize score
    # valid scores can be between -1 and 1
    if board.state == 1:
        # player 1 has won, score of position is 1
        return (1, -1)
    elif board.state == 2:
        # player 2 has won, score of position is -1
        return (-1, -1)
    elif depth == 0:
        return static_evaluation(board, player)
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

# static evaluation of position
def static_evaluation(board, player):
    return scaled_sigmoid(center_control_val(board, player) + vertical_chain_val(board, player))

# heuristic for center control
def center_control_val(board, player):
    difference_Val = 0
    if player == 1:
        difference_Val = center_control(board,1) - center_control(board, 2)
    else:
        difference_Val = center_control(board,2) - center_control(board, 1)
    
    return scaled_sigmoid(difference_Val)

# evaluates center control for the player
def center_control(board, player):
    center_columns = 3
    center_control_score = 0
    center_control_score += np.sum(board[player, :, center_columns])
    return center_control_score

# heuristic for vertical chains
def vertical_chain_val(board, player):
    if player == 1:
        return vertical_chain(board, 1) - vertical_chain(board, 2)
    else:
        return vertical_chain(board, 2) - vertical_chain(board, 1)
        
# checks the number of 3 consecutive same player move in a column alligned together
def vertical_chain(board, player):
    # find the number 3-length vertical chains for the player
    chain_score = 0
    for row in range(4):
        for col in range(7):
            # check for 3 adjacent pieces
            if board[player, row:row+3, col].sum() == 3: 
                chain_score += 1
    return(chain_score)