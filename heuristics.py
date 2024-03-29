import numpy as np
import game

# sigmoid function
def scaled_sigmoid(x): # map number from [-inf, inf] into [-1, 1]
    return 2 / (1 + np.exp(-x)) - 1

# static evaluation of position
def static_evaluation(board, player):
    # print(scaled_sigmoid(center_control_val(board, player) + 3 * vertical_chain_val(board, player)))
    return scaled_sigmoid(center_control_val(board, player) + 3 * vertical_chain_val(board, player) +
                          3 * horizontal_chain_val(board, player))

# heuristic for center control
def center_control_val(board, player):
    difference_Val = center_control(board,1) - center_control(board, 2)
    return scaled_sigmoid(difference_Val)

# evaluates center control for the player
def center_control(board, player):
    center_columns = 3
    center_control_score = 0
    center_control_score += np.sum(board[player, :, center_columns])
    return center_control_score

# heuristic for vertical chains
def vertical_chain_val(board, player):
    return vertical_chain(board, 1) - vertical_chain(board, 2)
        
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

# heuristic for horizontal chains
def horizontal_chain_val(board, player):
    return horizontal_chain(board, 1) - horizontal_chain(board, 2)
        
# checks the number of 3 consecutive same player move in a column alligned together
def horizontal_chain(board, player):
    # find the number 3-length vertical chains for the player
    chain_score = 0
    for row in range(4):
        for col in range(7):
            # check for 3 adjacent pieces
            if board[player, row, col:col+3].sum() == 3: 
                chain_score += 1
    return(chain_score)