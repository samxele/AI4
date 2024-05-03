import numpy as np
import game

# sigmoid function
def scaled_sigmoid(x): # map number from [-inf, inf] into [-1, 1]
    return 2 / (1 + np.exp(-x)) - 1

# static evaluation of position
def static_evaluation(board, player):
    # print(scaled_sigmoid(center_control_val(board, player) + 3 * vertical_chain_val(board, player)))
    return scaled_sigmoid(1 * center_control(board) +
                          1 * win_threat_val(board, player))

# evaluates center control 
def center_control(board):
    center_control_score = 0
    center_control_score += 0.25 * np.sum(board[:, 0])
    center_control_score += 0.50 * np.sum(board[:, 1])
    center_control_score += 0.75 * np.sum(board[:, 2])
    center_control_score += 1.00 * np.sum(board[:, 3])
    center_control_score += 0.75 * np.sum(board[:, 4])
    center_control_score += 0.50 * np.sum(board[:, 5])
    center_control_score += 0.25 * np.sum(board[:, 6])
    return center_control_score

# heuristic for win threats
def win_threat_val(board, player):
    return win_threat(board, 1) - win_threat(board, 2)

# checks the number of win threats that a player has
def win_threat(board, player):
    simple_threats = 0
    danger_threats = 0
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    # iterates through to find number of threats
    for row in range(6):
        for col in range(7):
            for (dx, dy) in dirs:
                if 0 <= (row + 3 * dx) <= 5 and 0 <= (col + 3 * dy) <= 6:
                    # check 4 in a row in the direction of (dx, dy)
                    summer = 0
                    emptyTile = (-1, -1)
                    for i in range(4):
                        if board[row + i * dx][col + i * dy] == 0:
                            emptyTile = (row + i * dx, col + i * dy)
                        summer += board[row + i * dx][col + i * dy]
                        
                    if summer == (3 if player == 1 else -3):
                        if emptyTile[0] == 5 or board[emptyTile[0] + 1][emptyTile[1]] != 0:
                            simple_threats += 1
                        else:
                            danger_threats += 1
    return 0.5 * simple_threats + 3 * danger_threats