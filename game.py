import numpy as np
import heuristics
from dataclasses import dataclass

# seed np rng
# np.random.seed(0)

epsilon = 0.00001
maxDepth = 4

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray

class Game: 
    def __init__(self):
        self.board = np.zeros((3, 6, 7), dtype = int)
        self.board[0] = np.ones((6, 7), dtype = int)
        self.state = -1  # 0 if drawn, {player} if won
        self.turn = 1    # Current player's turn
        self.column = -1 # Last picked column (starts at 0)
        self.history = []     # Contains all boards
        self.history.append(np.copy(self.board))
        self.experiences = [] # Contains all experiences

    def game_move(self, picked_column):

        if self.state != -1:
            print("Game over")
            return

        # calls move from outside the class
        move(self.board, picked_column, self.turn)

        # update last chosen column
        self.column = picked_column
        
        # check for win
        self.game_win()
        
        # add new board to history, add new experience
        self.history.append(np.copy(self.board))
        self.experiences.append((self.history[-2], picked_column, None, self.history[-1]))
        
        # turn change
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1

    def game_win(self):
        if self.state == -1:
            self.state = win(self.board, self.turn)

    def game_minimax(self, depth):
        return minimax(self.board, self.turn, 0, depth, -1, 1)

    def playGame(self, pick_display = 0):
        # play game until either player wins or game draws
        while (self.state == -1):
            # make best move for current player for the max depth
            self.game_move(self.game_minimax(maxDepth)[1])
            if pick_display == 1:
                display(self.board)

def display(board):
    output = ""
    for row in range(6):
        for column in range(7):
            if (board)[0][row][column] == 1:
                output += "."
            elif (board)[1][row][column] == 1:
                output += "X"
            elif (board)[2][row][column] == 1:
                output += "O"
            output += " "
        output += "\n"
    print(output)

def move(board, picked_column, turn):

    if picked_column == -1:
        return -1
        
    row = 5
    # check 
    while row >= 0 and board[0][row][picked_column] == 0:
        row -= 1
        
    if row == -1:
        return -1
    
    board[0][row][picked_column] = 0
    board[turn][row][picked_column] = 1

def win(board, turn):
    
    has_won = False

    dirs = [(0,1),(1,0),(1,1),(1,-1)] 

    for row in range(6):
        for col in range(7):
            for (dx, dy) in dirs:
                if 0 <= (row + 3 * dx) <= 5 and 0 <= (col + 3 * dy) <= 6:
                    # check 4 in a row in the direction of (dx, dy)
                    check_sum = 0
                    for i in range(4):
                        check_sum += board[turn][row + i * dx][col + i * dy]
                    if check_sum == 4:
                        has_won = True
                
    # modify the state based on boolean
    if has_won:
        return turn
    else:
        empty_sum = 0
        for col in range(7):
            empty_sum += board[0][0][col]
        if empty_sum == 0:
            return 0
    return -1

def minimax(board, player, picked_column, depth, alpha, beta):
    # check if we win, lose, or draw
    won_player = win(board, player)
    if won_player == 1:
        # print ("A", won_player, player)
        # display(board)
        return (2, picked_column)
    if won_player == 2:
        # print ("B", won_player, player)
        return (-2, picked_column)
    if won_player == 0:
        # print ("C")
        return (0, picked_column)

    # if depth is 0, then return heuristic value
    if depth == 0:
        return (heuristics.static_evaluation(board, player), -1)

    # if we are player 1, then:
        # value = -infinity
        # for each child of node, do:
            # set the value to max of (board, value, minimax(child, 2, depth - 1))
    # return value
    
    if player == 1:
        max_eval = float('-inf')
        best_move = -1
        for possible_move in range(7):
            mod_board = np.copy(board)
            if move(mod_board, possible_move, 1) == -1:
                continue
            val = minimax(mod_board, 2, possible_move, depth - 1, alpha, beta)[0]
            # CHECK FOR MATE IN ONE ERRORS
            if depth == maxDepth:
                print((possible_move, val))
            if val > max_eval:
                max_eval = val
                best_move = possible_move
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                break
        return (max_eval, best_move)
    if player == 2:
        min_eval = float('+inf')
        best_move = -1
        for possible_move in range(7):
            mod_board = np.copy(board)
            if move(mod_board, possible_move, 2) == -1:
                continue
            val = minimax(mod_board, 1, possible_move, depth - 1, alpha, beta)[0]
            # CHECK FOR MATE IN ONE ERRORS
            if depth == maxDepth:
                print((possible_move, val))
            if val < min_eval:
                min_eval = val
                best_move = possible_move
            beta = min(beta, min_eval)
            if beta <= alpha:
                break
        return (min_eval, best_move)