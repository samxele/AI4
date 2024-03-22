import numpy as np
import heuristics
import time

epsilon = 0.00001

class Game: 
    def __init__(self):
        self.board = np.zeros((3, 6, 7), dtype = int)
        self.board[0] = np.ones((6, 7), dtype = int)
        self.state = -1
        self.turn = 1
        self.column = -1

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
        
        # turn change
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1

    def game_win(self):
        if self.state == -1:
            self.state = win(self.board, self.turn)

    def game_minimax(self, depth):
        return minimax(self.board, self.turn, 0, depth)

    def playGame(self, pick_display = 0):
    # play game until either player wins or game draws
        while (self.state == -1):
            # make best move for current player for depth 3
            self.game_move(self.game_minimax(1)[1])
            self.game_win()
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
        return
    
    board[0][row][picked_column] = 0
    board[turn][row][picked_column] = 1

def win(board, turn):
    
    has_won = False

    dirs = [(0,1),(1,0),(1,1),(1,-1)] 

    # for each space (i, j)
        # for each direction d
            # visit 4 spaces in the given direction 
            # if off the grid, continue
            # if all X's or O's, finish 
            # otherwise, there's no win
    
    # [1 for i in range(5)] = [1, 1, 1, 1, 1]
    # [i for i in range(5)] = [0, 1, 2, 3, 4]
    # [i**2 for i in range(5)] = [0, 1, 4, 9, 16]
    # [(i, j) for i in range(7) for j in range(6)] = [(0,1),(0,2)...]
    
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

def minimax(board, player, picked_column, depth):
    # goal is for P1 to maximize score and for P2 to minimize score
    # valid scores can be between -1 and 1
    #display(board)
    #print (heuristics.static_evaluation(board, player),-1)
    if win(board, player) == player:
        return (1, picked_column)
    if win(board, player) == 3 - player:
        return (-1, picked_column)
    
    if depth == 0:
        return (heuristics.static_evaluation(board, player), -1)

    maxEval = -1
    evals = []
    for i in range(7):
        mod_board = np.copy(board)
        move(mod_board, i, player)
        # use new board, switch player's turn, and go deeper
        eval = minimax(mod_board, 3 - player, i, depth - 1)[0]
        if player == 2:
            eval *= -1
        evals.append(eval)
        if eval >= maxEval:
            maxEval = eval
    if depth == 1:
        print (evals)
    best_moves = []
    for i in range(len(evals)):
        if evals[i] > maxEval - epsilon:
            best_moves.append(i)
    best_move = np.random.choice(best_moves)
    return (maxEval * (1 if player == 1 else -1), best_move)