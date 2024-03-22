import numpy as np
import heuristics

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
        self.state = win(self.board, self.column, self.turn)

    def game_minimax(self, depth):
        return minimax(self.board, self.turn, 0, depth)

    def playGame(self, pick_display = 0):
    # play game until either player wins or game draws
        while (self.state == -1):
            # make best move for current player for depth 3
            self.game_move(self.game_minimax(3)[1])
            print("Hi: ", self.state)
            self.game_win()
            if pick_display == 1:
                display(self.board)
                print("Hi: ", self.state)

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

def win(board, picked_column, turn):
    # check if there is something in the last column
    check_column = 0
    for i in range(6):
        check_column += board[turn][i][picked_column]

    # the highest possible is at row 0
    highest = 0
    # iterate while the piece is there
    while board[turn][highest][picked_column] == 0:
        highest += 1
        if highest == 6:
            # the column is empty, and this edge case is triggered only at game start
            # so the game is ongoing
            return -1
    
    # check whether it works
    has_won = False
    squares_up = highest + 1
    squares_down = 6 - highest
    squares_left = picked_column + 1
    squares_right = 7 - picked_column
    
    # checks down
    if squares_down >= 4:
        square_sum = 0
        for i in range(0, 4):
            square_sum += board[turn][highest + i][picked_column]
        if square_sum == 4:
            has_won = True
    
    # checks left
    if squares_left >= 4:
        square_sum = 0
        for i in range(0, 4):
            square_sum += board[turn][highest][picked_column - i]
        if square_sum == 4:
            has_won = True
    
    # checks right
    if squares_right >= 4:
        square_sum = 0
        for i in range(0, 4):
            square_sum += board[turn][highest][picked_column + i]
        if square_sum == 4:
            has_won = True
    
    # checks top left
    if squares_up >= 4 and squares_left >= 4:
        square_sum = 0
        for i in range(0, 4):
            square_sum += board[turn][highest - i][picked_column - i]
        if square_sum == 4:
            has_won = True
    
    # checks bottom right
    if squares_down >= 4 and squares_right >= 4:
        square_sum = 0
        for i in range(0, 4):
            square_sum += board[turn][highest + i][picked_column + i]
        if square_sum == 4:
            has_won = True
    
    # checks top right 
    if squares_up >= 4 and squares_right >= 4:
        square_sum = 0
        for i in range(0, 4):
            square_sum += board[turn][highest - i][picked_column + i]
        if square_sum == 4:
            has_won = True
    
    # checks bottom left
    if squares_down >= 4 and squares_left >= 4:
        square_sum = 0
        for i in range(0, 4):
            square_sum += board[turn][highest + i][picked_column - i]
        if square_sum == 4:
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

def minimax(board, player, picked_column, depth):
    # goal is for P1 to maximize score and for P2 to minimize score
    # valid scores can be between -1 and 1

    #display(board)
    #print (heuristics.static_evaluation(board, player),-1)
    if win(board, picked_column, player) == player:
        return 1
    if win(board, picked_column, player) == 3 - player:
        return -1
    
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
    best_moves = []
    for i in range(len(evals)):
        if evals[i] > maxEval - epsilon:
            best_moves.append(i)
    best_move = np.random.choice(best_moves)
    return (maxEval * (1 if player == 1 else -1), best_move)
