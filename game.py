import numpy as np

class Game: 
    def __init__(self):
        self.board = np.zeros((3, 6, 7), dtype = int)
        self.board[0] = np.ones((6, 7), dtype = int)
        self.state = -1
        self.turn = 1
        self.column = -1
    
    def display(self):
        output = ""
        for row in range(6):
            for column in range(7):
                if (self.board)[0][row][column] == 1:
                    output += "."
                elif (self.board)[1][row][column] == 1:
                    output += "X"
                elif (self.board)[2][row][column] == 1:
                    output += "O"
                output += " "
            output += "\n"
        print(output)

    '''
    --> FIX THIS!
    '''
    def move(self, picked_column):
        
        
        if picked_column < 0 or picked_column > 6:
            # Print or return a value for invalid input
            pass
            
        row = 5
        
        # Check 
        while row >= 0 and self.board[0][row][picked_column] != 1:
            row -= 1
            
        if row == -1:
            # column full
            # Print column full
            return
        
        self.board[0][row][picked_column] = 0
        self.board[self.turn][row][picked_column] = 1
    '''
    --> End of code that needs to be fixed
    '''

    def win(self):
        # flips the turn (if it's P1's turn, P2 moved last)
        turn = 3 - self.turn
        
        # the highest possible is at row 0
        highest = 0
        # iterate while the piece is there
        while (self.board[turn][highest][self.column] == 0):
            highest += 1
        
        # check whether it works
        has_won = False
        squares_up = highest + 1
        squares_down = 6 - highest
        squares_left = self.column + 1
        squares_right = 7 - self.column
        
        # checks down
        if (squares_down >= 4):
            square_sum = 0
            for i in range(0, 4):
                square_sum += self.board[turn][highest + i][self.column]
            if (square_sum == 4):
                has_won = True
        
        # checks left
        if (squares_left >= 4):
            square_sum = 0
            for i in range(0, 4):
                square_sum += self.board[turn][highest][self.column - i]
            if (square_sum == 4):
                has_won = True
        
        # checks right
        if (squares_right >= 4):
            square_sum = 0
            for i in range(0, 4):
                square_sum += self.board[turn][highest][self.column + i]
            if (square_sum == 4):
                has_won = True
        
        # checks top left
        if (squares_up >= 4 and squares_left >= 4):
            square_sum = 0
            for i in range(0, 4):
                square_sum += self.board[turn][highest - i][self.column - i]
            if (square_sum == 4):
                has_won = True
        
        # checks bottom right
        if (squares_down >= 4 and squares_right >= 4):
            square_sum = 0
            for i in range(0, 4):
                square_sum += self.board[turn][highest + i][self.column + i]
            if (square_sum == 4):
                has_won = True
        
        # checks top right 
        if (squares_up >= 4 and squares_right >= 4):
            square_sum = 0
            for i in range(0, 4):
                square_sum += self.board[turn][highest - i][self.column + i]
            if (square_sum == 4):
                has_won = True
        
        # checks bottom left
        if (squares_down >= 4 and squares_left >= 4):
            square_sum = 0
            for i in range(0, 4):
                square_sum += self.board[turn][highest + i][self.column - i]
            if (square_sum == 4):
                has_won = True
        
        # modify the state based on boolean
        if has_won:
            self.state = turn
        else:
            empty_sum = 0
            for col in range(7):
                empty_sum += self.board[0][0][col]
            if (empty_sum == 7):
                self.state = 0