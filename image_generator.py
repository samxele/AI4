from PIL import Image
import numpy as np

def generate_image(board, gameEnded, turn):
    # Generate initial empty board
    empty_tile = Image.open("empty_tile.png")
    red_tile = Image.open("red_tile.png")
    yellow_tile = Image.open("yellow_tile.png")

    size = empty_tile.size[0]
    board_img = Image.new("RGB", (size * 7, size * 6))

    for row in range(6):
        for column in range(7):
            if (board)[row][column] == 0:
                board_img.paste(empty_tile, (column * size, row * size))
            elif (board)[row][column] == 1:
                board_img.paste(red_tile, (column * size, row * size))
            else:
                board_img.paste(yellow_tile, (column * size, row * size))

    # If game has ended, overlay corresponding message in image
    if (gameEnded):
        if (turn == 1):
            player_lose_message = Image.open("player_lose_message.png")
            board_img.paste(player_lose_message, 
                (int(size * 3.5 - player_lose_message.size[0] * 0.5), 
                int(size * 3 - player_lose_message.size[1] * 0.5)))
        else:
            player_win_message = Image.open("player_win_message.png")
            board_img.paste(player_win_message, 
                (int(size * 3.5 - player_win_message.size[0] * 0.5), 
                int(size * 3 - player_win_message.size[1] * 0.5)))
            

    # Save board to image file
    board_img.save("board_image.jpg")

    # # Display board
    # board_img.show()