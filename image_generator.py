from PIL import Image
import numpy as np

def generate_image(board):
    # Generate initial empty board
    empty_tile = Image.open("empty_tile.png")
    red_tile = Image.open("red_tile.png")
    yellow_tile = Image.open("yellow_tile.png")

    size = empty_tile.size[0]
    board_img = Image.new("RGB", (size * 7, size * 6))

    for row in range(6):
        for column in range(7):
            if (board)[0][row][column] == 1:
                board_img.paste(empty_tile, (column * size, row * size))
            elif (board)[1][row][column] == 1:
                board_img.paste(red_tile, (column * size, row * size))
            elif (board)[2][row][column] == 1:
                board_img.paste(yellow_tile, (column * size, row * size))

    # Save board to image file
    board_img.save("board_image.jpg")

    # # Display board
    # board_img.show()