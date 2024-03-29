from game import *
import gradio as gr
import image_generator

# Only for testing, remove in production!
# board = np.zeros((3, 6, 7), dtype = int)
# board[0] = np.ones((6, 7), dtype = int)

# board[0][5][2] = 0
# board[0][5][4] = 0
# board[1][5][2] = 1
# board[2][5][4] = 1

game = Game()
image_generator.generate_image(game.board)


def update_image(column):
    display(game.board)
    game.game_move(column)
    game.game_move(game.game_minimax(3)[1])
    image_generator.generate_image(game.board)
    return gr.Image(type = "pil", value = "board_image.jpg")


demo = gr.Interface(
    title = "Connect 4",
    description="Enter a number between 0-6 corresponding to what column you would like to place your square",
    fn = update_image,
    inputs = ["number"],
    outputs = ["image"]
)

if __name__ == "__main__":
    demo.launch()