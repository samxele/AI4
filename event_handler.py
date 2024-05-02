from game import *
import gradio as gr
import image_generator

gameStarted = False
game = Game()
image_generator.generate_image(game.board, False, 0)

def update_image(column):
    global gameStarted
    if gameStarted:
        game.game_move(column)
        game.game_move(game.minimax_agent(2)[1])

    gameStarted = True
    image_generator.generate_image(game.board, (game.state != -1), game.turn)
    return gr.Image(type = "pil", value = "board_image.jpg")

with gr.Blocks() as demo:
    gr.Markdown("## Connect 4")
    gr.Markdown("Pick a column to place your square.")

    image_output = gr.Image(label = "Game Board", type = "pil")

    buttons = []
    buttonValues = []
    for i in range(7):
        buttonValue = gr.Number(value = i, visible = False)
        buttonValues.append(buttonValue)
    
    startButton = gr.Button("Start Game")
    startButton.click(update_image, inputs = buttonValues[0], outputs = image_output)
    with gr.Row():
        for i in range(7):
            button = gr.Button(value = str(i))
            buttons.append(button)
            button.click(update_image, inputs=buttonValues[i], outputs = image_output)

if __name__ == "__main__":
    demo.launch()
    image_output = update_image(0)