from game import *
import gradio as gr
import image_generator

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetworkConnect4(nn.Module):
    # def __init__(self, env):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0)
        self.fc1 = nn.Linear(32*3*4, 42)
        self.fc2 = nn.Linear(42, 20)
        self.fc3 = nn.Linear(20, 7)


    def forward(self, x):
        x = x.unsqueeze(0)  # Add an extra dimension for the channels
        x = F.relu(self.conv(x))
        x = x.view(-1, 32*3*4)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(0)

gameStarted = False
game = Game()
net = DeepQNetworkConnect4()
# Load the network from q_conv_network.pth into net
net.load_state_dict(torch.load("best_q_weights_mm.pth"))
#net.load_state_dict(torch.load("best_q_weights_mm.pth").state_dict())
image_generator.generate_image(game.board, False, 0)

def update_image(column):
    global gameStarted
    if gameStarted:
        if game.game_move(column) != -1:
            game.get_computer_move(net)

    gameStarted = True
    image_generator.generate_image(game.board, game.state is not None, game.turn)
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