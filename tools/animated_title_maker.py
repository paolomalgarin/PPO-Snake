# Script per generare una GIF del gioco eseguito da un modello PPO addestrato
import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil, glob



class ImageManager:

    IMAGE_NUM = 0

    def __init__(self, background = '#21212100'):
        self.background = background
        self.images = {
            'head': Image.open(os.path.join('tools', 'title_imgs', 'head.png')).convert('RGBA'),
            'body': Image.open(os.path.join('tools', 'title_imgs', 'body.png')).convert('RGBA'),
            'food': Image.open(os.path.join('tools', 'title_imgs', 'food.png')).convert('RGBA'),
            'food_eaten': Image.open(os.path.join('tools', 'title_imgs', 'food_eaten.png')).convert('RGBA'),
        }
    
    def draw_frame(self, state, head_rotation):
        font = ImageFont.truetype(os.path.join('tools', 'font', 'DejaVuSansMono.ttf'), size=30)

        squareW, squareH = self.images['head'].size

        grid_height, grid_width = state['dim']
        grid = make_grid(state)
        paddingX, paddingY = 10, 10

        img_width = int(squareW*grid_width + paddingX)
        img_height = int(squareH*grid_height + paddingY)

        img = Image.new('RGBA', (img_width, img_height), color=self.background)


        for i in range(grid_height):
            for j in range(grid_width):
                if grid[i][j] != 0:

                    drawX = j*squareW + paddingX // 2
                    drawY = i*squareH + paddingY // 2

                    match grid[i][j]:
                        case -1:
                            toprint = self.images['food_eaten']
                        case 1:
                            toprint = self.images['food']
                        case 2:
                            toprint = self.images['body']
                        case 3:
                            toprint = self.images['head'].rotate(head_rotation)

                    img.paste(toprint, (drawX, drawY), toprint)
        
        img.save(os.path.join(FRAME_FOLDER, f'frame_{ImageManager.IMAGE_NUM:05d}.png'))
        
        ImageManager.IMAGE_NUM += 1

    def create_gif(self, frames = None):
        if (frames == None):
            filenames = sorted(glob.glob(os.path.join(FRAME_FOLDER, 'frame_*.png')))
            frames = [Image.open(f) for f in filenames]

        frames[0].save(
            os.path.join(OUTPUT_ROOT, 'animated_title.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=200,
            disposal=2,
            optimize=False
        )
        frames[0].save(
            os.path.join(OUTPUT_ROOT, 'animated_title_repeat.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
            disposal=2,
            optimize=False
        )

    def clean_frame_folder(self):
        if os.path.exists(FRAME_FOLDER):
            shutil.rmtree(FRAME_FOLDER)


def make_grid(state):
    gridH, gridW = state['dim']

    grid = np.zeros((gridH, gridW), dtype=np.int32)

    for p in state['food']:
        row, col = p
        grid[row][col] = 1
    for p in state['eaten_food']:
        row, col = p
        grid[row][col] = -1
    for p in state['snake']['body']:
        row, col = p
        grid[row][col] = 2

    row, col = state['snake']['head']
    grid[row][col] = 3
    
    return grid


def print_frame(state):

    grid_height, grid_width = state['dim']
    grid = make_grid(state)


    for i in range(grid_height):
        for j in range(grid_width):
            match grid[i][j]:
                case -1:
                    print('O', end=' ')
                case 0:
                    print('.', end=' ')
                case 1:
                    print('◎', end=' ')
                case 2:
                    print('▢', end=' ')
                case 3:
                    print('▣', end=' ')
        print()
    

OUTPUT_ROOT = os.path.join('.', 'generated_gif')
FRAME_FOLDER = os.path.join(OUTPUT_ROOT, 'title_frames')
GRID_SIZE = 6
MODEL_PATH = os.path.join('agent', 'pretrained_models', 'size_6', 'final_model.pth')

if __name__ == "__main__":
    os.makedirs(FRAME_FOLDER, exist_ok=True)

    printer = ImageManager()

    game = {
        'dim': (8, 49),
        'food': [
            (0,1), (0,2), (0,3), (0,7), (0,8), (0,9), (0,13), (0,14), (0,15), (0,21), (0,22), (0,23), (0,26), (0,30), (0,34), (0,39), (0,42), (0,45), (0,46), (0,47),
            (1,0), (1,4), (1,6), (1,10), (1,12), (1,16), (1,20), (1,24), (1,26), (1,27), (1,30), (1,33), (1,35), (1,38), (1,41), (1,44), (1,48),
            (2,0), (2,4), (2,6), (2,10), (2,12), (2,16), (2,20), (2,26), (2,28), (2,30), (2,32), (2,36), (2,38), (2,40), (2,44),
            (3,0), (3,4), (3,6), (3,10), (3,12), (3,16), (3,21), (3,22), (3,23), (3,26), (3,28), (3,30), (3,32), (3,36), (3,38), (3,39), (3,44), (3,45), (3,46),
            (4,0), (4,1), (4,2), (4,3), (4,6), (4,7), (4,8), (4,9), (4,12), (4,16), (4,24), (4,26), (4,28), (4,30), (4,32), (4,33), (4,34), (4,35), (4,36), (4,38), (4,40), (4,44),
            (5,0), (5,6), (5,12), (5,16), (5,20), (5,24), (5,26), (5,29), (5,30), (5,32), (5,36), (5,38), (5,41), (5,44), (5,48),
            (6,0), (6,6), (6,13), (6,14), (6,15), (6,21), (6,22), (6,23), (6,26), (6,30), (6,32), (6,36), (6,38), (6,42), (6,45), (6,46), (6,47),
        ],
        'eaten_food': [
        ],
        'snake': {
            'head': (7,0),
            'body': [],
        },
    }

    done = False
    head_rotation = 0
    while not done:
        print("\n\n")

        printer.draw_frame(game, head_rotation)
        print_frame(game)

        print(f'Len: {len(game["snake"]["body"])}/49')
        user_input = input("Move (w=up, s=down, a=left, d=right, q=quit): ").lower()
        
        dx, dy = 0, 0
        match user_input:
            case 'w':
                dx, dy = (-1, 0)
                head_rotation = 90
            case 's':
                dx, dy = (1, 0)
                head_rotation = -90
            case 'a':
                dx, dy = (0, -1)
                head_rotation = 180
            case 'd':
                dx, dy = (0, 1)
                head_rotation = 0
            case 'q':
                done = True
            case _:
                continue
            
        if dx != 0 or dy != 0:
            current_head = game['snake']['head']
            game['snake']['head'] = (current_head[0] + dx, current_head[1] + dy)

        if (game['snake']['head'] in game['food']) and (game['snake']['head'] not in game['eaten_food']):
            game['eaten_food'].insert(0, game['snake']['head'])

            if (len(game['snake']['body']) != 0):
                game['snake']['body'].insert(len(game['snake']['body']), game['snake']['body'][-1])
            else:
                game['snake']['body'].insert(0, game['snake']['head'])
        
        game['snake']['body'].insert(0, game['snake']['head'])
        game['snake']['body'].pop()

    printer.create_gif()
    printer.clean_frame_folder()


