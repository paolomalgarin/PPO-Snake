# Script per generare una GIF del gioco eseguito da un modello PPO addestrato
import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil, glob
from agent.ppo_agent import PPOAgent
from env.snake_env import SnakeEnv



class ImageManager:

    IMAGE_NUM = 0

    def __init__(self, foreground = '#eee', background = '#212121'):
        self.foreground = foreground
        self.background = background
    
    def draw_frame(self, obs):
        font = ImageFont.truetype(os.path.join('tools', 'font', 'DejaVuSansMono.ttf'), size=30)

        charW = font.getlength('_')
        ascent, descent = font.getmetrics()
        charH = ascent

        grid_size = len(obs[0][0])
        spacing = charW // 2
        paddingX, paddingY = charW, charW

        img_width = int(charW*grid_size + (grid_size - 1)*spacing + paddingX)
        img_height = int(charH*grid_size + paddingY)

        img = Image.new('RGB', (img_width, img_height), color=self.background)
        draw = ImageDraw.Draw(img)


        for i in range(grid_size):
            for j in range(grid_size):
                if (obs[0][i][j] != 0): 
                    char = '▣'
                elif (obs[1][i][j] != 0):
                    char = '▢'
                elif (obs[2][i][j] != 0):
                    char = '◎'
                else:
                    char = '.'

                drawX = j*charW + j*spacing + paddingX//2
                drawY = i*charH

                draw.text((drawX, drawY), char, fill=self.foreground, font=font)
        
        img.save(os.path.join(OUTPUT_ROOT, 'frames', f'frame_{ImageManager.IMAGE_NUM:05d}.png'))
        
        ImageManager.IMAGE_NUM += 1

    def create_gif(self, frames = None):
        if (frames == None):
            filenames = sorted(glob.glob(os.path.join(OUTPUT_ROOT, 'frames', 'frame_*.png')))
            frames = [Image.open(f) for f in filenames]

        frames[0].save(
            os.path.join(OUTPUT_ROOT, 'gameplay.gif'),
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0,
            disposal=2,
            optimize=False
        )

    def clean_frame_folder(self):
        FRAME_DIR = os.path.join(OUTPUT_ROOT, 'frames')
        if os.path.exists(FRAME_DIR):
            shutil.rmtree(FRAME_DIR)


OUTPUT_ROOT = os.path.join('.', 'generated_gif')
GRID_SIZE = 6
MODEL_PATH = os.path.join('agent', 'pretrained_models', 'size_6', 'final_model.pth')

if __name__ == "__main__":
    frames_dir = os.path.join(OUTPUT_ROOT, 'frames')
    os.makedirs(frames_dir, exist_ok=True)


    parser = argparse.ArgumentParser(description='arguments')
    
    parser.add_argument('--grid-size', type=int, default=None, help='Size of the grid')
    parser.add_argument('--path', type=str, default=None, help='Model path')
    args = parser.parse_args()
    
    if args.grid_size is not None:
        GRID_SIZE = args.grid_size
    if args.path is not None:
        MODEL_PATH = args.path
    else:
        MODEL_PATH = os.path.join('agent', 'pretrained_models', f'size_{GRID_SIZE}', 'final_model.pth')


    env = SnakeEnv(useGui=False, gridW=GRID_SIZE, gridH=GRID_SIZE)
    agent = PPOAgent(env)
    printer = ImageManager()

    agent.load(MODEL_PATH)

    obs, _ = env.reset()
    done = False
    step = 0

    printer.draw_frame(obs)

    with torch.no_grad():
        while not done:
            # Azione deterministica del modello
            action, _ = agent.get_action(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
            printer.draw_frame(obs)
    
    printer.create_gif()
    printer.clean_frame_folder()
            
    env.close()
    print(f"Episodio terminato dopo {step} passi. Punteggio: {env.game.score}")