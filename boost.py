# Training loop
from agent.ppo_agent import PPOAgent
from env.snake_env import SnakeEnv
from tools.beautyful_progress_bar import PBar
import os, json, torch, time, argparse, numpy as np
from torch import nn


TRAINING_TIMESTEPS = 8_000_000  # Number of timesteps the model will be trained for
PATH = os.path.join('results', 'model', 'final_model.pth')


if __name__ == "__main__":
    # Handle params
    parser = argparse.ArgumentParser(description='trainig arguments')
    
    parser.add_argument('--train-ts', type=int, default=None, help='Number of timesteps the model will be trained for')
    parser.add_argument('--path', type=int, default=None, help='Path where the model is')
    args = parser.parse_args()
    
    if args.train_ts is not None:
        TRAINING_TIMESTEPS = args.train_ts
    if args.path is not None:
        PATH = args.path

    
    env = SnakeEnv(False, 6, 6)
    agent = PPOAgent(env)

    timesteps = agent.load(PATH)

    timesteps += agent.learn(TRAINING_TIMESTEPS)
            

    # Save final model
    print('Saving final model...')

    agent.save(total_timesteps=timesteps, file_name='final_model.pth')

    print('Final model saved!')
    print('Training completed')