# Script to visualize trained model playing

from agent.ppo_agent import PPOAgent
from env.snake_env import SnakeEnv
import os, time, argparse, torch, pygame


PATH = os.path.join('results', 'model', 'final_model.pth')
GRID_SIZE = 6


if __name__ == "__main__":
    # Handle params
    parser = argparse.ArgumentParser(description='trainig arguments')
    
    parser.add_argument('--path', type=str, default=None, help='Path to the model file (file name included). It can be both absolute or relative to project\'s root folder')
    parser.add_argument('--disable-gui', action='store_true', help='no value needed, deactivates gui on env (it will use the cli)')
    parser.add_argument('--grid-size', type=int, default=None, help='Number of timesteps the model will be trained for')
    args = parser.parse_args()
    
    if args.path is not None:
        PATH = args.path
    use_gui = not args.disable_gui
    if args.grid_size is not None:
        GRID_SIZE = args.grid_size


    # Initialize env and agent
    env = SnakeEnv(use_gui, 10, 10)
    agent = PPOAgent(env)

    # Load weights
    print('Loading weights...')
    agent.load(PATH)

    with torch.no_grad():
        # Play game
        obs, _ = env.reset()
        stop = False
        while not stop:
            action, log_prob = agent.get_action(obs, deterministic=True)
            obs, _, termin, trunc, _ = env.step(action)
            env.render()

            stop = termin or trunc
            time.sleep(0.2)
            
            if use_gui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        stop = True
                        exit()


        print('\nGame over')
        print(f'Score: {env.game.score}')

    # Close env
    env.close()