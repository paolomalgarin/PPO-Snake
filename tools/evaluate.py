# Script to visualize trained model playing

from agent.ppo_agent import PPOAgent
from env.snake_env import SnakeEnv
import os, time, argparse, torch, pygame
from tools.beautyful_progress_bar import PBar


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
    env = SnakeEnv(False, GRID_SIZE, GRID_SIZE)
    agent = PPOAgent(env)

    # Load weights
    print('Loading weights...')
    agent.load(PATH)

    with torch.no_grad():
        print('\n\nEvaluation (press CTRL + C if you don\'t want it):')

        mean_rew_eval = 0
        mean_score_eval = 0
        mean_steps_eval = 0
        wins = 0
        max_rew = env.game.gridHeight * env.game.gridWidth - 1
        eval_games = 1000

        eval_pbar = PBar(eval_games, 'Eval games', preset='eval')

        for i in range(eval_games):
            stop = False
            tot_reward = 0
            obs, info = env.reset()

            while not stop:
                # Chose an action
                action, _ = agent.get_action(obs, deterministic=True)

                # Perform action
                obs, reward, terminated, truncated, info = env.step(action)
                stop = terminated or truncated

                tot_reward += reward

            # Visualize results
            mean_rew_eval += tot_reward
            mean_score_eval += info['score']
            mean_steps_eval += info['steps']
            
            if (info['score'] == max_rew):
                wins += 1

            eval_pbar.update(1)
        
        eval_pbar.close()

        mean_rew_eval = mean_rew_eval / eval_games
        mean_score_eval = mean_score_eval / eval_games
        mean_steps_eval = mean_steps_eval / eval_games

        print(f'Games played: \t{eval_games}')
        print(f'Wins: \t{wins}/{eval_games} [{(wins/eval_games)*100:.2f}]')
        print(f'Mean reward: \t{mean_rew_eval:.2f}')
        print(f'Mean score: \t{mean_score_eval:.2f}')
        print(f'Mean steps: \t{mean_steps_eval:.2f}')

    # Close env
    env.close()