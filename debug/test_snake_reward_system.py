# script to play the game using the SnakeGame class
# (made to test the game logic)

from pathlib import Path
import sys
import argparse
from env.snake_env import SnakeEnv


if __name__ == "__main__":
    # ------------------------ game setup ------------------------
    game = SnakeEnv(False,6,6)


    # ------------------------ game logic (debug part) ------------------------
    game.reset()
    game.render()

    isGameOver = False
    tot_rew = 0
    while not isGameOver:
        # Get input
        user_input = input("Move (a=left, d=right, w=up, s=down, q=quit): ").lower()
        
        print("\n\n")
        
        move = -1
        match user_input:
            case 'w':
                move = 0
            case 's':
                move = 1
            case 'd':
                move = 2
            case 'a':
                move = 3
            case 'q':
                break
            case _:
                continue  # Invalid input, skip move
        
        obs, reward, terminated, truncated, info = game.step(move)
        isGameOver = terminated or truncated
        tot_rew += reward

        game.render()
        print(f"Step Reward: {reward}")
        print(f"Steps: {game.steps}/{game.max_steps}")

    print("=== GAME OVER! ===")
    print("[Score " + str(game.game.score) + "]\n")
    print("[Tor reward " + str(tot_rew) + "]\n")

    game.close()