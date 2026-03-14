from env.snake_env import SnakeEnv

env = SnakeEnv()

env.game._grow(3)

env.step(1)
env.step(1)
obs, _, _, _, _ = env.step(1)

print(f'Head: {env.game.head}')
print(f'Food: {env.game.food}')

env._print_obs(obs)