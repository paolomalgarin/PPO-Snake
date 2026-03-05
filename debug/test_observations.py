from env.snake_env import SnakeEnv

env = SnakeEnv()

env.game.grow()
env.game.grow()
env.game.grow()

env.step(1)
env.step(1)
obs, _, _, _, _ = env.step(1)

env._print_obs(obs)