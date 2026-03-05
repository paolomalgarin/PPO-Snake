# Wrapper Gymnasium

from gymnasium import Env, spaces
import numpy as np
from env.snake_game import SnakeGame, Point, Direction


class SnakeEnv(Env):

    def __init__(self, useGui = False, gridW: int = 10, gridH: int = 10):
        self.game = SnakeGame(gridW, gridH, useGui=useGui)
        self.game.reset()
        
        self.useGui = useGui

        self.action_space = spaces.Discrete(
            4
        )  # 3 actions: up, down, right, left
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, self.game.gridHeight, self.game.gridWidth), dtype=np.float32
        )

        self.max_steps = self.game.gridHeight * self.game.gridWidth + 10
        self.steps = 0

        self.prev_score = self.game.score
        self.prev_food_distance = self.game.getFoodDistance()

    def reset(self, seed=None, options=None):
        self.game.reset()
        self.steps = 0
        self.prev_score = self.game.score
        self.prev_food_distance = self.game.getFoodDistance()
        self.max_steps = self.game.gridHeight * self.game.gridWidth + 10

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        self.steps += 1

        # 4 actions: 0 = up, 1 = down, 2 = right, 3 = left
        match action:
            case 0:
                newDir = Direction.UP
            case 1:
                newDir = Direction.DOWN
            case 2:
                newDir = Direction.RIGHT
            case 3:
                newDir = Direction.LEFT

        self.game.changeDir(newDir)
        self.game.move()


        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self.game.isGameOver or self.game.isGameWon
        truncated = self.steps >= self.max_steps

        info = {"score": self.game.score, "steps": self.steps}


        # Adding more steps if the snake eats
        if self.game.score > self.prev_score:
            self.max_steps = self.steps + self.game.gridHeight * self.game.gridWidth + 10
            self.prev_score = self.game.score

        return obs, reward, terminated, truncated, info

    def render(self):
        # renders the current game state
        if(not self.useGui):
            self.game.displayCMD()  # (temporary)
        else:
            self.game.drawWindow()

    def close(self):
        # closes pygame and widows
        self.game.close()

    def _get_obs(self):
        grid = np.zeros((self.observation_space.shape), dtype=np.float32)

        maxTailSize = self.game.gridWidth * self.game.gridHeight - 1

        for row in range(self.game.gridHeight):
            for col in range(self.game.gridWidth):
                currentPoint = Point(col, row)

                if currentPoint == self.game.head:
                    grid[0][row][col] = 1.0

                elif currentPoint in self.game.body:
                    reverse_idx = len(self.game.body) - 1 - self.game.body.index(currentPoint)
                    grid[1][row][col] = (maxTailSize - reverse_idx) / maxTailSize
                
                elif currentPoint == self.game.food:
                    grid[2][row][col] = 1.0

        # Set direction
        match self.game.direction:
            case Direction.UP:
                dir = (0, 0)
            case Direction.DOWN:
                dir = (1, 1)
            case Direction.RIGHT:
                dir = (1, 0)
            case Direction.LEFT:
                dir = (0, 1)

        dirX, dirY = dir
        grid[3][dirX][dirY] = 1.0

        return grid

    def _compute_reward(self):
        #   REWARD:
        #
        # + Food          100 - 1.5 * Get Closer
        # - Die           200
        # + Get closer    1
        #
        # - End game after max moves (resetted every apple eaten)   200
        #
        # Die > Food

        reward = 0

        
        
        if self.game.score > self.prev_score:
            # reward = (100 - self.steps * 1.5) if (100 - self.steps * 1.5 > 10) else 10
            reward = 1
            
        # elif (self.game.isGameOver or self.steps >= self.max_steps) and not (len(self.game.body) == self.game.gridHeight * self.game.gridWidth - 1):
        #     reward = -150

        # elif self.prev_food_distance > self.game.getFoodDistance():
        #     reward = 1
        
        # self.prev_food_distance = self.game.getFoodDistance()


        if(self.game.isGameWon):
            reward += self.game.gridWidth * self.game.gridHeight
        

        return float(reward)

    def _print_obs(self, obs):
        print("OBSERVATIONS:")
        
        print(obs.shape)

        for i in range(4):
            for j in range(self.game.gridHeight):
                for k in range(self.game.gridWidth):
                    toprint = (
                        "1"
                        if obs[i][j][k] == 1
                        else (
                            "."
                            if obs[i][j][k] == 0
                            else obs[i][j][k]
                        )
                    )
                    print(toprint, end=" ")
                print()
            print()
