# Wrapper Gymnasium

from gymnasium import Env, spaces
import numpy as np
from env.snake_game import SnakeGame, Direction


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

    def reset(self, seed=None, options=None):
        self.game.reset()
        self.steps = 0
        self.prev_score = self.game.score
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


        # Head
        headX, headY = self.game.head
        if((headX >= 0 and headX < self.game.gridWidth) and (headY >= 0 and headY < self.game.gridHeight)):
            grid[0][headY][headX] = 1.0

        # Body
        body_len = len(self.game.body)
        maxTailSize = self.game.gridWidth * self.game.gridHeight - 1

        for i, point in enumerate(self.game.body):
            reverse_idx = body_len - 1 - i
            grid[1][point.y][point.x] = (maxTailSize - reverse_idx) / maxTailSize
        
        # Food
        foodX, foodY = self.game.food
        if(foodX != -1 and foodY != -1):
            grid[2][foodY][foodX] = 1.0


        # Set direction
        match self.game.direction:
            case Direction.UP:
                grid[3][0][0] = 1.0
            case Direction.DOWN:
                dir = (1, 1)
                grid[3][1][1] = 1.0
            case Direction.RIGHT:
                grid[3][1][0] = 1.0
            case Direction.LEFT:
                grid[3][0][1] = 1.0

        return grid


    def _compute_reward(self):
        #   REWARD:
        #
        # - Die                                                     [ -1  ]
        # - End game after max moves (resetted every apple eaten)   [ -1  ]
        # + Eat food                                                [ +1  ]
        # + Win game                                                [ +30 ]

        reward = 0

        
        if self.game.score > self.prev_score:
            reward = 1
            
        elif (self.game.isGameOver or self.steps >= self.max_steps) and not (len(self.game.body) == self.game.gridHeight * self.game.gridWidth - 1):
            reward = -1

        if(self.game.isGameWon):
            reward += 30
        

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
                            else f'{obs[i][j][k]:.3f}'
                        )
                    )
                    print(toprint, end=" ")
                print()
            print()
