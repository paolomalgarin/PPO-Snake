# Logica snake
from enum import Enum
from typing import NamedTuple
import random, math
import pygame, os
import numpy as np


POINT_DTYPE = np.dtype([('x', int), ('y', int)])


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def getOpposite(self):
        match self:
            case Direction.UP:
                return Direction.DOWN
            case Direction.DOWN:
                return Direction.UP
            case Direction.LEFT:
                return Direction.RIGHT
            case Direction.RIGHT:
                return Direction.LEFT

    def getRightTurn(self):
        match self:
            case Direction.UP:
                return Direction.RIGHT
            case Direction.RIGHT:
                return Direction.DOWN
            case Direction.DOWN:
                return Direction.LEFT
            case Direction.LEFT:
                return Direction.UP

class CircArray:
    # IN THIS IMPLEMENTATION 
    # + HEAD IS THE POINTER TO THE 1st ELEMENT
    # + TAIL IS THE POINTER TO THE INDEX AFTER THE LAST
    # 
    # Eg:
    #   [ ][ ][ ][x][x][x][x][ ][ ]
    #             ^           ^
    #            Head        Tail

    def __init__(self, max_len: int, elm_dtype: np.dtype):
        self.dtype = elm_dtype
        
        self.max_len = max_len
        self.array = np.empty(self.max_len, dtype=self.dtype)

        self.head = 0
        self.tail = 0
        self.size = 0

    # -------------------- Add methods --------------------
    def tail_add(self, elm) -> bool:
        if(self.is_full()):
            return False

        self.array[self.tail] = np.array(elm, dtype=self.dtype)
        self.tail = (self.tail + 1) % self.max_len
        self.size += 1
        return True

    def head_add(self, elm) -> bool:
        if(self.is_full()):
            return False

        self.head = (self.max_len + self.head - 1) % self.max_len
        self.array[self.head] = np.array(elm, dtype=self.dtype)
        self.size += 1
        return True

    # -------------------- Remove methods --------------------
    def tail_remove(self) -> bool:
        if(self.is_empty()):
            return False

        self.tail = (self.max_len + self.tail - 1) % self.max_len
        self.size -= 1
        return True
    
    def head_remove(self) -> bool:
        if(self.is_empty()):
            return False

        self.head = (self.head + 1) % self.max_len
        self.size -= 1
        return True

    # -------------------- Other methods --------------------
    def clear(self):
        self.head = 0
        self.tail = 0
        self.size = 0

    def is_empty(self) -> bool:
        return self.size == 0

    def is_full(self) -> bool:
        return self.size == self.max_len
    
    def __len__(self) -> int:
        return self.size
    
    def __contains__(self, point) -> bool:
        # Case array empty
        if self.size == 0:
            return False
        
        # Case array full
        if self.size == self.max_len:
            return np.any(self.array == point)
        
        # General case
        if self.head <= self.tail:
            return np.any(self.array[self.head:self.tail] == point)
        else:
            part1 = np.any(self.array[self.head:] == point)
            part2 = np.any(self.array[:self.tail] == point)
            return part1 or part2
        
    def __iter__(self):
        if self.head <= self.tail:
            yield from self.array[self.head:self.tail]
        else:
            yield from self.array[self.head:]
            yield from self.array[:self.tail]

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError
        return self.array[(self.head + idx) % self.max_len]
        
    # -------------------- Helper methods --------------------
    def _get_contiguous_array(self) -> np.ndarray:
        # Returns a contiguous array made only of the array elements
        # (morphs the circ array in a normal array of the size of the elements) 
        if self.head <= self.tail:
            return self.array[self.head:self.tail]
        else:
            return np.concatenate((self.array[self.head:], self.array[:self.tail]))


class SnakeGame:
    
    def __init__(self, gridW: int = 10, gridH: int = 10, useGui = False, windowHeight = 680):
        self.gridWidth = gridW
        self.gridHeight = gridH

        self.head = np.array((0, 0), dtype=POINT_DTYPE)
        self.food = np.array((0, 0), dtype=POINT_DTYPE)

        body_max_len = self.gridHeight * self.gridWidth - 1
        self.body = CircArray(body_max_len, elm_dtype=POINT_DTYPE)
        
        self.spawnFood()

        self.direction = Direction.RIGHT

        self.score = 0
        self.isGameOver = False
        self.isGameWon = False

        if(useGui):
            pygame.init()

            gridWidthToHeightRatio = self.gridWidth/self.gridHeight
            screenHeight = windowHeight
            screenWidth = screenHeight * gridWidthToHeightRatio
            self.window = pygame.display.set_mode((screenWidth, screenHeight))
            
            self.squareSize = self.window.get_width() / self.gridWidth
            self.girdGrassTypes =  [random.randint(1, 5) for _ in range(self.gridHeight*self.gridWidth)]

            pygame.display.set_caption("PPO Snake")
            
            imagePath = os.path.join('.', 'env', 'assets', 'imgs')
            self.gameImgs = {
                "SNAKE": {
                    "HEAD": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'snake', 'head.png')), (self.squareSize, self.squareSize)),
                    "BODY": {
                        "HORIZONTAL": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'snake', 'body-horizontal.png')), (self.squareSize, self.squareSize)),
                        "VERTICAL": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'snake', 'body-vertical.png')), (self.squareSize, self.squareSize)),
                        "TURN": {
                            "LEFT-UP": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'snake', 'body-turn-left-up.png')), (self.squareSize, self.squareSize)),
                            "LEFT-DOWN": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'snake', 'body-turn-left-down.png')), (self.squareSize, self.squareSize)),
                            "UP-RIGHT": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'snake', 'body-turn-up-right.png')), (self.squareSize, self.squareSize)),
                            "DOWN-RIGHT": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'snake', 'body-turn-down-right.png')), (self.squareSize, self.squareSize)),
                        }
                    }
                },
                "GRASS": {
                    "TYPE-1": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'gnd', 'type-1.png')), (self.squareSize, self.squareSize)),
                    "TYPE-2": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'gnd', 'type-2.png')), (self.squareSize, self.squareSize)),
                    "TYPE-3": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'gnd', 'type-3.png')), (self.squareSize, self.squareSize)),
                    "TYPE-4": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'gnd', 'type-4.png')), (self.squareSize, self.squareSize)),
                    "TYPE-5": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'gnd', 'type-5.png')), (self.squareSize, self.squareSize)),
                },
                "FOOD": pygame.transform.scale(pygame.image.load(os.path.join(imagePath, 'food', 'food.png')), (self.squareSize, self.squareSize)),
            }
        else:
            self.window = None
            self.squareSize = 0
            self.girdGrassTypes = None
            self.gameImgs = None

    def spawnFood(self):
        # 100 attempts to find random spot
        max_attempts = 100
        for _ in range(max_attempts):
            randX = random.randint(0, self.gridWidth - 1)
            randY = random.randint(0, self.gridHeight - 1)
            
            candidate = np.array((randX, randY), dtype=POINT_DTYPE)
            if (not np.array_equal(self.head, candidate)) and (not self.body.__contains__(candidate)):
                self.food = candidate
                return


        # Find all free cells
        total_cells = self.gridHeight * self.gridWidth
        free_cell_len = total_cells - (1 + self.body.size)
        free_cells = np.empty(free_cell_len, dtype=POINT_DTYPE)
        free_cell_idx = 0
        for x in range(self.gridWidth):
            for y in range(self.gridHeight):    
                candidate = np.array((x, y), dtype=POINT_DTYPE)
                if (not np.array_equal(self.head, candidate)) and (not self.body.__contains__(candidate)):
                    free_cells[free_cell_idx] = candidate
                    free_cell_idx += 1

        # Chore random free cell
        if free_cell_len != 0:
            rand_idx = np.random.randint(free_cell_len)
            self.food = free_cells[rand_idx]
        else:
            self.food = np.array((-1, -1), dtype=POINT_DTYPE)
        
    def hittedWall(self):
        horizontalWallHit = self.head['x'] < 0 or self.head['x'] >= self.gridWidth
        verticalWallHit = self.head['y'] < 0 or self.head['y'] >= self.gridHeight
        return horizontalWallHit or verticalWallHit
    
    def hittedBody(self):
        if self.body.size == 0:
            return False
        
        return self.body.__contains__(self.head)

    def moveHead(self):
        # calcola la nuova posizione della testa
        x, y = self.head['x'], self.head['y']
        
        match self.direction:
            case Direction.UP:
                y -= 1
            case Direction.DOWN:
                y += 1
            case Direction.LEFT:
                x -= 1
            case Direction.RIGHT:
                x += 1
        
        # mette la testa in quella posizione
        self.head = np.array((x, y), dtype=POINT_DTYPE)

    def changeDir(self, dir: Direction):
        if(dir.getOpposite() == self.direction):
            return False
        else:
            self.direction = dir
            return True

    def move(self):
        # aggiungo la testa corrente nel body
        self.body.head_add(self.head)

        self.moveHead()

        if np.array_equal(self.head, self.food):
            self.spawnFood()
            self.score += 1
        else:
            self.body.tail_remove()
        
        if(self.hittedWall() or self.hittedBody()):
            self.isGameOver = True
        
        if(self.body.size == (self.gridHeight * self.gridWidth - 1)):
            self.isGameWon = True

    def displayCMD(self):
        gameString = " "

        for row in range(self.gridHeight):
            for col in range(self.gridWidth):
                currentPoint = np.array((col, row), dtype=POINT_DTYPE)
                
                if(currentPoint in self.body):
                    gameString += "▢ "
                elif np.array_equal(currentPoint, self.head):
                    gameString += "▣ "
                elif np.array_equal(currentPoint, self.food):
                    gameString += "◎ "
                else:
                    gameString += ". "
            gameString += "\n "
        
        print(gameString)

    def drawWindow(self):
        self.window.fill((30, 30, 30))
        squareSize = self.squareSize

        for row in range(self.gridHeight):
            for col in range(self.gridWidth):
                self.window.blit(self.gameImgs["GRASS"][f"TYPE-{self.girdGrassTypes[row*self.gridWidth + col]}"], pygame.Rect(col*squareSize, row*squareSize, squareSize, squareSize))
        
        match self.direction:
            case Direction.UP:
                headAngle = 0
            case Direction.RIGHT:
                headAngle = -90
            case Direction.LEFT:
                headAngle = 90
            case Direction.DOWN:
                headAngle = 180
             
        head = pygame.transform.rotate(self.gameImgs["SNAKE"]["HEAD"], headAngle)
        self.window.blit(head, pygame.Rect(self.head['x']*squareSize, self.head['y']*squareSize, squareSize, squareSize))

        for (i, elm) in enumerate(self.body):
            prev = self.head if (i == 0) else self.body[i-1]
            current = self.body[i]
            next = self.body[i] if (i == len(self.body) - 1) else self.body[i+1]
            
            # choosing piece type
            if(prev['x'] == current['x'] == next['x']):
                piece = self.gameImgs["SNAKE"]["BODY"]["HORIZONTAL"]
            elif(prev['y'] == current['y'] == next['y']):
                piece = self.gameImgs["SNAKE"]["BODY"]["VERTICAL"]
            else:
                if(prev['x'] < current['x'] and current['x'] == next['x']):
                    if(next['y'] > current['y']):
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["LEFT-DOWN"]
                    else:
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["LEFT-UP"]
                elif(prev['x'] > current['x'] and current['x'] == next['x']):
                    if(next['y'] > current['y']):
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["UP-RIGHT"]
                    else:
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["DOWN-RIGHT"]
                elif(prev['y'] > current['y'] and current['y'] == next['y']):
                    if(next['x'] > current['x']):
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["UP-RIGHT"]
                    else:
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["LEFT-DOWN"]
                else:
                    if(next['x'] > current['x']):
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["DOWN-RIGHT"]
                    else:
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["LEFT-UP"]

            self.window.blit(piece, pygame.Rect(current['x']*squareSize, current['y']*squareSize, squareSize, squareSize))

        self.window.blit(self.gameImgs["FOOD"], pygame.Rect(self.food['x']*squareSize, self.food['y']*squareSize, squareSize, squareSize))

        pygame.display.flip()

    def getFoodDistance(self):
        return math.sqrt(math.pow(self.head['x'] - self.food['x'], 2) + math.pow(self.head['y'] - self.food['y'], 2))
    
    def spawnRandomly(self):
        randX, randY = random.randint(0, self.gridWidth - 1), random.randint(0, self.gridHeight - 1)
        self.head = np.array((randX, randY), dtype=POINT_DTYPE)
        possibleDirs = []
        possibleDirs.append(Direction.LEFT if self.head['x'] > self.gridWidth/2 else Direction.RIGHT)
        possibleDirs.append(Direction.UP if self.head['y'] > self.gridHeight/2 else Direction.DOWN)
        self.direction = possibleDirs[random.randint(0, 1)]

    def reset(self):
        self.score = 0
        self.isGameOver = False
        self.isGameWon = False

        self.spawnRandomly()
        self.body.clear()
        self.spawnFood()

    def close(self):
        pygame.quit()

    def _grow(self, size):
        headX, headY = self.head['x'], self.head['y']
        
        x = (headX - 1) if (self.direction == Direction.RIGHT) else (headX + 1)if (self.direction == Direction.RIGHT) else headX
        y = (headY - 1) if (self.direction == Direction.DOWN) else (headY + 1)if (self.direction == Direction.UP) else headY
        
        new_elm = np.array((x, y), dtype=POINT_DTYPE)

        for i in range(size):
            self.body.tail_add(new_elm)