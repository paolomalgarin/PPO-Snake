# Logica snake
from enum import Enum
from typing import NamedTuple
import random, math
import pygame, os


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

class Point(NamedTuple):
    x: int
    y: int


class SnakeGame:
    
    def __init__(self, gridW: int = 10, gridH: int = 10, useGui = False, windowHeight = 680):
        self.gridWidth = gridW
        self.gridHeight = gridH

        self.head = Point(0, 0)
        self.body = []
        self.food = None
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
        randX = -1
        randY = -1

        attempt = 0
        maxAttempts = 100
        found = False

        while not found: # simula un do-while perchè non esiste in python 
            randX = random.randint(0, self.gridWidth - 1)
            randY = random.randint(0, self.gridHeight - 1)
            candidate = Point(randX, randY)

            attempt += 1
            found = ((candidate not in self.body and candidate != self.head) or attempt >= maxAttempts) 

        if(attempt >= maxAttempts):
            # print("no free cell found to spawn food, spawning in occupied cell")
            pass

        self.food = Point(randX, randY)

    def hittedWall(self):
        horizontalWallHit = self.head.x < 0 or self.head.x >= self.gridWidth
        verticalWallHit = self.head.y < 0 or self.head.y >= self.gridHeight
        return horizontalWallHit or verticalWallHit
    
    def hittedBody(self):
        return self.head in self.body
    
    def grow(self):
        # inserisce un nuovo elemento alla fine del body
        # in modo che venga rimosso al posto dell'ultimo pezzo della coda quando chiamo move,
        # allungando il body
        self.body.append(Point(-1, -1)) # (aggiunta in coda)

    def moveHead(self):
        # calcola la nuova posizione della testa
        x, y = self.head.x, self.head.y
        
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
        self.head = Point(x, y)

    def changeDir(self, dir: Direction):
        if(dir.getOpposite() == self.direction):
            return False
        else:
            self.direction = dir
            return True

    def move(self):
        # aggiungo la testa corrente nel body
        self.body.insert(0, self.head) # (aggiunta in testa)

        self.moveHead()

        if(self.head == self.food):
            self.spawnFood()
            self.grow()
            self.score += 1

        self.body.pop() # (rimozione in coda)
        
        if(self.hittedWall() or self.hittedBody()):
            self.isGameOver = True
        
        if(len(self.body) == (self.gridHeight * self.gridWidth - 1)):
            self.isGameWon = True



    def displayCMD(self):
        gameString = " "

        for row in range(self.gridHeight):
            for col in range(self.gridWidth):
                currentPoint = Point(col, row)
                
                if(currentPoint in self.body):
                    gameString += "▢ "
                elif(currentPoint == self.head):
                    gameString += "▣ "
                elif(currentPoint == self.food):
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
        self.window.blit(head, pygame.Rect(self.head.x*squareSize, self.head.y*squareSize, squareSize, squareSize))

        for (i, elm) in enumerate(self.body):
            prev = self.head if (i == 0) else self.body[i-1]
            current = self.body[i]
            next = self.body[i] if (i == len(self.body) - 1) else self.body[i+1]
            
            # choosing piece type
            if(prev.x == current.x == next.x):
                piece = self.gameImgs["SNAKE"]["BODY"]["HORIZONTAL"]
            elif(prev.y == current.y == next.y):
                piece = self.gameImgs["SNAKE"]["BODY"]["VERTICAL"]
            else:
                if(prev.x < current.x and current.x == next.x):
                    if(next.y > current.y):
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["LEFT-DOWN"]
                    else:
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["LEFT-UP"]
                elif(prev.x > current.x and current.x == next.x):
                    if(next.y > current.y):
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["UP-RIGHT"]
                    else:
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["DOWN-RIGHT"]
                elif(prev.y > current.y and current.y == next.y):
                    if(next.x > current.x):
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["UP-RIGHT"]
                    else:
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["LEFT-DOWN"]
                else:
                    if(next.x > current.x):
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["DOWN-RIGHT"]
                    else:
                        piece = self.gameImgs["SNAKE"]["BODY"]["TURN"]["LEFT-UP"]

            self.window.blit(piece, pygame.Rect(current.x*squareSize, current.y*squareSize, squareSize, squareSize))

        self.window.blit(self.gameImgs["FOOD"], pygame.Rect(self.food.x*squareSize, self.food.y*squareSize, squareSize, squareSize))

        pygame.display.flip()


    def getFoodDistance(self):
        return math.sqrt(math.pow(self.head.x - self.food.x, 2) + math.pow(self.head.y - self.food.y, 2))
    
    def spawnRandomly(self):
        self.head = Point(random.randint(0, self.gridWidth - 1), random.randint(0, self.gridHeight - 1))
        possibleDirs = []
        possibleDirs.append(Direction.LEFT if self.head.x > self.gridWidth/2 else Direction.RIGHT)
        possibleDirs.append(Direction.UP if self.head.y > self.gridHeight/2 else Direction.DOWN)
        self.direction = possibleDirs[random.randint(0, 1)]

    def reset(self):
        # self.head = Point(0, 0)
        # self.direction = Direction.RIGHT
        self.spawnRandomly()

        self.score = 0
        self.isGameOver = False
        self.isGameWon = False

        while self.body.__len__() > 0:
            self.body.pop()

        self.spawnFood()

    def close(self):
        pygame.quit()