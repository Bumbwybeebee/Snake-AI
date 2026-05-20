import numpy as np
from enum import Enum, auto

class Direction(Enum):
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, 1)
    DOWN = (0, -1)

class Snake:
    def __init__(self, res = 15):
        self.score = 0
        self.hasEaten = False
        self.res = res
        
        xPosition = res // 2
        yPosition = res // 2

        self.direction = Direction.RIGHT
        self.Headposition = np.array([xPosition, yPosition])
        self.snakeBody = [self.Headposition.copy(), np.array([xPosition-1, yPosition]), np.array([xPosition-2, yPosition])]

    def grow(self):
        self.score += 1
        self.hasEaten = True
        
    def move(self):
        self.Headposition += self.direction.value
        self.snakeBody.insert(0, self.Headposition.copy())

        if self.hasEaten:
            #don't remove last element in position, reset eaten
            self.hasEaten = False
            
        else:
            #remove last element in position
            self.snakeBody.pop()
