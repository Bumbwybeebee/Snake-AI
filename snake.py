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
        self.has_eaten = False
        self.res = res
        
        x_position = res // 2
        y_position = res // 2

        self.direction = Direction.RIGHT
        self.head_position = np.array([x_position, y_position])
        self.snake_body = [self.head_position.copy(), np.array([x_position-1, y_position]), np.array([x_position-2, y_position])]

    def grow(self):
        self.score += 1
        self.has_eaten = True
        
    def move(self):
        self.head_position += self.direction.value
        self.snake_body.insert(0, self.head_position.copy())

        if self.has_eaten:
            #don't remove last element in position, reset eaten
            self.has_eaten = False
        else:
            #remove last element in position
            self.snake_body.pop()
