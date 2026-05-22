import apple
import numpy as np
from enum import Enum, auto

class Direction(Enum):
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, 1)
    DOWN = (0, -1)

class Snake:
    def __init__(self, res):
        self.length = 3
        self.has_eaten = False
        self.res = res
        self.alive = True
        self.has_turned = False
        
        x_position = res // 2
        y_position = res // 2

        self.direction = Direction.RIGHT
        self.snake_head = np.array([x_position, y_position])
        self.snake_body = [self.snake_head.copy(), np.array([x_position-1, y_position]), np.array([x_position-2, y_position])]

    
    def grow(self):
        self.length += 1
        self.has_eaten = True
        
    def move(self):
        self.snake_head += self.direction.value
        self.snake_body.insert(0, self.snake_head.copy())

        if self.has_eaten:
            self.has_eaten = False
        else:
            self.snake_body.pop()

    def turn(self, attempted_direction):
        if not self.has_turned:
            if (
                (attempted_direction == Direction.LEFT and self.direction != Direction.RIGHT)
                or (attempted_direction == Direction.RIGHT and self.direction != Direction.LEFT)
                or (attempted_direction == Direction.UP and self.direction != Direction.DOWN)
                or (attempted_direction == Direction.DOWN and self.direction != Direction.UP)
            ):
                self.direction = attempted_direction
                self.has_turned = True

    def is_dead(self):
        if np.any(self.snake_head < 0) or np.any(self.snake_head >= self.res):
            self.alive = False
        
        for segment in self.snake_body[1:]:
            if np.array_equal(segment, self.snake_head):
                self.alive = False
    