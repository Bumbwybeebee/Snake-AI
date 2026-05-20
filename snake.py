import numpy as np
from enum import Enum, auto

class Direction(Enum):
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP = (0, 1)
    DOWN = (0, -1)

class Snake:
    def __init__(self, res = 15):
        self.length = 3
        self.has_eaten = False
        self.res = res
        
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
            #don't remove last element in position, reset eaten
            self.has_eaten = False
        else:
            #remove last element in position
            self.snake_body.pop()

    def rotate(self, attempted_direction):
        if (
            (attempted_direction == Direction.LEFT and self.direction != Direction.RIGHT)
            or (attempted_direction == Direction.RIGHT and self.direction != Direction.LEFT)
            or (attempted_direction == Direction.UP and self.direction != Direction.DOWN)
            or (attempted_direction == Direction.DOWN and self.direction != Direction.UP)
        ):
            self.direction = attempted_direction
    
    def is_dead(self) -> bool:
        if np.any(self.snake_head < 0) or np.any(self.snake_head >= self.res):
            return True
        
        for segment in self.snake_body[1:]:
            if np.array_equal(segment, self.snake_head):
                return True
            
        return False