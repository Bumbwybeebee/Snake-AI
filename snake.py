import pygame
import apple
import numpy as np
from enum import Enum, auto

class Direction(Enum):
    LEFT  = (-1,  0)
    RIGHT = ( 1,  0)
    UP    = ( 0,  1)
    DOWN  = ( 0, -1)

class SegmentType(Enum):
    HEAD_UP = auto()
    HEAD_DOWN = auto()
    HEAD_LEFT = auto()
    HEAD_RIGHT = auto()

    TAIL_UP = auto()
    TAIL_DOWN = auto()
    TAIL_LEFT = auto()
    TAIL_RIGHT = auto()

    VERTICAL = auto()
    HORIZONTAL = auto()
    UP_LEFT = auto()
    UP_RIGHT = auto()
    DOWN_LEFT = auto()
    DOWN_RIGHT = auto()
    

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

    def turn(self, attempted_direction: Direction):
        if not self.has_turned:
            if (
                   (attempted_direction == Direction.LEFT and  self.direction != Direction.RIGHT)
                or (attempted_direction == Direction.RIGHT and self.direction != Direction.LEFT)
                or (attempted_direction == Direction.UP and    self.direction != Direction.DOWN)
                or (attempted_direction == Direction.DOWN and  self.direction != Direction.UP)
            ):
                self.direction = attempted_direction
                self.has_turned = True

    def is_dead(self):
        if np.any(self.snake_head < 0) or np.any(self.snake_head >= self.res):
            self.alive = False
        
        for segment in self.snake_body[1:]:
            if np.array_equal(segment, self.snake_head):
                self.alive = False
    
    def find_segment_type(self, segment_index) -> SegmentType:

        if segment_index == 0:
            head_vector = tuple(self.snake_head - self.snake_body[1])

            if head_vector == Direction.UP.value:
                return SegmentType.HEAD_UP
            elif head_vector == Direction.DOWN.value:
                return SegmentType.HEAD_DOWN
            elif head_vector == Direction.LEFT.value:
                return SegmentType.HEAD_LEFT
            elif head_vector == Direction.RIGHT.value:
                return SegmentType.HEAD_RIGHT
            
        elif segment_index == self.length - 1:
            tail_vector = tuple(self.snake_body[segment_index - 1] - self.snake_body[segment_index])

            if tail_vector == Direction.UP.value:
                return SegmentType.TAIL_UP
            elif tail_vector == Direction.DOWN.value:
                return SegmentType.TAIL_DOWN
            elif tail_vector == Direction.LEFT.value:
                return SegmentType.TAIL_LEFT
            elif tail_vector == Direction.RIGHT.value:
                return SegmentType.TAIL_RIGHT

        prev_segment =    self.snake_body[segment_index - 1]
        current_segment = self.snake_body[segment_index]
        next_segment =    self.snake_body[segment_index + 1]

        prev_vector = tuple(self.snake_body[segment_index - 1] - self.snake_body[segment_index])
        next_vector = tuple(self.snake_body[segment_index + 1] - self.snake_body[segment_index])

        neighbor_vectors = {prev_vector, next_vector}

        if prev_segment[0] == next_segment[0]:
            return SegmentType.VERTICAL
        elif prev_segment[1] == next_segment[1]:
            return SegmentType.HORIZONTAL
        elif neighbor_vectors == {Direction.UP.value, Direction.LEFT.value}:
            return SegmentType.UP_LEFT
        elif neighbor_vectors == {Direction.UP.value, Direction.RIGHT.value}:
            return SegmentType.UP_RIGHT
        elif neighbor_vectors == {Direction.DOWN.value, Direction.LEFT.value}:
            return SegmentType.DOWN_LEFT
        elif neighbor_vectors == {Direction.DOWN.value, Direction.RIGHT.value}:
            return SegmentType.DOWN_RIGHT
        
        # so that there are no errors about it not always returning something
        return SegmentType.HORIZONTAL
    
    def draw(self, screen: pygame.Surface):
        pass