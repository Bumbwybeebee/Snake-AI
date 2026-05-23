import pygame
#import apple
import numpy as np
from enum import Enum, auto

class Direction(Enum):
    LEFT  = (-1,  0)
    RIGHT = ( 1,  0)
    UP    = ( 0, -1)
    DOWN  = ( 0,  1)

class SegmentType(Enum):
    HEAD_UP = "sprites/head_up.png"
    HEAD_DOWN = "sprites/head_down.png"
    HEAD_LEFT = "sprites/head_left.png"
    HEAD_RIGHT = "sprites/head_right.png"

    TAIL_UP = "sprites/tail_up.png"
    TAIL_DOWN = "sprites/tail_down.png"
    TAIL_LEFT = "sprites/tail_left.png"
    TAIL_RIGHT = "sprites/tail_right.png"

    VERTICAL = "sprites/body_vertical.png"
    HORIZONTAL = "sprites/body_horizontal.png"
    UP_LEFT = "sprites/body_topleft.png"
    UP_RIGHT = "sprites/body_topright.png"
    DOWN_LEFT = "sprites/body_bottomleft.png"
    DOWN_RIGHT = "sprites/body_bottomright.png"

class Snake:
    def __init__(self, res):
        self.length = 4
        self.has_eaten = False
        self.res = res
        self.alive = True
        self.has_turned = False
        
        x_position = res // 2
        y_position = res // 2

        self.direction = Direction.RIGHT
        self.snake_head = np.array([x_position, y_position])
        self.snake_body = [self.snake_head.copy(), np.array([x_position-1, y_position]), np.array([x_position-2, y_position]), np.array([x_position-3, y_position])]

    
    def grow(self):
        #self.length += 1
        self.has_eaten = True
        
    def move(self):
        self.has_turned = False
        self.snake_head += self.direction.value
        self.snake_body.insert(0, self.snake_head.copy())

        if self.has_eaten:
            self.length += 1
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

    def is_alive(self) -> bool:
        if np.any(self.snake_head < 0) or np.any(self.snake_head >= self.res):
            self.alive = False
        # 
        for segment in self.snake_body[1:]:
            if np.array_equal(segment, self.snake_head):
                self.alive = False
        return self.alive
    
    def find_segment_type(self, segment_index: int) -> SegmentType:

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
    
    def draw(self, screen: pygame.Surface, cell_size):
        # TODO: Alternating colored tiles
        for row in range(self.res):
            for col in range(self.res):
                if (row+col) % 2 == 0:
                    pygame.draw.rect(screen, (170,215,81), (col * cell_size, row * cell_size, cell_size, cell_size))
                else:
                    pygame.draw.rect(screen, (162, 209, 73), (col * cell_size, row * cell_size, cell_size, cell_size))

        for segment_index in range(len(self.snake_body)):
            segment_type = self.find_segment_type(segment_index).value
            self.snake_sprite = pygame.image.load(segment_type)
            screen.blit(self.snake_sprite, (self.snake_body[segment_index][0] * cell_size, self.snake_body[segment_index][1] * cell_size))