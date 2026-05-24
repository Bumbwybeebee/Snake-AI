import random
import numpy as np
import snake
import pygame



class Apple:
    def __init__(self, res, display: bool):
        self.res = res
        self.x_position = 10
        self.y_position = 8
        self.apple_pos = np.array([self.x_position, self.y_position])
        if display:
            self.apple_sprite = pygame.image.load("sprites/apple.png").convert_alpha()

    def generate(self, snake: snake.Snake):
        all_positions = set((x, y) for x in range(self.res) for y in range(self.res))
        for segment in snake.snake_body:
            if tuple(segment) in all_positions:
                all_positions.remove(tuple(segment))
        self.x_position, self.y_position = random.choice(list(all_positions)) if all_positions else (0, 0)
        self.apple_pos = np.array([self.x_position, self.y_position])
        
    def draw(self, screen: pygame.Surface, size):
        screen.blit(self.apple_sprite, (self.x_position * size, self.y_position * size))