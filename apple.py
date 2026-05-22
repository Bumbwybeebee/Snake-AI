import random
import numpy as np
import snake
import pygame

class Apple:
    def __init__(self, res):
        self.res = res
        self.x_position = 10#random.randint(0, self.res-1)
        self.y_position = 8#random.randint(0, self.res-1)
        self.apple_pos = np.array([self.x_position, self.y_position])
        self.apple_sprite = pygame.image.load("sprites/apple.png").convert_alpha()

    def generate(self, snake: snake.Snake):
        while True:
            self.x_position = random.randint(0, snake.res-1)
            self.y_position = random.randint(0, snake.res-1)
            potential_apple_pos = np.array([self.x_position, self.y_position])
            if not any(np.array_equal(potential_apple_pos, segment) for segment in snake.snake_body):
                break
        self.apple_pos = potential_apple_pos
        
    def draw(self, screen: pygame.Surface, size):
        screen.blit(self.apple_sprite, (self.x_position * size, self.y_position * size))