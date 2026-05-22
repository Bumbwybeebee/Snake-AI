import random
import numpy as np

class Apple:
    def __init__(self, res):
        x_position = random.randint(0, res-1)
        y_position = random.randint(0, res-1)
        self.apple_pos = np.array([x_position, y_position])

    def generate(self, snake):
        while True:
            x_position = random.randint(0, snake.res-1)
            y_position = random.randint(0, snake.res-1)
            potential_apple_pos = np.array([x_position, y_position])
            if not any(np.array_equal(potential_apple_pos, segment) for segment in snake.snake_body):
                break
        self.apple_pos = potential_apple_pos
    