import random
import numpy as np

class Apple:
    def __init__(self, res):
        #res = r
        xPosition = random.randint(0, res)
        yPosition = random.randint(0, res)
        applePos = np.array([xPosition, yPosition])
    def generate(self, snake):
        while(True):
            xPosition = random.randint(0, self.res)
            yPosition = random.randint(0, self.res)
            False
            for i in snake.snake_body:
                if(np.array_equal(self.applePos, i)):
                    True
