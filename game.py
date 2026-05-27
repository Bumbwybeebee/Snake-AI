import pygame
import numpy as np
import snake
import apple


class SnakeGame:
    def __init__(self, res=15):
        self.RES = res
        self.high_score = 0
        self.starvation = 0
        self.player_snake = snake.Snake(res)
        self.player_apple = apple.Apple(res, display=False)

    def reset(self):
        self.player_snake = snake.Snake(self.RES)
        self.player_apple = apple.Apple(self.RES, display=False)
        self.starvation = 0

    def get_state(self):
        head      = self.player_snake.snake_head
        apple_pos = self.player_apple.apple_pos
        res       = self.RES

        grid = np.zeros((3, res, res), dtype=np.float32)
        for segment in self.player_snake.snake_body[1:]:
            grid[0, segment[0], segment[1]] = 1.0
        grid[1, head[0], head[1]] = 1.0
        grid[2, apple_pos[0], apple_pos[1]] = 1.0

        p_l = head + np.array([-1,  0])
        p_r = head + np.array([ 1,  0])
        p_u = head + np.array([ 0, -1])
        p_d = head + np.array([ 0,  1])

        dir_l = self.player_snake.direction == snake.Direction.LEFT
        dir_r = self.player_snake.direction == snake.Direction.RIGHT
        dir_u = self.player_snake.direction == snake.Direction.UP
        dir_d = self.player_snake.direction == snake.Direction.DOWN

        def danger(pt):
            if np.any(pt < 0) or np.any(pt >= res):
                return True
            for seg in self.player_snake.snake_body[1:]:
                if np.array_equal(seg, pt):
                    return True
            return False

        flat_state = [
            (dir_r and danger(p_r)) or (dir_l and danger(p_l)) or (dir_u and danger(p_u)) or (dir_d and danger(p_d)),
            (dir_u and danger(p_l)) or (dir_l and danger(p_d)) or (dir_d and danger(p_r)) or (dir_r and danger(p_u)),
            (dir_u and danger(p_r)) or (dir_l and danger(p_u)) or (dir_d and danger(p_l)) or (dir_r and danger(p_d)),
            dir_l, dir_r, dir_u, dir_d,
            head[0] / res, head[1] / res,
            apple_pos[0] < head[0],
            apple_pos[0] > head[0],
            apple_pos[1] < head[1],
            apple_pos[1] > head[1],
        ]
        flat = np.array([float(x) for x in flat_state], dtype=np.float32)
        return grid, flat

    def apply_action(self, action):
        """Apply action [straight, left, right] and move. Returns (ate_apple, dead)."""
        dx, dy = self.player_snake.direction.value
        if action == [0, 1, 0]:
            self.player_snake.turn(snake.Direction((int(dy), int(-dx))))
        elif action == [0, 0, 1]:
            self.player_snake.turn(snake.Direction((int(-dy), int(dx))))

        old_dist = (abs(self.player_snake.snake_head[0] - self.player_apple.apple_pos[0])
                  + abs(self.player_snake.snake_head[1] - self.player_apple.apple_pos[1]))

        self.player_snake.move()
        dead = not self.player_snake.is_alive()

        ate_apple = False
        if not dead and np.array_equal(self.player_snake.snake_head, self.player_apple.apple_pos):
            self.player_snake.grow()
            self.player_apple.generate(self.player_snake)
            self.high_score = max(self.high_score, self.player_snake.length)
            ate_apple = True

        new_dist = (abs(self.player_snake.snake_head[0] - self.player_apple.apple_pos[0])
                  + abs(self.player_snake.snake_head[1] - self.player_apple.apple_pos[1]))

        # Reward — identical to main.py
        if dead:
            reward = -1
            self.starvation = 0
        elif ate_apple:
            reward = 3
            self.starvation = 0
        else:
            self.starvation += 1
            if self.starvation > self.RES * self.RES:
                dead = True
                reward = -3
            elif new_dist < old_dist:
                reward = 0.1
            else:
                reward = 0

        return reward, dead

    def draw(self, screen, background, cell_size):
        screen.blit(background, (0, 0))
        self.player_snake.draw(screen, cell_size, background)
        self.player_apple.draw(screen, cell_size)