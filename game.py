import snake
import apple
import numpy as np

class SnakeGame:
    def __init__(self, AI_PLAYING, DISPLAY, res=15):
        self.RES = res
        self.AI_PLAYING = AI_PLAYING
        self.DISPLAY = DISPLAY
        self.player_snake = snake.Snake(res)
        self.player_apple = apple.Apple(res, display=False)
        self.high_score = 0
        self.reset()

    def reset(self):
        self.player_snake = snake.Snake(self.RES)
        self.player_apple = apple.Apple(self.RES, display=False)
        # Keep the high score when resetting the game
        # Only reset the snake and apple, not the high score
        # self.high_score should be preserved across games

    def get_state(self):
        return self.get_game_state(self.player_snake, self.player_apple, self.RES)

    def step(self, action):
        old_head_pos = self.player_snake.snake_head.copy()
        old_dist = abs(old_head_pos[0] - self.player_apple.apple_pos[0]) + abs(old_head_pos[1] - self.player_apple.apple_pos[1])
        current_dir_x, current_dir_y = self.player_snake.direction.value
        if action == [1, 0, 0]:
            #straight
            pass
        elif action == [0, 1, 0]:
            #left
            self.player_snake.turn(snake.Direction((int(current_dir_y), int(-current_dir_x)))) #should output the direction snake is moving rotated by 90 degrees counterclockwise
        elif action == [0, 0, 1]:
            #right
            self.player_snake.turn(snake.Direction((int(-current_dir_y), int(current_dir_x))))#should output the direction snake is moving rotated by 90 degrees clockwise

        self.player_snake.move()

        done = not self.player_snake.is_alive()
        reward = 0

        # Calculate reward
        if done:
            reward = -50
        else:
            new_dist = abs(self.player_snake.snake_head[0] - self.player_apple.apple_pos[0]) + abs(self.player_snake.snake_head[1] - self.player_apple.apple_pos[1])
            if np.array_equal(self.player_snake.snake_head, self.player_apple.apple_pos):
                self.player_snake.grow()
                self.player_apple.generate(self.player_snake)
                # Update high score to the maximum of current high score and current snake length
                self.high_score = max(self.high_score, self.player_snake.length)
                reward = 10
            elif new_dist < old_dist:
                reward = 1
            else:
                reward = -1

        return self.get_game_state(self.player_snake, self.player_apple, self.RES), reward, done

    def get_game_state(self, player_snake : snake.Snake, player_apple : apple.Apple, res):
        head = player_snake.snake_head
        apple_pos = player_apple.apple_pos

        #coords of the 4 points around head
        p_l = head + np.array([-1, 0])
        p_r = head + np.array([1, 0])
        p_u = head + np.array([0, -1])
        p_d = head + np.array([0, 1])

        #bool values for which direction is the snake currently going, formatted for input array
        dir_l = player_snake.direction == snake.Direction.LEFT
        dir_r = player_snake.direction == snake.Direction.RIGHT
        dir_u = player_snake.direction == snake.Direction.UP
        dir_d = player_snake.direction == snake.Direction.DOWN


        #checks if the snake is in danger
        def danger(pt):
            #wall collisions
            if np.any(pt < 0) or np.any(pt >= res):
                return True
            #
            for segment in player_snake.snake_body[1:]:
                if np.array_equal(segment, pt):
                    return True
            return False
        state = [
            #danger values
            #going straight is deadly
            (dir_r and danger(p_r)) or (dir_l and danger(p_l)) or (dir_u and danger(p_u)) or (dir_d and danger(p_d)),
            #turning left is deadly
            (dir_u and danger(p_l)) or (dir_l and danger(p_d)) or (dir_d and danger(p_r)) or (dir_r and danger(p_u)),
            #turning right is deadly
            (dir_u and danger(p_r)) or (dir_l and danger(p_u)) or (dir_d and danger(p_l)) or (dir_r and danger(p_d)),

            #current direction
            dir_l, dir_r, dir_u, dir_d,

            #food location relative to head position
            apple_pos[0] < head[0],  # Food is Left
            apple_pos[0] > head[0],  # Food is Right
            apple_pos[1] < head[1],  # Food is Up
            apple_pos[1] > head[1]   # Food is Down
        ]
        return [int(x) for x in state]