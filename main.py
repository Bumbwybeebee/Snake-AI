import pygame
import sys
import snake
import apple
import ai
import numpy as np

def main():
    res = 15
    cell_size = 40
    window_size = res * cell_size

    pygame.init()
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Snake")

    clock = pygame.time.Clock()
    print("start")
    player_snake = snake.Snake(res)
    player_apple = apple.Apple(res)

    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        player_snake.turn(snake.Direction.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        player_snake.turn(snake.Direction.RIGHT)
                    elif event.key == pygame.K_UP:
                        player_snake.turn(snake.Direction.UP)
                    elif event.key == pygame.K_DOWN:
                        player_snake.turn(snake.Direction.DOWN)

            player_snake.move()
            running = player_snake.is_alive()

            if np.array_equal(player_snake.snake_head, player_apple.apple_pos):
                player_snake.grow()
                player_apple.generate(player_snake)

            screen.fill((255, 255, 255))
            # TODO: Implement drawing methods for snake and apple
            player_snake.draw(screen, cell_size)
            player_apple.draw(screen, cell_size)

            pygame.display.flip()
            clock.tick(2)

    except KeyboardInterrupt:
        print("Program stopped by user.")
    # finally:
        # pygame.quit()
        # sys.exit()

if __name__ == "__main__":
    main()