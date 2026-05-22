import pygame
import sys
import snake
import apple
import ai

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
        while running and player_snake.alive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            player_snake.move()

            if player_snake.snake_head == player_apple.apple_pos:
                player_snake.grow()
                player_apple.generate(player_snake)

            screen.fill((0, 0, 0))
            # TODO: Implement drawing methods for snake and apple
            # player_snake.draw(screen)
            # player_apple.draw(screen)

            pygame.display.flip()
            clock.tick(5)

    except KeyboardInterrupt:
        print("Program stopped by user.")
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()