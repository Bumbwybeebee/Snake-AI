#!/usr/bin/env python3
import pygame
import sys
import torch
import os
import numpy as np

import snake
import game as snake_game

KEY_TO_DIRECTION = {
    pygame.K_LEFT:  snake.Direction.LEFT,
    pygame.K_a:     snake.Direction.LEFT,
    pygame.K_RIGHT: snake.Direction.RIGHT,
    pygame.K_d:     snake.Direction.RIGHT,
    pygame.K_UP:    snake.Direction.UP,
    pygame.K_w:     snake.Direction.UP,
    pygame.K_DOWN:  snake.Direction.DOWN,
    pygame.K_s:     snake.Direction.DOWN,
}

def main():
    AI_PLAYING = True

    RES         = 15
    CELL_SIZE   = 40
    WINDOW_SIZE = RES * CELL_SIZE
    FPS         = 60 if AI_PLAYING else 10

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    background = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    for row in range(RES):
        for col in range(RES):
            color = (170, 215, 81) if (row + col) % 2 == 0 else (162, 209, 73)
            pygame.draw.rect(background, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    clock = pygame.time.Clock()
    print("--------------- Starting ---------------")

    if AI_PLAYING:
        import ai
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        model = ai.Conv_QNet(res=RES, flat_input_size=13, hidden_size=512, output_size=3).to(device)
        model_path = 'model/best_model.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['state'])
            model.eval()
            print(f"Loaded model — Games trained: {checkpoint['games_played']}")
        else:
            print("No trained model found at", model_path)
            return

    game = snake_game.SnakeGame(RES)
    alive = True
    direction = snake.Direction.RIGHT

    while alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                alive = False
            elif event.type == pygame.KEYDOWN:
                if not AI_PLAYING:
                    direction = KEY_TO_DIRECTION.get(event.key, direction)

        if AI_PLAYING:
            grid, flat = game.get_state()
            with torch.no_grad():
                grid_t = torch.tensor(grid, dtype=torch.float).unsqueeze(0).to(device)
                flat_t = torch.tensor(flat, dtype=torch.float).unsqueeze(0).to(device)
                move_idx = int(torch.argmax(model(grid_t, flat_t)).item())
            action = [0, 0, 0]
            action[move_idx] = 1
            _, done = game.apply_action(action)
        else:
            game.player_snake.turn(direction)
            game.player_snake.move()
            done = not game.player_snake.is_alive()
            if not done and np.array_equal(game.player_snake.snake_head, game.player_apple.apple_pos):
                game.player_snake.grow()
                game.player_apple.generate(game.player_snake)

        if done:
            alive = False

        pygame.display.set_caption(f"Snake - Score: {game.player_snake.length}")
        game.draw(screen, background, CELL_SIZE)
        pygame.display.flip()
        clock.tick(FPS)

    print(f"--------- Game Over! Score: {game.player_snake.length} ---------")
    pygame.quit()

if __name__ == "__main__":
    main()