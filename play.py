#!/usr/bin/env python3
import pygame
import sys
from enum import Enum
import torch
import os
import numpy as np

import snake
import apple
import game


def main():
    AI_PLAYING = True

    RES = 15
    CELL_SIZE = 40
    WINDOW_SIZE = RES*CELL_SIZE
    FPS = 60 if AI_PLAYING else 10

    player_game = game.SnakeGame(AI_PLAYING, RES)
    alive = True
    current_score = 4
    direction = snake.Direction.RIGHT

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    background = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    for row in range(RES):
        for col in range(RES):
            color = (170,215,81) if (row+col) % 2 == 0 else (162,209,73)
            pygame.draw.rect(background, color, (col*CELL_SIZE, row*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    clock = pygame.time.Clock()
    print("--------------- Starting ---------------")

    if AI_PLAYING:
        import ai
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ai.Linear_QNet(input_size=11, hidden_size=256, output_size=3).to(device)
        model_path = 'model/best_model.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['state'])
            model.eval()
            print(f"Loaded model — Games trained: {checkpoint['games_played']}")
        else:
            print("No trained model found at", model_path)
            return

    while alive and AI_PLAYING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                alive = False

        state = player_game.get_state()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            move_idx = int(torch.argmax(model(state_tensor)).item())
        final_move = [0, 0, 0]
        final_move[move_idx] = 1

        _, _, done = player_game.step(final_move)
        alive = not done

        current_score = player_game.player_snake.length
        pygame.display.set_caption(f"Snake - Score: {current_score}")

        player_game.draw(screen, background, CELL_SIZE)
        pygame.display.flip()
        clock.tick(FPS)

    while alive and not AI_PLAYING:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                alive = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    direction = snake.Direction.LEFT
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    direction = snake.Direction.RIGHT
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    direction = snake.Direction.UP
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    direction = snake.Direction.DOWN

        alive = player_game.step(direction)
        current_score = player_game.player_snake.length
        pygame.display.set_caption(f"Snake - Score: {current_score}")


        player_game.draw(screen, background, CELL_SIZE)
        pygame.display.flip()
        clock.tick(FPS)

    print(f"--------- Game Over! Score: {player_game.player_snake.length} ---------")
    pygame.quit()

if __name__ == "__main__":
    main()