#!/usr/bin/env python3

import pygame
import sys
import snake
import apple
import ai
import numpy as np
import torch
import os
import game as snake_game
import random
import time
from collections import deque

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    RES = 15
    CELL_SIZE = 40
    WINDOW_SIZE = RES * CELL_SIZE
    DISPLAY = False
    SIMULTANEOUS_GAMES = 64

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    background = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    for row in range(RES):
        for col in range(RES):
            color = (170,215,81) if (row+col) % 2 == 0 else (162,209,73)
            pygame.draw.rect(background, color, (col*CELL_SIZE, row*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.set_caption("Snake")

    clock = pygame.time.Clock()
    print("start")

    games = []
    for i in range(SIMULTANEOUS_GAMES):
        games.append(snake_game.SnakeGame(True, RES))


    #AI stuff
    model = ai.Linear_QNet(input_size=11, hidden_size=256, output_size=3).to(device)
    saved_model_path = './model/best_model.pth'
    epsilon = 20 #increases variability at the beginning
    games_played = 0
    high_score = 0
    game_timestamps = deque()

    if os.path.exists(saved_model_path):
        checkpoint = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(checkpoint['state'])
        games_played = checkpoint['games_played']
        high_score = checkpoint['high_score']
        epsilon = checkpoint['epsilon']
        print("loaded saved model")
    else:
        print("no saved model found")
    trainer = ai.QTrainer(model=model, lr=0.001, gamma=0.9)
    agent = ai.Agent(model=model, trainer=trainer)
    running = True



    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        DISPLAY = not DISPLAY
                if event.type == pygame.QUIT:
                    running = False

            # Get states from all games
            states = []
            game_states = []
            for i, game in enumerate(games):
                state = game.get_state()
                states.append(state)
                game_states.append({
                    'index': i,
                    'state': state,
                    'game': game,
                    'old_head_pos': game.player_snake.snake_head.copy()
                })

            # Process all states in a batch to get actions
            if states and len(states) > 0:
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)

                with torch.no_grad():
                    predictions = model(states_tensor)
                    action_indices = torch.argmax(predictions, dim=1).cpu().tolist()

                # Exploration override
                for i in range(len(action_indices)):
                    if random.random() * 100 < epsilon:
                        action_indices[i] = random.randint(0, 2)

                # Apply actions and collect experiences
                games_finished = 0
                for i, (game_state, action_idx) in enumerate(zip(game_states, action_indices)):
                    game = game_state['game']
                    old_state = game_state['state']

                    action = [0, 0, 0]
                    action[action_idx] = 1

                    new_state, reward, done = game.step(action)

                    if game.high_score > high_score:
                        high_score = game.high_score

                    agent.remember(old_state, action, reward, new_state, done)

                    if done:
                        game.reset()
                        games_finished += 1
                        games_played += 1
                        epsilon = max(0, 20 * 0.9999**games_played)
                        current_time = time.time()
                        game_timestamps.append(current_time)
                        while game_timestamps and game_timestamps[0] < current_time - 2:
                            game_timestamps.popleft()

                        games_per_second = len(game_timestamps) / 2.0

                        print(f"Game {games_played} Over. Epsilon: {epsilon:.4f} High Score: {high_score} Speed: {games_per_second:.1f} games/sec")

                # FIX: Only train long memory when at least one game finishes!
                if games_finished > 0 and games_played % 50 == 0:
                    # print(f"{games_finished} games finished. Total games played: {games_played}")
                    if len(agent.memory) >= agent.batch_size:
                        # print("--> Training long memory...")
                        agent.train_long_memory()

            # FIX: Clean display handling that handles both AI monitoring and Single Player
            if DISPLAY:
                # Draw the background surface grid to clear old positions
                screen.blit(background, (0, 0))

                # Draw just the first game's snake and apple to monitor training visually
                games[0].player_snake.draw(screen, CELL_SIZE, background)
                games[0].player_apple.draw(screen, CELL_SIZE)
                clock.tick(60) # Limits speed while watching so you can see it

            pygame.display.flip()


    except KeyboardInterrupt:
        print("Program stopped by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        checkpoint = {
                'state': model.state_dict(),
                'games_played': games_played,
                'high_score': high_score,
                'epsilon': epsilon
            }
        model.save(checkpoint, file_name='best_model.pth')
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()