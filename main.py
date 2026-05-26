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
    DISPLAY = True
    AI_PLAYING = True
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

    if AI_PLAYING:
        games = []
        for i in range(SIMULTANEOUS_GAMES):
            games.append(snake_game.SnakeGame(True, False, RES))
    else:
        player_snake = snake.Snake(RES)
        player_apple = apple.Apple(RES, DISPLAY)


    #AI stuff
    model = ai.Linear_QNet(input_size=11, hidden_size=512, output_size=3).to(device)
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

            if AI_PLAYING:
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
            else:
                # Single player game processing placeholder
                pass

            # FIX: Clean display handling that handles both AI monitoring and Single Player
            if DISPLAY:
                # Draw the background surface grid to clear old positions
                screen.blit(background, (0, 0))

                if AI_PLAYING:
                    # Draw just the first game's snake and apple to monitor training visually
                    games[0].player_snake.draw(screen, CELL_SIZE, background)
                    games[0].player_apple.draw(screen, CELL_SIZE)
                    clock.tick(60) # Limits speed while watching so you can see it
                else:
                    player_snake.draw(screen, CELL_SIZE, background)
                    player_apple.draw(screen, CELL_SIZE)
                    clock.tick(10)

                pygame.display.flip()


    except KeyboardInterrupt:
        print("Program stopped by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if AI_PLAYING:
            checkpoint = {
                    'state': model.state_dict(),
                    'games_played': games_played,
                    'high_score': high_score,
                    'epsilon': epsilon
                }
            model.save(checkpoint, file_name='best_model.pth')
        pygame.quit()
        sys.exit()

def ai_move(player_snake : snake.Snake, final_move):
    current_dir = player_snake.direction.value
    v = np.array(current_dir)
    # print(current_dir)
    # print(v)
    # print(snake.Direction((v[1], v[0])))
    if final_move[0] == 1: #ai keeps moving forwards
        pass
    elif final_move[1] == 1: #ai chooses to turn left
        player_snake.turn(snake.Direction((int(v[1]), int(-v[0])))) #should output the direction snake is moving rotated by 90 degrees counterclockwise
    else: #ai turns right
        player_snake.turn(snake.Direction((int(-v[1]), int(v[0]))))#should output the direction snake is moving rotated by 90 degrees clockwise

if __name__ == "__main__":
    main()

# def get_game_state(player_snake : snake.Snake, player_apple : apple.Apple, res):
    # centered_map = np.zeros((res, res))
    # center = (res // 2, res // 2)
    # shift_x, shift_y = player_snake.snake_head - center
    # # print(f"shifted coordinates: ({shift_x}, {shift_y})")
    # for segment in player_snake.snake_body:
    #     x, y = segment
    #     shifted_x = (x - shift_x) % res
    #     shifted_y = (y - shift_y) % res
    #     centered_map[shifted_y, shifted_x] = 1

    # apple_x, apple_y = player_apple.apple_pos
    # apple_shifted_x = (apple_x - shift_x) % res
    # apple_shifted_y = (apple_y - shift_y) % res
    # centered_map[apple_shifted_y, apple_shifted_x] = -1

    # apple_pos = player_apple.apple_pos
    # # print(f"centered map:")
    # # print(f"{centered_map}")
    # dir_l = 1 if player_snake.direction == snake.Direction.LEFT else 0
    # dir_r = 1 if player_snake.direction == snake.Direction.RIGHT else 0
    # dir_u = 1 if player_snake.direction == snake.Direction.UP else 0
    # dir_d = 1 if player_snake.direction == snake.Direction.DOWN else 0
    # final_map = np.concatenate([centered_map.flatten(), [apple_pos[0], apple_pos[1]], [dir_l, dir_r, dir_u, dir_d]])
    # # state_map = np.zeros((res*res*2 + 4))
    # # # head_map = np.zeros((res*res))
    # # head_index = player_snake.snake_head[1] * res + player_snake.snake_head[0]

    # # if 0 <= head_index < res*res:
    # #     state_map[head_index] = 1

    # # # snake_map = np.zeros((res*res))
    # # for segment in player_snake.snake_body:
    # #     index = segment[1] * res + segment[0]
    # #     if 0 <= index < res * res:
    # #         state_map[index + res*res] = 1

    # # apple_index = player_apple.apple_pos[1] * res + player_apple.apple_pos[0]
    # # state_map[apple_index + res*res] = -1

    # # state_map[res*res*2 + 0] = 1 if player_snake.direction == snake.Direction.LEFT else 0
    # # state_map[res*res*2 + 1] = 1 if player_snake.direction == snake.Direction.RIGHT else 0
    # # state_map[res*res*2 + 2] = 1 if player_snake.direction == snake.Direction.UP else 0
    # # state_map[res*res*2 + 3] = 1 if player_snake.direction == snake.Direction.DOWN else 0

    # return final_map