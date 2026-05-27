#!/usr/bin/env python3

import pygame
import sys
import snake
import apple
import ai
import numpy as np
import torch
import random
import os
import json
from collections import deque

# Set device for computation
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    RES = 15
    CELL_SIZE = 40
    WINDOW_SIZE = RES * CELL_SIZE
    DISPLAY = True
    AI_PLAYING = True
    starvation = 0

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
    player_snake = snake.Snake(RES)
    player_apple = apple.Apple(RES, DISPLAY)

    #AI stuff
    # model = ai.Linear_QNet(input_size=11, hidden_size=256, output_size=3).to(device)
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = ai.Conv_QNet(res=RES, flat_input_size=13, hidden_size=512, output_size=3).to(device)
    saved_model_path = './model/best_model.pth'
    stats_path = './model/stats.json'
    epsilon = 200 #increases variability at the beginning
    games_played = 0
    high_score = 0
    if os.path.exists(saved_model_path):
        checkpoint = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(checkpoint['state'])
        games_played = checkpoint['games_played']
        high_score = checkpoint['high_score']
        epsilon = checkpoint['epsilon']
        print("loaded saved model")
    else:
        print("no saved model found")
    os.makedirs('./model', exist_ok=True)
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        avg_score = stats['avg_score']
        recent_scores = deque(stats['recent_scores'], maxlen=2000)
    else:
        avg_score = 0
        recent_scores = deque(maxlen=2000)
    trainer = ai.QTrainerCNN(model=model, lr=0.001, gamma=0.99)
    agent = ai.AgentCNN(model=model, trainer=trainer)
    running = True

    

    try:
        while running: 
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        DISPLAY = not DISPLAY 
                    if not AI_PLAYING:
                        if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                            player_snake.turn(snake.Direction.LEFT)
                        elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                            player_snake.turn(snake.Direction.RIGHT)
                        elif event.key == pygame.K_UP or event.key == pygame.K_w:
                            player_snake.turn(snake.Direction.UP)
                        elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                            player_snake.turn(snake.Direction.DOWN)
                if event.type == pygame.QUIT:
                    running = False
                    
            #AI input here
            #stores old state for comparison
            old_grid, old_flat = get_game_state(player_snake=player_snake, player_apple=player_apple, res=RES)
            #choosing action
            if AI_PLAYING:
                final_move = [0,0,0]
                if random.randint(0, 200) < epsilon:
                    final_move[random.randint(0,2)] = 1
                else:
                    with torch.no_grad():
                        grid_tensor = torch.tensor(old_grid, dtype = torch.float).unsqueeze(0).to(device)
                        flat_tensor = torch.tensor(old_flat, dtype = torch.float).unsqueeze(0).to(device)
                        prediction = model(grid_tensor, flat_tensor)
                        #print(f"prediction: {prediction}")
                        final_move[int(torch.argmax(prediction).item())] = 1
    
                ai_move(player_snake=player_snake, final_move=final_move)

            old_dist = abs(player_snake.snake_head[0] - player_apple.apple_pos[0]) + abs(player_snake.snake_head[1] - player_apple.apple_pos[1])

            player_snake.move()
            dead = not player_snake.is_alive()

            new_dist = abs(player_snake.snake_head[0] - player_apple.apple_pos[0]) + abs(player_snake.snake_head[1] - player_apple.apple_pos[1])

            reward = 0

            if dead:
                reward = -1
                starvation = 0
            elif np.array_equal(player_snake.snake_head, player_apple.apple_pos):
                player_snake.grow()
                player_apple.generate(player_snake)
                reward = 3
                starvation = 0
            else:
                if new_dist < old_dist:
                    starvation += 1
                    reward = .1 #to encourage it to move towards apple
                elif starvation > (RES * RES):
                    dead = True
                    reward = -3
                else:
                    starvation += 1
                    reward = 0
                
                #reward += 5 * flood_fill_count(player_snake.snake_head, player_snake.snake_body, RES)/(RES * RES)
            if dead:
                new_grid, new_flat = old_grid, old_flat
            else:
                new_grid, new_flat = get_game_state(player_snake=player_snake, player_apple=player_apple, res=RES)

            if AI_PLAYING:
                trainer.train_step(old_grid, old_flat, final_move, reward, new_grid, new_flat, dead)
                agent.remember(old_grid, old_flat, final_move, reward, new_grid, new_flat, dead)

                if dead:
                    epsilon = max(0, 200 - games_played * 0.05)
                    games_played += 1
                    avg_score = (avg_score + (player_snake.length - avg_score)/games_played)
                    recent_scores.append(player_snake.length)
                    recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                    stats = {
                        'avg_score': avg_score,
                        'recent_scores': list(recent_scores)
                    }
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f)
                    if player_snake.length > high_score:
                        high_score = player_snake.length
                        
                        checkpoint = {
                            'state': model.state_dict(),
                            'games_played': games_played,
                            'high_score': high_score,
                            'epsilon': epsilon
                        }
                        model.save(checkpoint, file_name='best_model.pth')
                    #resets board
                    player_snake = snake.Snake(RES)
                    player_apple.generate(player_snake)
                
                    agent.train_long_memory()

                    print(f"Game {games_played} Over. Epsilon: {epsilon} High Score: {high_score} Average Score: {avg_score} Recent Average: {recent_avg}")
            if not AI_PLAYING and dead:
                player_snake = snake.Snake(RES)
                player_apple.generate(player_snake)
            
            if DISPLAY:
                player_snake.draw(screen, CELL_SIZE, background)
                player_apple.draw(screen, CELL_SIZE)
    
                pygame.display.flip()
                clock.tick(100 if AI_PLAYING else 10)

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


def flood_fill_count(start, snake_body, res):
        visited = set()
        stack = [tuple(start)]
        body_set = {tuple(s) for s in snake_body[1:]}
        while stack:
            pos = stack.pop()
            if pos in visited:
                continue
            x, y = pos
            if x < 0 or x >= res or y < 0 or y >= res:
                continue
            if pos in body_set:
                continue
            visited.add(pos)
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        return len(visited)

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

def get_game_state(player_snake : snake.Snake, player_apple : apple.Apple, res):
    head = player_snake.snake_head
    apple_pos = player_apple.apple_pos

    grid = np.zeros((3, res, res), dtype=np.float32)

    for segment in player_snake.snake_body[1:]:
        grid[0, segment[0], segment[1]] = 1.0

    grid[1, head[0], head[1]] = 1.0

    grid[2, apple_pos[0], apple_pos[1]] = 1.0

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
    
    #checks how open the map is to prevent trapping itself
    


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
    
    
    
    # space_l = flood_fill_count(p_l, player_snake.snake_body, res) / (res * res)
    # space_r = flood_fill_count(p_r, player_snake.snake_body, res) / (res * res)
    # space_u = flood_fill_count(p_u, player_snake.snake_body, res) / (res * res)
    # space_d = flood_fill_count(p_d, player_snake.snake_body, res) / (res * res)

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

        #head position
        head[0]/res,
        head[1]/res,

        # #food location relative to head position
        apple_pos[0] < head[0],  # Food is Left
        apple_pos[0] > head[0],  # Food is Right
        apple_pos[1] < head[1],  # Food is Up
        apple_pos[1] > head[1]   # Food is Down
        
        # #free space when going in the different directions
        # space_l,
        # space_r,
        # space_u,
        # space_d
    ]
    flat = np.array(state, dtype=np.float32)
    return grid, flat
    #return [int(x) for x in state[:-4]] + [ i for i in state[-4:]]

if __name__ == "__main__":
    main()