import pygame
import sys
import snake
import apple
import ai
import numpy as np
import torch
import random
import os

def main():
    res = 15
    cell_size = 40
    window_size = res * cell_size
    display = False
    starvation = 0

    pygame.init()
    #if display:
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Snake")

    clock = pygame.time.Clock()
    print("start")
    player_snake = snake.Snake(res)
    player_apple = apple.Apple(res, True)

    #AI stuff
    model = ai.Linear_QNet(input_size=11, hidden_size=256, output_size=3)
    saved_model_path = './model/best_model.pth'
    epsilon = 80 #increases variability at the beginning
    games_played = 0
    high_score = 0
    if os.path.exists(saved_model_path):
        checkpoint = torch.load(saved_model_path)
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
                    if event.key == pygame.K_LEFT or event.key == pygame.K_SPACE:
                        display = not display 
            #AI input here
            #stores old state for comparison
            old_state = get_game_state(player_snake=player_snake, player_apple=player_apple, res=res)
            #choosing action
            final_move = [0,0,0]
            if random.randint(0, 200) < epsilon:
                final_move[random.randint(0,2)] = 1
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(old_state, dtype = torch.float).unsqueeze(0)
                    prediction = model(state_tensor)
                    final_move[torch.argmax(prediction).item()] = 1

            ai_move(player_snake=player_snake, final_move=final_move)
            old_dist = abs(player_snake.snake_head[0] - player_apple.apple_pos[0]) + abs(player_snake.snake_head[1] - player_apple.apple_pos[1])

            player_snake.move()
            new_dist = abs(player_snake.snake_head[0] - player_apple.apple_pos[0]) + abs(player_snake.snake_head[1] - player_apple.apple_pos[1])

            reward = 0
            dead = not player_snake.is_alive()

            if dead:
                reward = -5
                starvation = 0
            elif np.array_equal(player_snake.snake_head, player_apple.apple_pos):
                player_snake.grow()
                player_apple.generate(player_snake)
                reward = 100
                starvation = 0
            else:
                if new_dist < old_dist:
                    starvation += .1
                    reward = 1 #to encourage it to move towards apple

                else:
                    starvation += .1
                    reward = -2 - starvation
            
            new_state = get_game_state(player_snake=player_snake, player_apple=player_apple, res=res)

            trainer.train_step(old_state, final_move, reward, new_state, dead)
            agent.remember(old_state, final_move, reward, new_state, dead)

            if dead:
                epsilon = max(0, 80 - games_played * 0.005)
                games_played += 1
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
                player_snake = snake.Snake(res)
                player_apple.generate(player_snake)
               
                agent.train_long_memory()

                print(f"Game {games_played} Over. Epsilon: {epsilon} High Score: {high_score}")
            
            if display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                screen.fill((255, 255, 255))
                player_snake.draw(screen, cell_size)
                player_apple.draw(screen, cell_size)
    
                pygame.display.flip()
                clock.tick(1000)
    except KeyboardInterrupt:
        print("Program stopped by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
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

def get_game_state(player_snake : snake.Snake, player_apple : apple.Apple, res):
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

if __name__ == "__main__":
    main()