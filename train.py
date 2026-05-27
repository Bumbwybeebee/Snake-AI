#!/usr/bin/env python3

import pygame
import sys
import snake
import ai
import numpy as np
import torch
import os
import game as snake_game
import random
import time
from collections import deque
import json

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    RES                = 15
    CELL_SIZE          = 40
    WINDOW_SIZE        = RES * CELL_SIZE
    DISPLAY            = False
    SIMULTANEOUS_GAMES = 64

    pygame.init()
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w * .9, info.current_h * .9

    # Fit as many cells as possible in the smaller screen dimension
    CELL_SIZE = min(screen_w, screen_h) // RES
    WINDOW_SIZE = RES * CELL_SIZE

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    background = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    for row in range(RES):
        for col in range(RES):
            color = (170, 215, 81) if (row + col) % 2 == 0 else (162, 209, 73)
            pygame.draw.rect(background, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()

    games = [snake_game.SnakeGame(RES) for _ in range(SIMULTANEOUS_GAMES)]

    # ── Model — identical to main.py ──────────────────────────────────────────
    model            = ai.Conv_QNet(res=RES, flat_input_size=13, hidden_size=512, output_size=3).to(device)
    saved_model_path = './model/best_model.pth'
    epsilon          = 200
    games_played     = 0
    high_score       = 0
    game_timestamps  = deque()
    stats_path       = './model/stats.json'

    if os.path.exists(saved_model_path):
        checkpoint   = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(checkpoint['state'])
        games_played = checkpoint['games_played']
        high_score   = checkpoint['high_score']
        epsilon      = checkpoint['epsilon']
        print("loaded saved model")
    else:
        print("no saved model found")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        avg_score = stats['avg_score']
        recent_scores = deque(stats['recent_scores'], maxlen=2000)
    else:
        avg_score = 0
        recent_scores = deque(maxlen=2000)

    trainer = ai.QTrainerCNN(model=model, lr=0.001, gamma=0.99)
    agent   = ai.AgentCNN(model=model, trainer=trainer)
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    DISPLAY = not DISPLAY

            # ── 1. Get old states for all games ───────────────────────────────
            old_grids = []
            old_flats = []
            for game in games:
                g, f = game.get_state()
                old_grids.append(g)
                old_flats.append(f)

            # ── 2. Batch inference (epsilon-greedy, identical to main.py) ─────
            grids_t = torch.tensor(np.array(old_grids), dtype=torch.float32, device=device)
            flats_t = torch.tensor(np.array(old_flats), dtype=torch.float32, device=device)
            with torch.no_grad():
                preds          = model(grids_t, flats_t)
                action_indices = torch.argmax(preds, dim=1).cpu().tolist()

            actions = []
            for i in range(SIMULTANEOUS_GAMES):
                if random.randint(0, 200) < epsilon:
                    action_indices[i] = random.randint(0, 2)
                a = [0, 0, 0]
                a[action_indices[i]] = 1
                actions.append(a)

            # ── 3. Step all games, collect experiences ────────────────────────
            new_grids  = []
            new_flats  = []
            rewards    = []
            dones      = []

            for i, game in enumerate(games):
                reward, done = game.apply_action(actions[i])

                if done:
                    # terminal state — use old state as new (masked in Bellman anyway)
                    new_grids.append(old_grids[i])
                    new_flats.append(old_flats[i])
                else:
                    ng, nf = game.get_state()
                    new_grids.append(ng)
                    new_flats.append(nf)

                rewards.append(reward)
                dones.append(done)

                agent.remember(old_grids[i], old_flats[i], actions[i], reward,
                               new_grids[i], new_flats[i], done)

            # ── 4. Online batch train_step — mirrors main.py's per-step call ──
            # main.py calls trainer.train_step() once per step with one experience.
            # Here we call it once with all SIMULTANEOUS_GAMES experiences — same
            # total gradient signal, just batched for efficiency.
            trainer.train_step(old_grids, old_flats, actions, rewards,
                               new_grids, new_flats, dones)

            # ── 5. Handle deaths — identical logic to main.py ─────────────────
            for i, (game, done) in enumerate(zip(games, dones)):
                if done:
                    if game.high_score > high_score:
                        high_score = game.high_score
                    avg_score = (avg_score + (game.high_score - avg_score)/games_played)
                    recent_scores.append(game.high_score)
                    recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                    stats = {
                        'avg_score': avg_score,
                        'recent_scores': list(recent_scores)
                    }
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f)

                    game.reset()
                    games_played += 1
                    epsilon = max(0, 200 * 0.999 ** games_played)

                    current_time = time.time()
                    game_timestamps.append(current_time)
                    while game_timestamps and game_timestamps[0] < current_time - 10:
                        game_timestamps.popleft()
                    gps = len(game_timestamps) / 10.0

                    print(f"Game {games_played:6d} | eps {epsilon:.5f} | "
                          f"high {high_score} | {gps:.1f} games/s | avg {avg_score} | recent avg {recent_avg}")

                    # Replay — identical to main.py's on-death train_long_memory
                    agent.train_long_memory()

            # ── 6. Display ────────────────────────────────────────────────────
            if DISPLAY:
                screen.blit(background, (0, 0))
                games[0].player_snake.draw(screen, CELL_SIZE, background)
                games[0].player_apple.draw(screen, CELL_SIZE)
                clock.tick(60)

            pygame.display.flip()

    except KeyboardInterrupt:
        print("Program stopped by user.")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        checkpoint = {
            'state':        model.state_dict(),
            'games_played': games_played,
            'high_score':   high_score,
            'epsilon':      epsilon,
        }
        model.save(checkpoint, file_name='best_model.pth')
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()