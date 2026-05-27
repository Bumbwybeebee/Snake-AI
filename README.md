# Snake AI - Reinforcement Learning Implementation
<!-- 
## Overview
This project implements an AI player for the classic Snake game using reinforcement learning. The AI learns to play the game by using a neural network to make decisions about which direction to move the snake in order to maximize its score and survive as long as possible.

## Project Structure
```
Snake-AI/
├── ai.py          # Neural network implementation
├── apple.py        # Apple generation and positioning
├── game.py        # Core game logic
├── main.py        # Main game loop and execution
├── snake.py       # Snake entity implementation
├── requirements.txt # Python dependencies
└── model/        # Saved model directory
└── sprites/        # Game sprites and images
```

## Features
- **Reinforcement Learning AI**: The AI uses a Q-learning algorithm with a neural network to learn how to play the game
- **Simultaneous Games**: The system can train multiple games in parallel for efficient learning
- **PyGame Visualization**: Visual representation of the game using PyGame
- **Model Persistence**: The trained model is automatically saved and can be loaded for continued training

## Requirements
- Python 3.x
- PyTorch >= 1.7.0
- NumPy >= 1.19.0
- PyGame >= 2.0.0

## Installation
1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the game:
```bash
python main.py
```

## How it Works

### State Representation
The AI uses a neural network to evaluate the game state and decide on the best action. The state is represented by 11 boolean values:
- 3 danger values (straight, left, right)
- 4 direction values (current direction: up, down, left, right)
- 4 food direction values (food is left/right/up/down of snake head)

### Neural Network Architecture
The neural network has:
- Input layer of 11 nodes (state representation)
- One hidden layer with 512 neurons
- Output layer with 3 nodes (straight, left, right actions)

### Training Process
The AI uses saved model from previous sessions to continue training. During each training session:
- Multiple games run simultaneously for efficient training
- Experiences are stored in a replay buffer and used for training
- The model uses epsilon-greedy exploration strategy that decreases over time

## File Descriptions

### Core Components
- `main.py` - Main game loop that runs the game and handles the AI training process
- `game.py` - Core game logic and state management
- `ai.py` - AI implementation using Q-Learning with neural networks
- `snake.py` - Snake entity with movement, collision detection, and rendering
- `apple.py` - Apple generation and positioning
- `requirements.txt` - Python dependencies needed to run the application

## Technical Details
The system implements an intelligent agent that learns to play the classic snake game using reinforcement learning. The agent uses a neural network to make decisions about which direction to move the snake in order to maximize its score and survive as long as possible.

The neural network implementation uses PyTorch and consists of:
- An input layer that takes 11 features
- A hidden layer with 512 neurons
- An output layer with 3 nodes for actions (straight, left, right)

## Training Method
The AI uses Q-learning with experience replay:
1. Multiple games run simultaneously for efficient training
2. Experiences are stored in a replay buffer (memory) and used for training
3. The model uses epsilon-greedy exploration strategy that decreases over time

## Controls
- SPACE key: Toggle display on/off

## Project Information
This project demonstrates a complete reinforcement learning implementation for the classic Snake game. The AI learns to play the game by using a neural network to make decisions about which direction to move the snake in order to maximize its score and survive as long as possible. -->