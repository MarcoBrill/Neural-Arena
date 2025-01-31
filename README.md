# Neural-Arena
Python scripts that lavareges GPU and TPU for gaming purposes. The application is a real-time strategy game where players train and deploy AI agents to compete in a virtual arena. The GPU is used for rendering the game environment and handling real-time player interactions, while the TPU is used for training and optimizing the AI agents.

# Neural Arena: GPU-TPU Hybrid Gaming Workflow

Neural Arena is a real-time strategy game that leverages GPU for rendering and TPU for AI training. Players design and train AI agents to compete in a dynamic arena.

## Features
- Real-time rendering using GPU.
- AI agent training using TPU.
- Multiplayer support for competitive gameplay.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Pygame

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-arena.git
   cd neural-arena
   
## Usage
Use the arrow keys to control your agent.

Press T to start training your AI agent on the TPU.
Press Q to quit the game.

## Inputs and Outputs
Inputs:
Player Input: Real-time commands from players (e.g., move, attack, train).

Training Data: Game state data for RL training (e.g., agent actions, rewards).

Game Configuration: Settings for the game environment (e.g., map size, agent count).

## Outputs:
Rendered Game Environment: Visual output of the game (GPU).

Trained AI Models: Optimized AI agents ready for deployment (TPU).

Game Metrics: Performance statistics (e.g., win/loss ratio, training progress).


### How It Works
1. The GPU handles the game's real-time rendering and physics.
2. The TPU is used for training AI agents using reinforcement learning.
3. Players can switch between manual control and AI-driven gameplay.

This workflow is a novel way to combine gaming and machine learning, offering a unique experience for players and developers alike.
