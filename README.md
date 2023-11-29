# RockPaperScissorRL

## Description

This project contains an algorithm for the simple game Rock Paper Scissors. The algorithm utilizes reinforcement learning Q-learning algorithm with human assistance. The agent will be directly playing against the player and try to learn from the pattern the player makes. The learning process utilizes a ε-greedy algorithm to allow the agent to explore counter to the player's strategy. The ε value is also dynamically updated based on whether the agent wins or loses a game. Under this strategy, after playing many games with the agent, the agent's win rate almost never goes below 33.33%.

## Dependencies

|Dependency|Version|
|-|-|
|gym|0.26.2|
|torch|2.1.0|

## Run the program
