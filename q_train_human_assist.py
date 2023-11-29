import torch, random
import torch.nn.functional as F
from rps_env import RockPaperScissorsEnv
from rl_model import RPSModel
import numpy as np
from q_agent import QAgent

env = RockPaperScissorsEnv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_agents = 1
agents = [QAgent(RPSModel().to(device), device, verbose=True) for _ in range(num_agents)]

def get_human_action():
    user_input = input("Choose rock (0), paper (1) or scissors (2): ")
    if user_input == 'exit': return user_input
    if user_input not in ["0", "1", "2", "exit"]:
        print("INVALID INPUT!")
        return get_human_action()
    return int(user_input)

def print_game_result(human_action, agent_action):
    if human_action == agent_action:
        print("It's a tie!")
    elif (human_action == 0 and agent_action == 2) or \
         (human_action == 1 and agent_action == 0) or \
         (human_action == 2 and agent_action == 1):
        print("You win!")
    else:
        print("You lose!")

state1, state2 = env.reset()
agent1 = agents[0]
player_win_count = 0

while True:
    action1 = agent1.select_action(state1)
    action2 = get_human_action()

    if action2 == 'exit': break

    print(f'A1: {action1}, Human: {action2}')

    next_state1, next_state2, reward1, reward2, done, _ = env.step(action1, action2)

    print_game_result(action2, action1)

    agent1.train(state1, action1, reward1, next_state1, done)
    state1 = next_state1

    if reward1 > 0:
        agent1.win_count += 1
        if agent1.epsilon > 0:
            agent1.epsilon -= 0.1
    elif reward1 == 0:
        if agent1.epsilon < 1 and not agent1.random_act:
            agent1.epsilon += 0.05
    else:
        if agent1.epsilon < 1 and not agent1.random_act:
            agent1.epsilon += 0.1
        player_win_count += 1
    agent1.epsilon = round(agent1.epsilon, 1)
    agent1.total_game_count += 1

    print(f'Agent Winrate: {round(agent1.win_count / agent1.total_game_count * 100, 3)}%')
    print(f'Agent Epsilon: {agent1.epsilon}')

    print(f'Scoreboard: Player: {player_win_count}, Agent: {agent1.win_count}')

    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

