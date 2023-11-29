import torch, random
import torch.nn.functional as F
from rps_env import RockPaperScissorsEnv
from rl_model import RPSModel
import numpy as np

class QAgent:
    def __init__(self, model, device, epsilon=1.0, gamma=0.99, verbose=False):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        self.device = device
        self.epsilon = epsilon
        self.gamma = gamma
        self.win_count = 0
        self.total_game_count = 0
        self.verbose = verbose
        self.q_cache = None
        self.random_act = False

    def select_action(self, state):
        if self.verbose:
            print(f"Current State: {[v for v in state.tolist()]}")
        if random.random() < self.epsilon:
            self.random_act = True
            if self.verbose: 
                print("Random Action Selected!")
            return random.choice([0, 1, 2])  # 随机选择动作
        else:
            self.random_act = False
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.model(state)
            self.q_cache = q_values
            if self.verbose:
                print(f"Q: {[round(v, 3) for v in q_values.tolist()[0]]}")
            return torch.argmax(q_values, dim=1).item()  # 选择具有最高Q值的动作

    def train(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        current_q = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q = self.model(next_state).max(1)[0].detach()
        expected_q = reward + self.gamma * next_q * (1 - done.float())

        loss = F.mse_loss(current_q, expected_q)

        #print(f'currQ: {round(current_q.item(), 3)}, nextQ: {round(next_q.item(), 3)}, expQ: {round(expected_q.item(), 3)}, loss {round(loss.item(), 3)}')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def infer_action_probabilities(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():  # 确保在推断模式下不计算梯度
            q_values = self.model(state)
        return F.softmax(q_values, dim=1).cpu().numpy()[0]

if __name__ == "__main__":
    # 创建环境
    env = RockPaperScissorsEnv()

    # 训练循环
    num_episodes = 100
    print_interval = 1
    win_1, win_2, total_games = 0, 0, 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_agents = 10
    agents = [QAgent(RPSModel().to(device), device) for _ in range(num_agents)]

    num_rounds_per_episode = 100  # 每个episode的对战轮数

    for episode in range(num_episodes):
        state1, state2 = env.reset()
        agent_indices = np.random.choice(len(agents), 2, replace=False)
        agent1, agent2 = agents[agent_indices[0]], agents[agent_indices[1]]

        for _ in range(num_rounds_per_episode):
            action1 = agent1.select_action(state1)
            action2 = agent2.select_action(state2)

            #print(f'A1: {action1}, A2: {action2}')

            next_state1, next_state2, reward1, reward2, done, _ = env.step(action1, action2)

            #print(f'State1: {state1}, State2: {state2}')

            #print('A1')
            agent1.train(state1, action1, reward1, next_state1, done)

            #probabilities_after_training = agent1.infer_action_probabilities(state1)
            #print("A1 after training:", probabilities_after_training)

            #print('A2')
            agent2.train(state2, action2, reward2, next_state2, done)

            #probabilities_after_training = agent2.infer_action_probabilities(state2)
            #print("A2 after training:", probabilities_after_training)

            state1, state2 = next_state1, next_state2

            # 更新胜利和游戏次数
            if reward1 > 0:
                agent1.win_count += 1
                # epsilon动态调整
                if agent1.epsilon > 0:
                    agent1.epsilon -= 0.1
                if agent2.epsilon < 1:
                    agent2.epsilon += 0.1
            elif reward2 > 0:
                agent2.win_count += 1
                if agent2.epsilon > 0:
                    agent1.epsilon -= 0.1
                if agent1.epsilon < 1:
                    agent2.epsilon += 0.1
            else:
                # 平局时增加随机性
                if agent2.epsilon < 1:
                    agent1.epsilon += 0.1
                if agent1.epsilon < 1:
                    agent2.epsilon += 0.1

            agent1.total_game_count += 1
            agent2.total_game_count += 1

        if episode % print_interval == 0:
            print(f"Episode {episode}")


    for idx, agent in enumerate(agents):
        winrate = agent.win_count / agent.total_game_count if agent.total_game_count > 0 else 0
        print(f"Agent {idx+1}: Winrate = {winrate:.2f}, Total Games = {agent.total_game_count}")

    action_mapping = {'rock': 0, 'paper': 1, 'scissors': 2}

    def play_with_agent(agent):
        state = np.zeros(5)
        while True:
            user_input = input("Choose rock, paper or scissors (or 'exit' to stop): ")
            if user_input == 'exit':
                break
            '''
            if user_input not in action_mapping.values():
                print("Invalid input. Please choose rock, paper or scissors.")
                continue'''

            #user_action = action_mapping[user_input]
            user_action = int(user_input)
            state = np.roll(state, -1)
            state1[-1] = user_action

            agent_action = agent.select_action(state)
            print(f"Agent chose: {list(action_mapping.keys())[agent_action]}")

            if user_action == agent_action:
                print("It's a tie!")
            elif (user_action == 0 and agent_action == 2) or \
                (user_action == 1 and agent_action == 0) or \
                (user_action == 2 and agent_action == 1):
                print("You win!")
            else:
                print("You lose!")

    best_agent = max(agents, key=lambda agent: agent.win_count / agent.total_game_count)

    def save_model(agent, file_name):
        torch.save(agent.model.state_dict(), file_name)
    save_model(best_agent, 'best_agent_model.pth')
    play_with_agent(best_agent)