import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from rps_env import RockPaperScissorsEnv
from rl_model import RPSModel
import numpy as np

class Agent:
    def __init__(self, model, device, epsilon=0.1):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.device = device
        self.action_counts = [0, 0, 0]
        self.win_count = 0  # 新增：跟踪胜利次数
        self.total_game_count = 0
        self.epsilon = epsilon

    def select_action(self, state):
        # 随机选择动作的概率
        if random.random() < self.epsilon:
            action = random.choice([0, 1, 2])  # 随机选择一个动作
            log_prob = None  # 在随机选择时，没有log概率
            self.action_counts[action] += 1
            return action, log_prob
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            raw_scores = self.model(state)
            probs = F.softmax(raw_scores, dim=1)
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            self.action_counts[action.item()] += 1
            return action.item(), log_prob

    def train(self, log_probs, rewards):
        # 累积奖励的贴现和
        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + 0.99 * R
            discounted_rewards.insert(0, R)

        #print(f'Discounted Reward 1 {discounted_rewards}')

        # 标准化奖励
        discounted_rewards = torch.tensor(discounted_rewards)
        #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        #print(f'Discounted Reward 2 {discounted_rewards}')

        policy_losses = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            if log_prob is not None:  # 只处理非随机选择的动作
                policy_losses.append(-log_prob * reward)

        if policy_losses:  # 只有在有损失值时才执行反向传播
            self.optimizer.zero_grad()
            total_loss = torch.stack(policy_losses).sum()  # 使用torch.stack处理一维张量的列表
            total_loss.backward()  # 反向传播
            self.optimizer.step()

def print_action_frequency(agent, agent_name):
    total_actions = sum(agent.action_counts)
    print(f"Action frequencies for {agent_name}:")
    for idx, count in enumerate(agent.action_counts):
        frequency = count / total_actions
        print(f"  Action {idx} (Rock/Paper/Scissors): {frequency:.2f}")

# 创建两个agent
input_size = 6  # This can be adjusted based on how you define the state
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 3  # Rock, Paper, Scissors
# 检查CUDA是否可用并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型和agent
#model1 = RPSModel(input_size, hidden_size, output_size).to(device)
#model2 = RPSModel(input_size, hidden_size, output_size).to(device)
#agent1 = Agent(model1, device)
#agent2 = Agent(model2, device)


# 创建环境
env = RockPaperScissorsEnv()

# 训练循环
num_episodes = 10000
print_interval = 100
win_1, win_2, total_games = 0, 0, 0

num_agents = 10
agents = [Agent(RPSModel(input_size, hidden_size, output_size).to(device), device) for _ in range(num_agents)]

for episode in range(num_episodes):
    state = env.reset()
    log_probs1, log_probs2 = [], []
    rewards1, rewards2 = [], []
    done = False

    agent_indices = np.random.choice(len(agents), 2, replace=False)
    agent1, agent2 = agents[agent_indices[0]], agents[agent_indices[1]]

    while not done:
        # 两个agent分别选择动作
        action1, log_prob1 = agent1.select_action(state)
        action2, log_prob2 = agent2.select_action(state)

        #print(log_prob1, log_prob2)

        # 执行动作
        _, reward, done, _ = env.step(action1, action2)

        # 存储结果
        log_probs1.append(log_prob1)
        rewards1.append(reward)
        log_probs2.append(log_prob2)
        rewards2.append(-reward)  # 对于第二个agent来说，奖励是相反的

        if reward > 0:
            agent1.win_count += 1
        elif reward < 0:
            agent2.win_count += 1

        agent1.total_game_count += 1
        agent2.total_game_count += 1
        total_games += 1

        if episode % print_interval == 0:
        #    print(f"Episode {episode}: Agent1 Reward: {sum(rewards1)}, Agent2 Reward: {sum(rewards2)}")
            print(f"Episode {episode}")

    #print(log_probs1, rewards1)
    #print(log_probs2, rewards2)
    # 训练两个agent
    agent1.train(log_probs1, rewards1)
    agent2.train(log_probs2, rewards2)

print(f'Total games: {total_games}')

# 打印每个agent的行为模式
for idx, agent in enumerate(agents):
    print_action_frequency(agent, f"Agent {idx+1}")
    print(f'Winrate: {agent.win_count / agent.total_game_count}')

# 确定胜率较高的agent
best_agent = max(agents, key=lambda agent: agent.win_count)



# 确定胜率较高的agent
if win_1 / total_games > win_2 / total_games:
    best_agent = agent1
else:
    best_agent = agent2

# 定义动作到数字的映射
action_mapping = {'rock': 0, 'paper': 1, 'scissors': 2}

# 创建与用户交互的循环
def play_with_agent(agent):
    while True:
        user_input = input("Choose rock, paper or scissors (or 'exit' to stop): ")
        if user_input == 'exit':
            break
        if user_input not in action_mapping:
            print("Invalid input. Please choose rock, paper or scissors.")
            continue

        user_action = action_mapping[user_input]
        state = np.zeros(6)  # 假设没有历史信息
        state[user_action] = 1  # 设置用户的动作
        agent_action, _ = agent.select_action(state)
        print(f"Agent chose: {list(action_mapping.keys())[agent_action]}")

        if user_action == agent_action:
            print("It's a tie!")
        elif (user_action == 0 and agent_action == 2) or \
             (user_action == 1 and agent_action == 0) or \
             (user_action == 2 and agent_action == 1):
            print("You win!")
        else:
            print("You lose!")

# 与胜率较高的agent玩游戏
play_with_agent(best_agent)