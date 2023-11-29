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
            return random.choice([0, 1, 2])
        else:
            self.random_act = False
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.model(state)
            self.q_cache = q_values
            if self.verbose:
                print(f"Q: {[round(v, 3) for v in q_values.tolist()[0]]}")
            return torch.argmax(q_values, dim=1).item()

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

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def infer_action_probabilities(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad(): 
            q_values = self.model(state)
        return F.softmax(q_values, dim=1).cpu().numpy()[0]
