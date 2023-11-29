import gym
from gym import spaces
import numpy as np

class RockPaperScissorsEnv(gym.Env):
    def __init__(self):
        super(RockPaperScissorsEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(1)
        self.state1, self.state2 = np.zeros(5), np.zeros(5)

    def reset(self):
        self.state1, self.state2 = np.zeros(5), np.zeros(5)
        return self.state1, self.state2

    def step(self, action1, action2):
        if action1 == action2:
            reward1, reward2 = 0, 0
        elif (action1 == 0 and action2 == 2) or (action1 == 1 and action2 == 0) or (action1 == 2 and action2 == 1):
            reward1, reward2 = 1, -1
        else:
            reward1, reward2 = -1, 1
        
        self.state1 = np.roll(self.state1, -1)
        self.state1[-1] = action2

        self.state2 = np.roll(self.state2, -1)
        self.state2[-1] = action1

        return self.state1, self.state2, reward1, reward2, True, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    env = RockPaperScissorsEnv()
    env.reset()

    action1, action2 = 0, 1
    state, reward, done, info = env.step(action1, action2)
    print(state, reward, done, info)

