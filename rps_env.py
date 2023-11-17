import gym
from gym import spaces
import numpy as np

class RockPaperScissorsEnv(gym.Env):
    """
    A simple Rock, Paper, Scissors environment following OpenAI Gym interface.
    Actions are as follows: 0 - Rock, 1 - Paper, 2 - Scissors.
    """
    def __init__(self):
        super(RockPaperScissorsEnv, self).__init__()
        # Define action and observation space
        # Both agents have the same action space (0: Rock, 1: Paper, 2: Scissors)
        self.action_space = spaces.Discrete(3)
        # Observation space is actually not used in this game, but defined for compatibility
        self.observation_space = spaces.Discrete(1)

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        # In this game, the reset state does not have any meaningful information
        return np.zeros(6)

    def step(self, action1, action2):
        """
        Execute the actions by two agents and return the new state, reward, done, and additional info.
        """
        # Determine the winner
        if action1 == action2:
            reward = 0  # Tie
        elif (action1 == 0 and action2 == 2) or (action1 == 1 and action2 == 0) or (action1 == 2 and action2 == 1):
            reward = 1  # Agent 1 wins
        else:
            reward = -1  # Agent 2 wins
        
        new_state = np.zeros(6)
        new_state[action1] = 1
        new_state[3 + action2] = 1


        # State, reward, done, info
        return new_state, reward, True, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    # Create an instance of the environment to test
    env = RockPaperScissorsEnv()
    env.reset()

    # Testing the environment with some actions
    action1, action2 = 0, 1  # Rock vs Paper
    state, reward, done, info = env.step(action1, action2)
    print(state, reward, done, info)  # Display the result of the step

