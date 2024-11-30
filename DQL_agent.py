from collections import deque
from DQN import DQN
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.position_history = deque(maxlen=100)  # Store last 100 positions
        self.action_history = deque(maxlen=100)    # Store last 100 actions
        self.gamma = 0.15    # discount rate
        self.epsilon = 1.5  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0022
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.position_history.append(state)  # Assuming state includes position
        self.action_history.append(action)

    def act(self, state):
        # Optionally use position_history and action_history here
        if len(self.memory) > 0:
            if len(self.memory) > 1:
                previous_reward = self.memory[-2][2]
            else:
                previous_reward = 0
            reward = self.memory[-1][2]
            '''
            if reward - previous_reward > 0:
                self.epsilon = 1
                return 8
            '''
            print(reward)
            if reward == -2:
                self.epsilon = .8
                if np.random.rand() <= self.epsilon:
                    print("chose random action")
                    return random.randint(0, self.action_size - 1)
                else:
                    print("chose optimal action")
                    act_values = self.model(torch.FloatTensor(state))
                    return torch.argmax(act_values).item()
            if reward == -8:  # select action 5 when reward is low
                print("chose to go down because drone to high")
                return 5
            if reward == -18:
                self.epsilon = 1
                return 4
            if reward == -16:
                return 8
            if reward == -14:
                return 8
            if reward == -12:
                return 8
            if reward == -10:
                return 8
            if 1 > reward > -4:
                if np.random.rand() <= self.epsilon:
                    if np.random.rand() <= .75:
                        print("chose rotate right")
                        return 1
                    else:
                        print("chose rotate left")
                        return 2
                else:
                    print("chose optimal action")
                    act_values = self.model(torch.FloatTensor(state))
                    return torch.argmax(act_values).item()
            if reward > 2.0:  # select action 5 when reward is low
                print("chose to go down because drone to high")
                return 0
            if np.random.rand() <= self.epsilon:
                print("chose random action")
                return random.randint(0, self.action_size -1)
            else:
                print("chose optimal action")
                act_values = self.model(torch.FloatTensor(state))
                return torch.argmax(act_values).item()
        else:
            print("chose random action")
            return random.randint(0, self.action_size -1)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.FloatTensor(next_state))).item()
            target_f = self.model(torch.FloatTensor(state))
            target_f = target_f.clone()  # Clone to avoid in-place operation
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay