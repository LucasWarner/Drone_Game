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
        self.gamma = 0.2    # discount rate
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
            reward = self.memory[-1][2]
            if len(self.memory) > 1:
                previous_reward = self.memory[-2][2]
                previous_action = self.memory[-2][1]
                if reward - previous_reward > 0:
                    if np.random.rand() <= self.epsilon:
                        if previous_action == 1:
                            return 2
                        elif previous_action == 2:
                            return 1
                        elif previous_action == 0:
                            self.epsilon = 1
                            return 8
                    else:
                        print("chose optimal action")
                        act_values = self.model(torch.FloatTensor(state))
                        return torch.argmax(act_values).item()

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
            elif reward == -8:  # select action 5 when reward is low
                print("chose to go down because drone to high")
                return 5
            elif reward == -18:
                self.epsilon = 1
                return 4
            elif reward == -16:
                return 8
            elif reward == -14:
                return 8
            elif reward == -12:
                return 8
            elif reward == -10:
                return 8
            elif 2 > reward > -4:
                if np.random.rand() <= self.epsilon:
                    print("chose rotate right")
                    return 1
                else:
                    print("chose optimal action")
                    act_values = self.model(torch.FloatTensor(state))
                    return torch.argmax(act_values).item()
            elif reward == 10:
                print("chose to go forward or down because drone is aligned")
                for_or_down = random.randint(0, 1)
                if for_or_down == 0:
                    return 0
                else:
                    return 5
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