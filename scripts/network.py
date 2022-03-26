import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

BATCH_SIZE = 100
LR = 0.02
GAMMA = 0.90
EPSILON = 0.9
Q_NETWORK_ITERATION = 1000
MEMORY_CAPACITY = 10000

NUM_ACTIONS = 5
NUM_STATES = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 80)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(80, NUM_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 80)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(80, 160)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(160, 80)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(80, NUM_ACTIONS)
        self.fc4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)


class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 80)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(80, 160)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(160, 300)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(300, 30)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc5 = nn.Linear(30, NUM_ACTIONS)
        self.fc5.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)


class DQN:
    def __init__(self, network):
        self.eval_net, self.target_net = network(), network()
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.RMSprop(self.eval_net.parameters())
        self.eval_net.to(device)
        self.target_net.to(device)
        self.loss = nn.SmoothL1Loss()

    def store_trans(self, state, action, reward, next_state):
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, action, reward, next_state))
        self.memory[index, :] = trans
        self.memory_counter += 1
        
    def save(self, path):
        self.eval_net.save(path)

    def choose_action(self, state):
        state = torch.unsqueeze(torch.cuda.FloatTensor(state, device=device), 0)
        if np.random.randn() <= EPSILON:
            action_value = self.eval_net.forward(state)
            action = int(torch.max(action_value, 1)[1])
        else:
            action = np.random.randint(0, NUM_ACTIONS)
        return action

    def learn(self):
        if self.learn_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.cuda.FloatTensor(batch_memory[:, :NUM_STATES], device=device)
        # note that the action must be a int
        batch_action = torch.cuda.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int), device=device)
        batch_reward = torch.cuda.FloatTensor(batch_memory[:, NUM_STATES + 1: NUM_STATES + 2], device=device)
        batch_next_state = torch.cuda.FloatTensor(batch_memory[:, -NUM_STATES:], device=device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss(q_eval, q_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
