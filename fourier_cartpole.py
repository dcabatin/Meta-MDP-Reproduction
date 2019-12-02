import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import count
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import time

def random_ratio(lower, upper):
    return lower + (upper-lower) * random.random()

def set_environmental_variables(env, length = 0.5, masspole = 0.5, \
                               masscart = 1.0, force_mag = 10.0):
    env.length = length
    env.masspole = masspole
    env.masscart = masscart
    env.force_mag = force_mag

def randomize_environment(env, lower = 0.5, upper = 2):
    set_environmental_variables(env,
                                env.length * random_ratio(lower, upper),
                                env.masspole * random_ratio(lower, upper),
                                env.masscart * random_ratio(lower, upper),
                                env.force_mag * random_ratio(lower, upper))
    return env

class REINFORCE(nn.Module):
    def __init__(self, order=5, gamma = 0.99, num_actions = 2):
        super(REINFORCE, self).__init__()
        self.gamma = gamma
    
        #self.fourier_weights = torch.randint(0, order, [4, order], dtype=torch.float)
        self.fourier_weights = nn.Linear(4, order)
        self.affine = nn.Linear(order, num_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        norm_state = self.normalize(x)
        weighted = self.fourier_weights(x)
        transformed = torch.cos(np.pi * weighted)
        action_scores = self.affine(transformed)

        return F.softmax(action_scores, dim=1)

    def normalize(self, x):
        x += torch.tensor([2.4, 1, 12, 2.5])
        x /= torch.tensor([4.8, 2, 24, 5.0])
        return x

    def select_action(self, state, training=True):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        if training:
            action = m.sample()
        else:
            action = torch.argmax(probs)
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self, optimizer):
        R = 0
        self_loss = []
        n = len(self.rewards)
        returns = [0 for i in range(n)]
        for i in reversed(range(n)):
            if i == n-1:
                returns[i] = self.rewards[i]
            else:
                returns[i] = self.gamma * returns[i+1] + self.rewards[i]
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-06)
        for log_prob, R in zip(self.saved_log_probs, returns):
            self_loss.append(-log_prob * R)
        optimizer.zero_grad()
        self_loss = torch.cat(self_loss).sum()
        self_loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

class RandomAgent():
    def __init__(self, num_actions = 2):
        self.rewards = []
        self.num_actions = num_actions

    def forward(self, x):
        pass

    def select_action(self, state, training=True):
        return np.random.randint(self.num_actions)

    def finish_episode(self, optimizer):
        del self.rewards[:]

def train(advisor,
          advisor_optimizer = None,
          Imeta = 500,
          I = 1000,
          T = 10000,
          max_itrs = 20000,
          exploiter_learning_rate = 1e-2,
          return_all = False):

    start = time.time()

    reward_sums = []

    if return_all:
        all_returns = []
    
    env = gym.make('CartPole-v1')
    for meta_episode in range(Imeta):
        print("starting meta-iteration " + str(meta_episode))
        print(time.time() - start)
        eps = 0.8 / 0.995

        env.reset()
        env = randomize_environment(env)
        exploiter = REINFORCE(order=3)
        exploiter_optimizer = optim.Adam(exploiter.parameters(),
                                         lr=exploiter_learning_rate)
        reward_sum = 0
        
        if return_all:
            all_returns.append([])

        # training phase
        for i_episode in range(I):
            eps *= 0.995

            state, curr_reward = env.reset(), 0
            for t in range(T):  # Don't infinite loop while learning
                action_advisor = advisor.select_action(state)
                action_exploiter = exploiter.select_action(state)
                if random.random() < eps:
                    action = action_advisor
                else:
                    action = action_exploiter
                state, reward, done, _ = env.step(action)
                if return_all:
                    curr_reward += reward
                reward_sum += reward
                exploiter.rewards.append(reward)
                advisor.rewards.append(reward)
                if done:
                    break
            exploiter.finish_episode(exploiter_optimizer)
            if return_all:
                all_returns[-1].append(curr_reward)

        if meta_episode % 5 == 4:
            print("avg reward per episode was " + str(len(advisor.rewards)/5000))
            advisor.finish_episode(advisor_optimizer)

        reward_sums.append(reward_sum)

    if return_all:
        return reward_sums, all_returns
    else:
        return reward_sums

def main():
    Imeta = 500      # meta episodes
    I = 1000         # episodes per meta episode
    T = 500        # episodes per inner agent
    max_itrs = 2000 # max iterations for inner agent per trial on CartPole

    advisor_learning_rate = 1e-1
    exploiter_learning_rate = 1e-2
    gamma = 0.99
    if sys.argv[1] == 'advisor':
        start_time = time.time()

        advisor = REINFORCE(gamma = gamma, order=9)
        advisor_optimizer = optim.Adam(advisor.parameters(), lr=advisor_learning_rate)
        advisor_returns_reinforce, \
        advisor_returns_reinforce_all = train(advisor,
                                              advisor_optimizer,
                                              Imeta,
                                              I,
                                              T,
                                              max_itrs,
                                              exploiter_learning_rate,
                                              return_all = True)
        print("REINFORCE --- %s seconds ---" % (time.time() - start_time))

        np.save('advisor_returns_reinforce.npy',
                np.array(advisor_returns_reinforce))
        np.save('advisor_returns_reinforce_all.npy',
                np.array([np.array(row) for row in advisor_returns_reinforce_all]))


    if sys.argv[1] == 'random':
        start_time = time.time()


        advisor = RandomAgent()
        advisor_optimizer = None
        advisor_returns_random = train(advisor,
                                      advisor_optimizer,
                                      Imeta,
                                      I,
                                      T,
                                      max_itrs,
                                      exploiter_learning_rate)

        print("RANDOM    --- %s seconds ---" % (time.time() - start_time))

        np.save('advisor_returns_random.npy',
                np.array(advisor_returns_random))

main()

def make_first_chart():
    advisor_returns_reinforce = np.load('advisor_returns_reinforce.npy')
    advisor_returns_random = np.load('advisor_returns_random.npy')

    x_axis = list(range(len(advisor_returns_reinforce)))
    plt.plot(x_axis, advisor_returns_reinforce, 'b')
    plt.plot(x_axis, advisor_returns_random, 'r')
    plt.legend(['Advisor Policy', 'Random Exploration'], loc='upper left')
    plt.xlabel('Training Iteration')
    plt.ylabel('Sum of Returns')
    plt.title('Exploration Training Progress')
    plt.savefig('chart_one.png')
    plt.close()


def make_second_chart():
    advisor_returns_reinforce_all = np.load('advisor_returns_reinforce_all.npy')

    num_rows = 5

    first = np.sum(advisor_returns_reinforce_all[:num_rows], axis = 0) / num_rows
    last = np.sum(advisor_returns_reinforce_all[-num_rows:], axis = 0) / num_rows

    x_axis = list(range(len(first)))
    plt.plot(x_axis, first, marker='*')
    plt.plot(x_axis, last, marker = '^')
    plt.legend(['Iter: 0-50 R:{}'.format(np.sum(first)),
                'Iter: 450-500 R:{}'.format(np.sum(last))], loc='upper left')
    plt.xlabel('Exploitation Policy Training Episode')
    plt.ylabel('Average Return')
    plt.title('Progression of Learning Curves')
    plt.savefig('chart_two.png')

if sys.argv[1] == 'plot':
    make_second_chart()
    make_first_chart()
