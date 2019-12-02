import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import random
import sys


def make_first_chart():
    rf0 = np.load('rf0.npy')
    rf1 = np.load('rf1.npy')
    rf2 = np.load('rf2.npy')
    rf3 = np.load('rf3.npy')
    rf4 = np.load('rf4.npy')
    
    rand0 = np.load('rand0.npy')
    rand1 = np.load('rand1.npy')
    rand2 = np.load('rand2.npy')
    rand3 = np.load('rand3.npy')
    rand4 = np.load('rand4.npy')

    rf0 = np.stack((rf0, rf1, rf2, rf3, rf4), axis=-1)
    rf0 = np.mean(rf0, axis=1)

    rand0 = np.stack((rand0, rand1, rand2, rand3, rand4), axis=-1)
    rand0 = np.mean(rand0, axis=1)

    advisor_mean = np.mean(rf0.reshape(-1, 10), axis=1)
    random_mean = np.mean(rand0.reshape(-1, 10), axis=1)

    x_axis = [i*10 for i in range(len(advisor_mean))]
    plt.plot(x_axis, advisor_mean, 'b')
    plt.plot(x_axis, random_mean, 'r')
    plt.legend(['Advisor Policy', 'Random Exploration'], loc='upper left')
    plt.xlabel('Training Iteration')
    plt.ylabel('Sum of Returns')
    plt.title('Exploration Training Progress')
    plt.savefig('chart_one.png')
    plt.show()
    plt.close()


def make_second_chart():
    rfall0 = np.load('rfall0.npy')
    rfall1 = np.load('rfall1.npy')
    rfall2 = np.load('rfall2.npy')
    rfall4 = np.load('rfall4.npy')

    num_rows = 50

    rfall0 = np.stack((rfall0, rfall1, rfall2, rfall4), axis=-1)
    rfall0 = np.mean(rfall0, axis=2)
    

    first = np.sum(rfall0[:num_rows], axis = 0) / num_rows
    last = np.sum(rfall0[-num_rows:], axis = 0) / num_rows

    fm = np.mean(first.reshape(-1, 10), axis=1)
    lm = np.mean(last.reshape(-1, 10), axis=1)

    x_axis = [i*10 for i in range(len(fm))]
    plt.plot(x_axis, fm, marker='*')
    plt.plot(x_axis, lm, marker = '^')
    plt.legend(['Iter: 0-50 R:{}'.format(np.sum(first)),
                'Iter: 450-500 R:{}'.format(np.sum(last))], loc='upper left')
    plt.xlabel('Exploitation Policy Training Episode')
    plt.ylabel('Average Return')
    plt.title('Progression of Learning Curves')
    plt.savefig('chart_two.png')
    plt.show()

if sys.argv[1] == 'plot':
    make_second_chart()
    make_first_chart()
