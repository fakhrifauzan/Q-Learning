import numpy as np
import random

load_reward = np.loadtxt("DataTugasML3.txt", dtype='i', delimiter='\t')
reward = load_reward[::-1]
q = np.zeros((10,10), dtype=int)
gamma = 0.8

current = random.randint(0,9)

def next_action(current) :
    next = []
    for i in range(reward[current].size):
        if(reward[current, i] != -1) :
            next.append(i)
    return random.choice(next)

print(reward)
print(current)
print(next_action(current))

# print(reward[::-1])