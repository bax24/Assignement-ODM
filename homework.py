#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:12:31 2021

@author: bax
"""

import numpy as np
import math

domain = np.matrix([[ -3,   1,  -5,   0,  19],
                    [  6,   3,   8,   9,  10],
                    [  5,  -8,   4,   1,  -8],
                    [  6,  -9,   4,  19,  -5],
                    [-20, -17,  -4,  -3,   9]])

init  = (3, 0)
stocha = True


# Gets reward of a certain state on the grid
def get_reward(grid, state):
    r = grid[state[0], state[1]]
    return r

# Apply action to a state on a grid (determinisitc or stochastic)
def move(state, action, grid):
    n = grid.shape[0]
    m = grid.shape[1]
    reward = get_reward(grid, state)
    new_state = state

    if(action=='down'):
        new_state = (min(state[0] + 1, n-1), state[1])
    if(action=='up'):
        new_state = (max(state[0] - 1, 0), state[1])
    if(action=='right'):
        new_state = (state[0], min(state[1] + 1, m-1))
    if(action=='left'):
        new_state = (state[0], max(state[1] - 1, 0))
        
    return state, action, reward, new_state

# Returns a random action
def get_policy():
    actions = ['down', 'up', 'left', 'right']
    num = np.random.uniform(0, 4)
    i = math.floor(num)
    
    return actions[i]

# Display the trahectory using tuples
def trajectory(grid, init, stocha):
    state = init
        
    for i in range(10):
        if(stocha and np.random.uniform(0,1) > 0.5):
            next_action = 'none'
        else:
            next_action = get_policy()
        step = move(state, next_action, grid)
        state = step[3]
        print(step)

trajectory(domain, init, stocha)      

    
    

    

