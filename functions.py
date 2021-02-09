import numpy as  np
import math

# Gets reward of a certain state on the grid
def get_reward(grid, state):
    r = grid[state[0], state[1]]
    return r

# Apply action to a state on a grid (determinisitc or stochastic)
def move(state, action, grid):
    n, m = grid.shape[0], grid.shape[1]
    i, j = action[0], action[1]    
    new_state = (min(max(state[0] + i, 0), n-1), min(max(state[1] + j, 0), m-1))
		
    reward = get_reward(grid, new_state)
        
    return state, action, reward, new_state

# Returns a random action
def get_policy(random):
    actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	
    if random: # random 
        num = np.random.uniform(0, 4)
        i = math.floor(num)
    else: # always right
        i = 3
    
    return actions[i]