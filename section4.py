import numoy as np
import math

# Gets reward of a certain state on the grid
def get_reward(grid, state):
    r = grid[state[0], state[1]]
    return r

# Apply action to a state on a grid (determinisitc or stochastic)
def move(state, action, grid):
    n = grid.shape[0]
    m = grid.shape[1]
    new_state = state

    if(action=='down'):
        new_state = (min(state[0] + 1, n-1), state[1])
        action = (1, 0)
    if(action=='up'):
        new_state = (max(state[0] - 1, 0), state[1])
        action = (-1, 0)
    if(action=='right'):
        new_state = (state[0], min(state[1] + 1, m-1))
        action = (0, 1)
    if(action=='left'):
        new_state = (state[0], max(state[1] - 1, 0))
        action = (0, -1)
		
    reward = get_reward(grid, new_state)
        
    return state, action, reward, new_state


# Returns a random action
def get_policy(p):
    actions = ['down', 'up', 'left', 'right']
	
    if p == 1: # random 
        num = np.random.uniform(0, 4)
        i = math.floor(num)
    else: # always right
        i = 3
    
    return actions[i]



