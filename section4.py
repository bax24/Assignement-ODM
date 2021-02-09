import numpy as np
import math
import functions as f
import matplotlib.pyplot as plt


# Returns a random action
def get_policy(p):
    actions = ['down', 'up', 'left', 'right']
	
    if p == 1: # random 
        num = np.random.uniform(0, 4)
        i = math.floor(num)
    else: # always right
        i = 3
    
    return actions[i]

def reward_function(state, action, stocha, grid):
	if stocha:
		return (f.get_reward(grid, (0,0)) + f.get_reward(grid, f.move(state, action, grid)[3]))/2
	else:
		return f.get_reward(grid, f.move(state, action, grid)[3])
    


# Display the trajectory using tuples
def trajectory(grid, init, stocha, t):
    state = init
    random_policy = True # random movement (if false then 'always right')
    traj = []
    for i in range(t):
        next_action = f.get_policy(random_policy)
        step = f.move(state, next_action, grid)
        
        if (stocha and np.random.uniform(0,1) > 0.5):
            step = (step[0], step[1], f.get_reward(grid, (0, 0)), (0,0))
        
        state = step[3]
        #print(step)
        traj.append(step)
    return traj


# ---------- MAIN --------------
if __name__ == "__main__":
	
	# Grid
    domain = np.matrix([[ -3,   1,  -5,   0,  19],
					 [  6,   3,   8,   9,  10],
					 [  5,  -8,   4,   1,  -8],
					 [  6,  -9,   4,  19,  -5],
					 [-20, -17,  -4,  -3,   9]])
	
	# Initial state
    init  = (3, 0)
    
    # Discount factor
    gamma = 0.99
    	
    # Lengths of the trajectories
    N = [10, 100, 1000, 10000, 1000000] 
    	
    # Deterministic: stocha = False; Stochastic: stocha = True
    stocha = True
    
    # Estimation of reward
    r_hat = []
    p_hat = []
    for i in range(1, 10000, 100):
        ht = trajectory(domain, init, stocha, i)
        rew = []
        prob = []
        for j in range(i):
            r = ht[j][2]
            rew.append(r)
            if f.move(ht[j][0], ht[j][1], domain)[3] == ht[j][3]:
                prob.append(1)
        r_hat.append(np.mean(rew))
        p_hat.append(len(prob)/i)
        
    r = plt.plot(range(1, 10000, 100), r_hat, label = 'r')
    p = plt.plot(range(1, 10000, 100), p_hat, label = 'p')
    plt.legend(handles=[r, p], loc = 'lower right')
    plt.show()
    



