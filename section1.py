import numpy as np
import functions as f

# Display the trajectory using tuples
def trajectory(grid, init, stocha, t):
    state = init
    policy = 1 # random movement (any other integer means 'always right')
    for i in range(t):
        next_action = f.get_policy(policy)
        step = f.move(state, next_action, grid)
		
        if(stocha and np.random.uniform(0,1) >= 0.5):
            step = (step[0], step[1], f.get_reward(grid, (0, 0)), step[0])
        else:
            state = step[3]
        
        print(step)


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
	
	# Deterministic: stocha = False; Stochastic: stocha = True
	stocha = True
	
	trajectory(domain, init, stocha, 10) 