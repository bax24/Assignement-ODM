import numpy as np
import functions as f


def expected_return(state,grid,gamma,N,stocha):
	
	policy = 1 # random movement (any other integer means 'always right')
	
	if N == 0:
		return 0
	else:
		w = np.array([0])
		if stocha:
			np.random.uniform(0,1,1000)
		
		J_N_values = np.zeros(w.size)
		
		for i in range(w.size):
			if(stocha and w[i] >= 0.5):
				step = f.move(state, 'none', grid)
				rew = f.get_reward(grid, (0,0))
			else:
				step = f.move(state,  f.get_policy(policy), grid)
				rew = f.get_reward(grid, step[3])
			
			J_N_values[i] = rew + gamma*expected_return(step[3],grid,gamma,N-1,stocha)
			
		return np.mean(J_N_values)
			
	


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
	
	# Number of steps
	N = 916 # above that gamma**N < 0.0001 
	
	# Deterministic: stocha = False; Stochastic: stocha = True
	stocha = True
	
	J_values = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
			state = (i, j)
			
			J_values[i,j] = expected_return(state,domain,gamma,N,stocha)
