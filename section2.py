import numpy as np
import functions as f

def expected_return(state,grid,gamma,N,stocha,matrix):
	
	policy = 2 # random movement (any other integer means 'always right')
	
	if N == 0:
		return 0
	else:
		m = matrix[N-1][state[0]][state[1]]
		if not m == np.inf:
			return m
		else:
			step = f.move(state, f.get_policy(policy), grid)
			rew = f.get_reward(grid, step[3])
			
			if not stocha:
				m = rew + gamma*expected_return(step[3],grid,gamma,N-1,stocha,matrix)
				matrix[N-1][state[0]][state[1]] = m
				return m
			else:
				exp_rew = (-3 + rew)/2
				exp_ret_no_disturb = expected_return(step[3],grid,gamma,N-1,stocha,matrix)
				exp_ret_disturb = expected_return((0, 0),grid,gamma,N-1,stocha,matrix)
				
				m = exp_rew + gamma*(exp_ret_no_disturb + exp_ret_disturb)/2
				matrix[N-1][state[0]][state[1]] = m
				return  m
			


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
	stocha = False
	
	dyn_matrix = np.full((N,domain.shape[0],domain.shape[1]),np.inf)
	
	#J = expected_return(init,domain,gamma,N,stocha,dyn_matrix)
	
	J_values = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
 			state = (i, j)
 			
 			J_values[i,j] = expected_return(state,domain,gamma,N,stocha,dyn_matrix)
