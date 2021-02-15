import numpy as np
import functions as f

# ---------- MAIN --------------
if __name__ == "__main__":
	
	# Grid
	domain = np.matrix([[ -3,   1,  -5,   0,  19],
					 [  6,   3,   8,   9,  10],
					 [  5,  -8,   4,   1,  -8],
					 [  6,  -9,   4,  19,  -5],
					 [-20, -17,  -4,  -3,   9]])
	
	# Set of actions U
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	
	# Initial state
	init  = (3, 0)
	
	# Discount factor
	gamma = 0.99
	
	# Number of steps
	N = 7
	
	# Deterministic: stocha = False; Stochastic: stocha = True
	stocha = True
	
	test = f.probability((0,2), (0,4), (0,1), stocha, domain)
	
	# mu* the optimal policy
	opt_policy = f.compute_policy(gamma, N, stocha, domain)
	
	dyn_matrix = np.full((916,domain.shape[0],domain.shape[1]),np.inf)
	
	# J_{mu*}^N for every state
	J_values = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
 			state = (i, j)
 			
 			J_values[i,j] = f.expected_return(state,domain,gamma,916,stocha,dyn_matrix,opt_policy)
			 
	print(J_values)		
	