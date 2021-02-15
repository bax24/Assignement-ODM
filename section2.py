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
	
	# Initial state
	init  = (3, 0)
	
	# Discount factor
	gamma = 0.99
	
	# Number of steps
	N = 916 # above that gamma**N < 0.0001 
	
	# Deterministic: stocha = False; Stochastic: stocha = True
	stocha = True
	
	dyn_matrix = np.full((N,domain.shape[0],domain.shape[1]),np.inf)
	
	J_values = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
 			state = (i, j)
 			
 			J_values[i,j] = f.expected_return(state,domain,gamma,N,stocha,dyn_matrix)
			 
	print(J_values)
