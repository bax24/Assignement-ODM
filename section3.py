import numpy as np
import functions as f

def reward_function(state, action, stocha, grid):
	if stocha:
		return (f.get_reward(grid, (0,0)) + f.get_reward(grid, f.move(state, action, grid)[3]))/2
	else:
		return f.get_reward(grid, f.move(state, action, grid)[3])
			

def probability(next_state, curr_state, action, stocha, grid):
	fct = f.move(curr_state, action, grid)[3]
	if fct == next_state:
		if stocha and f != (0,0):
			return 0.5
		else:
			return 1
	else:
		return 0
	

def Q_function(state, action, gamma, N, stocha, grid, matrix):
	if N == 0:
		return 0
	
	if action == (1, 0):
		ac = 0
	if action == (-1, 0):
		ac = 1
	if action == (0, -1):
		ac = 2
	if action == (0, 1):
		ac = 3
		
	m = matrix[N-1][state[0]][state[1]][ac]
	if not m == np.inf:
		return m
	
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	
	r = reward_function(state, action, stocha, grid) # r(x,u)
	
	sum_ = np.zeros(grid.size)
	
	for i in range(grid.shape[0]):
		for j in range(grid.shape[1]):
			x_prime = (i,j)
			p = probability(x_prime, state, action, stocha, grid) # p(x'|x,u)
			
			if p != 0: # no need to compute the Q_N-1 values if p is 0
				Qs = np.zeros(len(actions))
				j = 0
				for a in actions:
					Qs[j] = Q_function(x_prime, a, gamma, N-1, stocha, grid, matrix)
					j = j + 1
				sum_[i] = p * np.max(Qs)
				
	m = r + gamma * np.sum(sum_)
	matrix[N-1][state[0]][state[1]][ac] = m
	return m
			
			
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
	#N = 916 # above that gamma**N < 0.0001 
	#N = 687 # above that gamma**N < 0.001 
	N = 458 # above that gamma**N < 0.01 
	
	# Deterministic: stocha = False; Stochastic: stocha = True
	stocha = True
	
	dyn_matrix = np.full((N,domain.shape[0],domain.shape[1],4),np.inf)
	
	Q = Q_function(init, (-1,0), gamma, N, stocha, domain,dyn_matrix)