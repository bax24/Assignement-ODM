import numpy as np
import funtions as f

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
	

def Q_function(state, action, gamma, N, stocha, grid): # May delete it later
	if N == 0:
		return 0
	
	actions = ['down', 'up', 'left', 'right']
	
	r = reward_function(state, action, stocha, grid)
	
	sum_ = np.zeros(grid.size)
	
	for i in range(grid.shape[0]):
		for j in range(grid.shape[1]):
			x_prime = (i,j)
			p = probability(x_prime, state, action, stocha, grid)
			
			if p != 0: # no need to compute the Q_N-1 values if p is 0
				Qs = np.zeros(len(actions))
				j = 0
				for a in actions:
					Qs[j] = Q_function(x_prime, a, gamma, N-1, stocha, grid)
					j = j + 1
				sum_[i] = p * np.max(Qs)
			
	return r + gamma * np.sum(sum_)
			
			
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
	
	Q = Q_function(init, 'up', gamma, 8, stocha, domain)
	
	# Faut enncore déterminer la valuer de N, faire mu* et J^N_mu* mais pas sûr de capter