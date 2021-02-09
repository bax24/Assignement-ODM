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


def compute_policy(gamma, N, stocha, grid):
	dyn_matrix = np.full((N,domain.shape[0],domain.shape[1],4),np.inf)
	
	policies = []
	
	for i in range(domain.shape[0]):
		policies_i = []
		for j in range(domain.shape[1]):
			Q_max = -1*np.inf
			action = (2, 2)
			for a in actions:
				Q_value = Q_function((i, j), a, gamma, N, stocha, domain, dyn_matrix)
				
				if Q_value > Q_max:
					Q_max = Q_value
					action = a
			
			policies_i.append(action)
		policies.append(policies_i)
		
	return policies


def expected_return(state,grid,gamma,N,stocha,matrix,policy):
	
	if N == 0:
		return 0
	
	m = matrix[N-1][state[0]][state[1]]
	if not m == np.inf: # value of J already evaluated
		return m
	
	step = f.move(state, policy[state[0]][state[1]], grid)
	rew = f.get_reward(grid, step[3])
	
	if not stocha:
		m = rew + gamma*expected_return(step[3],grid,gamma,N-1,stocha,matrix,policy)
		matrix[N-1][state[0]][state[1]] = m
		return m
	else:
		exp_rew = (-3 + rew)/2
		exp_ret_no_disturb = expected_return(step[3],grid,gamma,N-1,stocha,matrix,policy)
		exp_ret_disturb = expected_return((0, 0),grid,gamma,N-1,stocha,matrix,policy)
		
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
	
	# mu* the optimal policy
	opt_policy = compute_policy(gamma, N, stocha, domain)
	
	dyn_matrix = np.full((916,domain.shape[0],domain.shape[1]),np.inf)
	
	# J_N^{mu*} for every state
	J_values = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
 			state = (i, j)
 			
 			J_values[i,j] = expected_return(state,domain,gamma,916,stocha,dyn_matrix,opt_policy)
			 
	print(J_values)		
	