import numpy as np
import math
import functions as f


def sect5_1():
	# Trajectory length
	T = 10000
	
	# Trajectory of wanted size
	ht = f.trajectory(domain, init, stocha, T)
	
	Q_values = np.zeros((domain.size,4))
	
	Q_learning(ht,alpha,gamma,Q_values,0,T)
	
	policy = compute_policy_Q_learning(Q_values, domain)
	
	dyn_matrix = np.full((916,domain.shape[0],domain.shape[1]),np.inf)
	
	# J^N_{mu^*} for every state
	J_values = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
 			state = (i, j)
 			
 			J_values[i,j] = f.expected_return(state,domain,gamma,916,stocha,dyn_matrix,policy)
			 
	return policy, J_values


def Q_learning(ht, alpha, gamma, Q_values, k, T):
	
	while not k == T:
		x_k = ht[k][0]
		u_k = ht[k][1]
		r_k = ht[k][2]
		x_next = ht[k][3]
		
		index_x_k = x_k[0]*5 + x_k[1]
		index_x_next = x_next[0]*5 + x_next[1]
		
		if u_k == (1, 0):
			ac = 0
		if u_k == (-1, 0):
			ac = 1
		if u_k == (0, -1):
			ac = 2
		if u_k == (0, 1):
			ac = 3
			
		Q_values[index_x_k][ac] = (1 - alpha)*Q_values[index_x_k][ac] + \
			 alpha*(r_k + gamma*Q_values[index_x_next].max())
		
		k = k + 1
		
		
def epsilon_greddy_Q_learning(ep, trans, x_0, grid, alpha, gamma, epsilon, rate_alpha = 1):
	
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	
	Q = np.zeros((domain.size,4))
	
	for e in range(ep):
		state = x_0
		alpha_t = alpha
		for t in range(trans):
			index = state[0]*5 + state[1]
			
			ac = get_action_id(Q,state,epsilon)
			
			step = f.move(state, actions[ac], grid)
			
			index_next = step[3][0]*5 + step[3][1]
				
			Q[index][ac] = (1 - alpha_t)*Q[index][ac] + alpha_t*(step[2] + gamma*Q[index_next].max()) 
			
			state = step[3]
			alpha_t = rate_alpha*alpha_t
	
	return Q


def get_action_id(Q, state, epsilon):
	r = np.random.uniform(0,1)
	
	if r < epsilon:
		return math.floor(np.random.uniform(0,4))
	else:
		index = state[0]*5 + state[1]
		return np.argmax(Q[index])


def compute_policy_Q_learning(Q, grid):
	
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	
	policies = []
	
	for i in range(grid.shape[0]):
		policies_i = []
		for j in range(grid.shape[1]):
			Q_max = -1*np.inf
			action = (2, 2)
			for a in range(len(actions)):
				Q_value = Q[i*5 + j][a]
				
				if Q_value > Q_max:
					Q_max = Q_value
					action = actions[a]
			
			policies_i.append(action)
		policies.append(policies_i)
		
	return policies
		
		 
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
	
	# Learning rate 
	alpha = 0.05
	
	# Probability to explore
	epsilon = 0.25
	
	if False:
		policy, J_values = sect5_1()
		
	Q = epsilon_greddy_Q_learning(100, 1000, init, domain, alpha, gamma, epsilon)
		