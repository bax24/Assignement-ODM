import numpy as  np
import math


# ----------------------------------------------------------------------
# -------------------- 'BASIC' FUNCTIONS ---------------------------------
# ----------------------------------------------------------------------

# Gets reward of a certain state on the grid
def get_reward(grid, state):
	"""
	:param grid: domain instance
	:type grid: matrix of int32
	:param state: state of the domain
	:type state: 2-tuple
	:return: returns the reward contained in 'state' in the 'grid'
	:rtype: int32
	"""
	r = grid[state[0], state[1]]
	return r

# Apply action to a state on a grid (determinisitc or stochastic)
def move(state, action, grid):
	"""
	:param state: state of the domain
	:type state: 2-tuple
	:param action: action to be executed from the 'state'
	:type action: 2-tuple
	:param grid: domain instance
	:type grid: matrix of int32
	:return: returns a 4-tuple (state, action, reward, new_state) where new_state
	         is the state in the domain reached from 'state' by executing 'action'
			 and reward is the value in 'new_state'
	:rtype: 4-tuple
	"""
	n = grid.shape[0]
	m = grid.shape[1]
	i, j = action[0], action[1]
	new_state = state
	
	new_state = (min(max(state[0] + i, 0), n-1), min(max(state[1] + j, 0), m-1))
	
	reward = get_reward(grid, new_state)
	
	return state, action, reward, new_state

# Returns a random action
def get_policy(random):
	"""
	:param p: choice of policy among random policy and always right policy
	:type p: int or bool
	:return: the action corresponding to the chosen policy
	:rtype: 2-tuple
	"""
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	
	if random: # random
		num = np.random.uniform(0, 4)
		i = math.floor(num)
	else: # always right
		i = 3
		
	return actions[i]


# Display the trajectory using tuples
def trajectory(grid, init, stocha, t):
    state = init
    random_policy = True # random movement (if false then 'always right')
    traj = []
    for i in range(t):
        next_action = get_policy(random_policy)
        step = move(state, next_action, grid)
        
        if (stocha and np.random.uniform(0,1) > 0.5):
            step = (step[0], step[1], get_reward(grid, (0, 0)), (0,0))
        
        state = step[3]
        #print(step)
        traj.append(step)
    return traj


def expected_return(state,grid,gamma,N,stocha,matrix,policy = True):
	"""
	:param state: one state of the domain
	:type state: 2-tuple
	:param grid: domain instance
	:type grid: matrix of int32
	:param gamma: value of the decay factor
	:type gamma: float
	:param N: number of steps in the recursion function J
	:type N: int
	:param stocha: tells whether or not the domain is stochastic
	:type stocha: bool
	:param matrix: matrix containing the value of the potential already computed values of J
	:type matrix: matrix of size N x grid.shape[0] x grid.shape[1]
	:optional param policy: matrix of actions to take for each corresponding state int 'grid'
	:type policy: matrix of 2-tuples of size grid.shape[0] x grid.shape[1]
	:return: returns the value of J_N^mu*(state)
	:rtype: float
	"""
	
	if N == 0:
		return 0
	
	m = matrix[N-1][state[0]][state[1]]
	if not m == np.inf: # value of J already evaluated
		return m
	
	if type(policy) == type(True):
		step = move(state, get_policy(policy), grid)
		rew = get_reward(grid, step[3])
	else:
		step = move(state, policy[state[0]][state[1]], grid)
		rew = get_reward(grid, step[3])
	
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

# ---------------------------------------------------------------------
# --------------------- Q_N functions elements ------------------------
# ---------------------------------------------------------------------
def reward_function(state, action, stocha, grid):
	"""
	:param state: state of the domain
	:type state: 2-tuple
	:param action: action to be executed from the 'state'
	:type action: 2-tuple
	:param stocha: tells whether or not the domain is stochastic
	:type stocha: bool
	:param grid: domain instance
	:type grid: matrix of int32
	:return: returns the expected reward from 'state' executing 'action' in 'grid'
	:rtype: float
	"""
	
	if stocha:
		return (get_reward(grid, (0,0)) + get_reward(grid, move(state, action, grid)[3]))/2
	else:
		return get_reward(grid, move(state, action, grid)[3])
			

def probability(next_state, curr_state, action, stocha, grid):
	"""
	:param next_state: one state of the domain
	:type next_state: 2-tuple
	:param curr_state: one state of the domain
	:type curr_state: 2-tuple
	:param action: action to be executed from the 'state'
	:type action: 2-tuple
	:param stocha: tells whether or not the domain is stochastic
	:type stocha: bool
	:param grid: domain instance
	:type grid: matrix of int32
	:return: returns the probability of reaching 'next_state' from 'curr_state'
	         executing 'action' in 'grid'
	:rtype: float
	"""
	
	fct = move(curr_state, action, grid)[3]
	if fct == next_state:
		if stocha and fct != (0,0):
			return 0.5
		else:
			return 1
	else:
		if next_state == (0,0) and stocha:
			return 0.5
		else:
			return 0
	

def Q_function(state, action, gamma, N, stocha, grid, matrix):
	"""
	:param state: one state of the domain
	:type state: 2-tuple
	:param action: action to be executed from the 'state'
	:type action: 2-tuple
	:param gamma: value of the decay factor
	:type gamma: float
	:param N: number of steps in the recursion function J
	:type N: int
	:param stocha: tells whether or not the domain is stochastic
	:type stocha: bool
	:param grid: domain instance
	:type grid: matrix of int32
	:param matrix: matrix containing the value of the potential already computed values of Q
	:type matrix: matrix of size N x grid.shape[0] x grid.shape[1] x #actions
	:return: returns the evaluation of Q_N(state,action)
	:rtype: float
	"""
	
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
				k = 0
				for a in actions:
					Qs[k] = Q_function(x_prime, a, gamma, N-1, stocha, grid, matrix)
					k = k + 1
				sum_[i] = p * np.max(Qs)
				
	m = r + gamma * np.sum(sum_)
	matrix[N-1][state[0]][state[1]][ac] = m
	return m


def compute_r_p_values(grid,stocha):
	"""
	:param grid: domain instance
	:type grid: matrix of int32
	:param stocha: tells whether or not the domain is stochastic
	:type stocha: bool
	:return: returns all values of r(u,x) and p(x'|x,u) according to the grid
	"""
	
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]

	r_values = np.zeros((grid.size,len(actions)))
	p_values = np.zeros((grid.size,grid.size,len(actions)))
	
	for k in range(grid.size):
		x = math.floor(k/5)
		y = k % 5
		state = (x,y) # state x
		
		for a in range(len(actions)):
			
			r_values[k][a] = reward_function(state, actions[a], stocha, grid) # r(x,u)
			
			for l in range(grid.size):
				x = math.floor(l/5)
				y = l % 5
				next_state = (x,y) # state x'
				
				p_values[l][k][a] = probability(next_state, state, actions[a], stocha, grid) # p(x'|x,u)
				
	return r_values,p_values


def compute_policy(gamma, N, stocha, grid):
	"""
	:param gamma: value of the decay factor
	:type gamma: float
	:param N: number of steps in the recursion function Q
	:type N: int
	:param stocha: tells whether or not the domain is stochastic
	:type stocha: bool
	:param grid: domain instance
	:type grid: matrix of int32
	:return: returns the matrix of actions to take for each corresponding state int 'grid'
	:rtype: matrix of 2-tuples of size grid.shape[0] x grid.shape[1]
	"""
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	
	dyn_matrix = np.full((N,grid.shape[0],grid.shape[1],4),np.inf)
	
	policies = []
	
	for i in range(grid.shape[0]):
		policies_i = []
		for j in range(grid.shape[1]):
			Q_max = -1*np.inf
			action = (2, 2)
			for a in actions:
				Q_value = Q_function((i, j), a, gamma, N, stocha, grid, dyn_matrix)
				
				if Q_value > Q_max:
					Q_max = Q_value
					action = a
			
			policies_i.append(action)
		policies.append(policies_i)
		
	return policies


# ----------------------------------------------------------------------
# --------------- Q^_N functions elements ------------------------------
# ----------------------------------------------------------------------

def reward_hat(state, action, traj):
	"""
	:param state: one state of the domain
	:type state: 2-tuple
	:param action: action to be executed from the 'state'
	:type action: 2-tuple
	:param traj: trajectory with random uniform policy
	:type traj: list of lists
	:return: returns r^(state, action) according to the given trajectory
	:rtype: float
	"""
	rew = []
	r_hat = 0
	for j in range(len(traj)):
		if traj[j][0] == state and traj[j][1] == action: # (x_k, u_k) = (x, u)
			r = traj[j][2]
			rew.append(r)
	if not len(rew) == 0:
		r_hat = np.mean(rew)
	return r_hat

	
def probability_hat(next_state, curr_state, action, traj):
	"""
	:param next_state: one state of the domain
	:type next_state: 2-tuple
	:param curr_state: one state of the domain
	:type curr_state: 2-tuple
	:param action: action to be executed from the 'state'
	:type action: 2-tuple
	:param traj: trajectory with random uniform policy
	:type traj: list of lists
	:return: returns p^(next_state | curr_state, action) according to the given trajectory
	:rtype: float
	"""
	prob = []
	p_hat = 0
	for j in range(len(traj)):
		#print('j = ' + str(j) + '; state: ' + str(traj[j][0]) + '  action: ' + str(traj[j][1]))
		if traj[j][0] == curr_state and traj[j][1] == action: # (x_k, u_k) = (x, u)
			if traj[j][3] == next_state: # x_{k+1} = x'
				prob.append(1)
			else:
				prob.append(0)
	if not len(prob) == 0:
		p_hat = np.mean(prob)
	return p_hat


def Q_function_hat(state, action, gamma, N, r_values, p_values, grid, matrix):
	"""
	:param state: one state of the domain
	:type state: 2-tuple
	:param action: action to be executed from the 'state'
	:type action: 2-tuple
	:param gamma: value of the decay factor
	:type gamma: float
	:param N: number of steps in the recursion function J
	:type N: int
	:param r_values: all values of r^(u,x)
	:param p_values: all values of p^(x'|x,u)
	:param grid: domain instance
	:type grid: matrix of int32
	:param matrix: matrix containing the value of the potential already computed values of Q
	:type matrix: matrix of size N x grid.shape[0] x grid.shape[1] x #actions
	:return: returns the evaluation of Q^_N(state,action)
	:rtype: float
	"""
	
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
	
	index = state[0]*5 + state[1]	
	r = r_values[index][ac] # r^(x,u)
	
	sum_ = np.zeros(grid.size)
	
	for i in range(grid.shape[0]):
		for j in range(grid.shape[1]):
			x_prime = (i,j)
			index_prime = i*5 + j
			p = p_values[index_prime][index][ac] # p^(x'|x,u)
			
			if p != 0: # no need to compute the Q^_N-1 values if p is 0
				Qs = np.zeros(len(actions))
				k = 0
				for a in actions:
					Qs[k] = Q_function_hat(x_prime, a, gamma, N-1, r_values, p_values, grid, matrix)
					k = k + 1
				sum_[i] = p * np.max(Qs)
				
	m = r + gamma * np.sum(sum_)
	matrix[N-1][state[0]][state[1]][ac] = m
	return m


def compute_r_hat_p_hat_values(ht,grid):
	"""
	:param ht: trajectory with random uniform policy
	:type ht: list of lists
	:param grid: domain instance
	:type grid: matrix of int32
	:return: return all values of r^(u,x) and p^(x'|x,u) according to
			 the grid and the trajectory ht
	"""
	
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]

	r_hat_values = np.zeros((grid.size,len(actions)))
	p_hat_values = np.zeros((grid.size,grid.size,len(actions)))
	
	for k in range(grid.size):
		x = math.floor(k/5)
		y = k % 5
		state = (x,y) # state x
		
		for a in range(len(actions)):
			
			r_hat_values[k][a] = reward_hat(state, actions[a], ht) # r^(x,u)
			
			for l in range(grid.size):
				x = math.floor(l/5)
				y = l % 5
				next_state = (x,y) # state x'
				
				p_hat_values[l][k][a] = probability_hat(next_state, state, actions[a], ht) # p^(x'|x,u)
				
	return r_hat_values,p_hat_values


def compute_policy_hat(ht,gamma, N, grid):
	"""
	:param ht: trajectory with random uniform policy
	:type ht: list of lists
	:param gamma: value of the decay factor
	:type gamma: float
	:param N: number of steps in the recursion function Q
	:type N: int
	:param grid: domain instance
	:type grid: matrix of int32
	:return: returns the matrix of actions to take for each corresponding state int 'grid'
	:rtype: matrix of 2-tuples of size grid.shape[0] x grid.shape[1]
	"""
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	
	dyn_matrix = np.full((N,grid.shape[0],grid.shape[1],4),np.inf)
	
	r_values, p_values = compute_r_hat_p_hat_values(ht,grid)
	
	policies = []
	
	for i in range(grid.shape[0]):
		policies_i = []
		for j in range(grid.shape[1]):
			Q_max = -1*np.inf
			action = (2, 2)
			for a in actions:
				Q_value = Q_function_hat((i, j), a, gamma, N, r_values, p_values, grid, dyn_matrix)
				
				if Q_value > Q_max:
					Q_max = Q_value
					action = a
			
			policies_i.append(action)
		policies.append(policies_i)
		
	return policies

# ------------------------------------------------------------------
# -------------------- 'COMPARISON' FUNCTIONS ----------------------
# ------------------------------------------------------------------

def inf_norm_Q(N,gamma,stocha,ht,grid):
	"""
	:param N: number of steps in the recursion function Q
	:type N: int
	:param gamma: value of the decay factor
	:type gamma: float
	:param stocha: tells whether or not the domain is stochastic
	:type stocha: bool
	:param ht: trajectory with random uniform policy
	:type ht: list of lists
	:param grid: domain instance
	:type grid: matrix of int32
	:return: returns the value of the infinite norm ||Q - Q^||_inf
	:rtype: float
	"""
	
	actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
	
	dyn_matrix = np.full((N,grid.shape[0],grid.shape[1],4),np.inf)
	dyn_matrix_hat = np.full((N,grid.shape[0],grid.shape[1],4),np.inf)
	
	Q_values = np.zeros((grid.size,len(actions)))
	Q_hat_values = np.zeros((grid.size,len(actions)))
	
	r_values, p_values = compute_r_hat_p_hat_values(ht,grid)		
	
	for i in range(grid.size):
		x = math.floor(i/5)
		y = i % 5
		state = (x,y) # state x
		
		for a in range(len(actions)):
			
			Q_values[i][a] = Q_function(state, actions[a], gamma, N, stocha, grid, dyn_matrix)
			Q_hat_values[i][a] = Q_function_hat(state, actions[a], gamma, N, r_values, p_values, grid, dyn_matrix_hat)
			
			diff_Q = abs(Q_values - Q_hat_values)
			
	return diff_Q.max()


# -----------------------------------------------------------------
# ----------------------- Q-LEARNING ------------------------------
# -----------------------------------------------------------------

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
		

def epsilon_greddy_Q_learning(ep, trans, x_0, grid, alpha, gamma, epsilon, rate_alpha = 1, protocol = 1):
    actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    Q = np.zeros((grid.size,4))
    Q_max = np.zeros((ep,grid.shape[0],grid.shape[1])) # contains for each episode and each state the maximum Q(x,u)
    if protocol == 3:
        buffer = []
    
    for e in range(ep):
    	state = x_0
    	alpha_t = alpha
    	for t in range(trans):
            index = state[0]*5 + state[1]
            ac = get_action_id(Q,state,epsilon)
            step = move(state, actions[ac], grid)
            if protocol == 3:
                buffer.append(step)
                for i in range(10):
                    rand = np.random.uniform(0,len(buffer))
                    rand = math.floor(rand)
                    step = buffer[rand]
                    index_next = step[3][0]*5 + step[3][1]
                    Q[step[0][0]*5 + step[0][1]][ac] = (1 - alpha_t)*Q[step[0][0]*5 + step[0][1]][ac] + alpha_t*(step[2] + gamma*Q[index_next].max())
                    state = step[3]
            else:
                index_next = step[3][0]*5 + step[3][1]
                Q[index][ac] = (1 - alpha_t)*Q[index][ac] + alpha_t*(step[2] + gamma*Q[index_next].max())
                state = step[3]
                alpha_t = rate_alpha*alpha_t
            
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    Q_max[e][i][j] = Q[i*5 + j].max()
        
    return Q, Q_max

     
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