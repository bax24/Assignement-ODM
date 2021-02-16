import numpy as np
import matplotlib.pyplot as plt
import functions as f


def sect5_1():
	# Trajectory length
	T = 10000
	
	# Trajectory of wanted size
	ht = f.trajectory(domain, init, stocha, T)
	
	Q_values = np.zeros((domain.size,4))
	
	f.Q_learning(ht,alpha,gamma,Q_values,0,T)
	
	policy = f.compute_policy_Q_learning(Q_values, domain)
	
	dyn_matrix = np.full((916,domain.shape[0],domain.shape[1]),np.inf)
	
	# J^N_{mu^*} for every state
	J_values = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
 			state = (i, j)
 			
 			J_values[i,j] = f.expected_return(state,domain,gamma,916,stocha,dyn_matrix,policy)
			 
	return policy, J_values


def sect5_2():
	
	# mu* the optimal policy
	opt_policy = f.compute_policy(gamma, 7, stocha, domain)
	
	dyn_matrix = np.full((916,domain.shape[0],domain.shape[1]),np.inf)
	
	# J_{mu*}^N for every state
	J_values = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
 			state = (i, j)
 			
 			J_values[i,j] = f.expected_return(state,domain,gamma,916,stocha,dyn_matrix,opt_policy)
			 
		
	# First protocol
	inf_norm_Q_J_1 = np.zeros((100))
	Q, Q_max = f.epsilon_greddy_Q_learning(100, 1000, init, domain, alpha, gamma, epsilon)
	
	for e in range(100):
		inf_norm_Q_J_1[e] = abs(Q_max[e] - J_values).max()
	
	fig = plt.figure(facecolor='w')
	ax = fig.add_subplot(111,  axisbelow=True)
	ax.plot(range(100), inf_norm_Q_J_1, 'b', alpha=1, linewidth=0.8)
	
	ax.set_xlabel('Episodes')
	ax.set_ylabel('||Q^ - J^N_{mu*}||_inf')
	plt.show()
    
    # Second protocol
	inf_norm_Q_J_2 = np.zeros((100))
	Q, Q_max = f.epsilon_greddy_Q_learning(100, 1000, init, domain, alpha, gamma, epsilon, rate_alpha = 0.8)
	
	for e in range(100):
		inf_norm_Q_J_2[e] = abs(Q_max[e] - J_values).max()
	
	fig = plt.figure(facecolor='w')
	ax = fig.add_subplot(111,  axisbelow=True)
	ax.plot(range(100), inf_norm_Q_J_2, 'b', alpha=1, linewidth=0.8)
	
	ax.set_xlabel('Episodes')
	ax.set_ylabel('||Q^ - J^N_{mu*}||_inf')
	plt.show()

    # Third protocol
	inf_norm_Q_J_3 = np.zeros((100))
	Q, Q_max = f.epsilon_greddy_Q_learning(100, 1000, init, domain, alpha, gamma, epsilon, protocol = 3)    
      		
	fig = plt.figure(facecolor='w')
	ax = fig.add_subplot(111,  axisbelow=True)
	ax.plot(range(100), inf_norm_Q_J_3, 'b', alpha=1, linewidth=0.8)
	
	ax.set_xlabel('Episodes')
	ax.set_ylabel('||Q^ - J^N_{mu*}||_inf')
	plt.show()
 
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
	stocha = False
	
	# Learning rate 
	alpha = 0.05
	
	# Probability to explore
	epsilon = 0.25
	
	# Section 5.1
	if False: 
		policy, J_values = sect5_1()
		
	# Section 5.2
	if True:
		b = sect5_2()