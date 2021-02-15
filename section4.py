import numpy as np
import functions as f
import matplotlib.pyplot as plt


def convergence_plot(stocha):
	# Lengths of the trajectories
	N = [50,100,500,1000,10000,50000,100000,2000000]
	
	conv_r = np.zeros((len(N)))
	conv_p = np.zeros((len(N)))
	
	for i in range(len(N)):
		ht = f.trajectory(domain, init, stocha, N[i]) # trajectory of size N[i]
		
		r_hat_values, p_hat_values = f.compute_r_hat_p_hat_values(ht,domain,stocha)	
		r_values, p_values = f.compute_r_p_values(domain,stocha)
		
		diff_r = abs(r_values - r_hat_values)
		diff_p = abs(p_values - p_hat_values)
		
		conv_r[i] = diff_r.max()
		conv_p[i] = diff_p.max()
		
		
	# ------- ||r - r^||_inf ---------
	fig,(ax,ax2) = plt.subplots(1, 2, sharey=True)
	
	# plot the same data on both axes
	ax.plot(N, conv_r, marker = 'o')
	ax2.plot(N, conv_r, marker = 'o', label = '||r - r^||_inf')
	
	# zoom-in / limit the view to different portions of the data
	ax.set_xlim(0,11000)
	ax2.set_xlim(1000000,2200000)
	
	# hide the spines between ax and ax2
	ax.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	ax.yaxis.tick_left()
	ax2.yaxis.tick_right()
	
	# Make the spacing between the two axes a bit smaller
	plt.subplots_adjust(wspace=0.15)
	legend = ax2.legend()
	legend.get_frame().set_alpha(0.5)
	
	plt.show()
		
	# ------- ||p - p^||_inf ---------
	fig,(ax,ax2) = plt.subplots(1, 2, sharey=True)
	
	# plot the same data on both axes
	ax.plot(N, conv_p, 'r', marker = 'o')
	ax2.plot(N, conv_p, 'r', marker = 'o', label = '||p - p^||_inf')
	
	# zoom-in / limit the view to different portions of the data
	ax.set_xlim(0,11000)
	ax2.set_xlim(1000000,2200000)
	
	# hide the spines between ax and ax2
	ax.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	ax.yaxis.tick_left()
	ax2.yaxis.tick_right()
	
	# Make the spacing between the two axes a bit smaller
	plt.subplots_adjust(wspace=0.15)
	legend = ax2.legend()
	legend.get_frame().set_alpha(0.5)
	
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
	stocha = True
	
	conv_plot = False
	
	if conv_plot:
		convergence_plot(stocha)
	
	# Number of steps
	N = 7
	
	# Trajectory of wanted size
	ht = f.trajectory(domain, init, stocha, 1000)
	
	# || Q - Q^ ||_inf for the given trajectory
	inf_norm_Q = f.inf_norm_Q(N,gamma,stocha,ht,domain)
	
	# mu^* the optimal policy
	opt_policy_hat = f.compute_policy_hat(ht, gamma, N, domain)
	
	dyn_matrix = np.full((916,domain.shape[0],domain.shape[1]),np.inf)
	
	# J^N_{mu^*} for every state
	J_values = np.zeros((5,5))
	for i in range(5):
		for j in range(5):
 			state = (i, j)
 			
 			J_values[i,j] = f.expected_return(state,domain,gamma,916,stocha,dyn_matrix,opt_policy_hat)
			 
	print(J_values)		