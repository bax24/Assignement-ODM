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
	
	# Deterministic: stocha = False; Stochastic: stocha = True
	stocha = True
	
	ht = f.trajectory(domain, init, stocha, 10)