from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np
from cvxpy import *
import time

class NLP(object):
	""" Non-Linear Program
	"""

	def __init__(self, N, Q, R, Qf, goal, dt, bx, bu, printLevel):
		# Define variables
		self.N    = N
		self.n    = Q.shape[1]
		self.d    = R.shape[1]
		self.bx   = bx
		self.bu   = bu
		self.Q    = Q
		self.Qf   = Qf
		self.R    = R
		self.goal = goal
		self.dt   = dt

		self.bx = bx
		self.bu = bu

		self.printLevel = printLevel

		print("Initializing FTOCP")
		self.buildFTOCP()
		self.solverTime = []
		print("Done initializing FTOCP")

	def solve(self, x0, verbose=False):
		# Set initial condition + state and input box constraints
		self.lbx = x0.tolist() + (-self.bx).tolist()*(self.N) + (-self.bu).tolist()*self.N
		self.ubx = x0.tolist() + ( self.bx).tolist()*(self.N) + ( self.bu).tolist()*self.N
		# Solve nonlinear programm
		start = time.time()
		sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
		end = time.time()
		self.solverTime = end - start

		# Check if the solution is feasible
		if (self.solver.stats()['success']):
			self.feasible = 1
			x = sol["x"]
			self.xPred = np.array(x[0:(self.N+1)*self.n].reshape((self.n,self.N+1))).T
			self.uPred = np.array(x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.d,self.N))).T
			self.mpcInput = self.uPred[0][0]

			if self.printLevel >= 2:
				print("xPredicted:")
				print(self.xPred)
				print("uPredicted:")
				print(self.uPred)

			if self.printLevel >= 1: print("NLP Solver Time: ", self.solverTime, " seconds.")

		else:
			self.xPred = np.zeros((self.N+1, self.n) )
			self.uPred = np.zeros((self.N, self.d))
			self.mpcInput = []
			self.feasible = 0
			print("Unfeasible")
			
		return self.uPred[0]

	def buildFTOCP(self):
		# Define variables
		n  = self.n
		d  = self.d

		# Define variables
		X      = SX.sym('X', n*(self.N+1));
		U      = SX.sym('U', d*self.N);

		# Define dynamic constraints
		self.constraint = []
		for i in range(0, self.N):
			# X_next = self.dynamics(X[n*i:n*(i+1)], U[d*i:d*(i+1)])
			X_next = self.dyModel(X[n * i:n * (i + 1)], U[d * i:d * (i + 1)])
			for j in range(0, self.n):
				self.constraint = vertcat(self.constraint, X_next[j] - X[n*(i+1)+j] ) 

		# Defining Cost (We will add stage cost later)
		self.cost = 0
		for i in range(0, self.N):
			self.cost = self.cost + (X[n*i:n*(i+1)]-self.goal).T @ self.Q @ (X[n*i:n*(i+1)] - self.goal)
			self.cost = self.cost + U[d*i:d*(i+1)].T @ self.R @ U[d*i:d*(i+1)]

		self.cost = self.cost + (X[n*self.N:n*(self.N+1)] - self.goal).T @ self.Qf @ (X[n*self.N:n*(self.N+1)] - self.goal)

		# Set IPOPT options
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#\\, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(X,U), 'f':self.cost, 'g':self.constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force n*N state dynamics
		self.lbg_dyanmics = [0]*(n*self.N)
		self.ubg_dyanmics = [0]*(n*self.N)

	def dyModel(self, x, u):
		m = 1.98
		lf = 0.125
		lr = 0.125
		Iz = 0.024
		Df = 0.8 * m * 9.81 / 2.0
		Cf = 1.25
		Bf = 1.0
		Dr = 0.8 * m * 9.81 / 2.0
		Cr = 1.25
		Br = 1.0

		alpha_f = u[0] - np.arctan2(x[1] + lf * x[2], x[0])
		alpha_r = - np.arctan2(x[1] - lf * x[2], x[0])

		Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
		Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))

		vx_next = x[0] + self.dt * (u[1] - 1 / m * Fyf * np.sin(u[0]) + x[2] * x[1])
		vy_next = x[1] + self.dt * (1 / m * (Fyf * np.cos(u[0]) + Fyr) - x[2] * x[0])
		phi_next = x[2] + self.dt * (1 / Iz * (lf * Fyf * np.cos(u[0]) - lr * Fyr))
		ephi_next = x[3] + self.dt * (x[2])
		ey_next = x[4] + self.dt * (x[0] * np.cos(x[3]) - x[1] * np.sin(x[3]))
		s_next = x[5] + self.dt * (x[0] * np.sin(x[3]) + x[1] * np.cos(x[3]))

		state_next = [vx_next, vy_next, phi_next, ephi_next, ey_next, s_next]
		print("sn: ", state_next)

		return state_next



	# def dynamics(self, x, u):
	# 	# state x = [x,y, vx, vy]
	# 	x_next      = x[0] + self.dt * cos(x[3]) * x[2]
	# 	y_next      = x[1] + self.dt * sin(x[3]) * x[2]
	# 	v_next      = x[2] + self.dt * u[0]
	# 	theta_next  = x[3] + self.dt * u[1]
	#
	# 	state_next = [x_next, y_next, v_next, theta_next]
	#
	# 	return state_next
