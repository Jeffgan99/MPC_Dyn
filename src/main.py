import numpy as np
from utils import system
import pdb
import matplotlib.pyplot as plt
# from ftocp import FTOCP
from dyn_ftocp import FTOCP
from nlp import NLP
from matplotlib import rc
from numpy import linalg as la
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)

# =============================
# Initialize system parameters
# x0 = np.zeros(4)
# x0 = np.zeros(6)
x0 = np.array([0.5, 0.1, 0.1, 0.1, 0, 0])
# x0 = np.zeros(6)
dt = 0.1  # Discretization time
sys        = system(x0, dt) # initialize system object
maxTime    = 14 # Simulation time
goal = np.array([0, 0, 0, np.pi/2, 10, 10])
# goal = np.array([10,10,0,np.pi/2])

# Initialize mpc parameters
N  = 20
n = 6 # 4 6
d = 2
Q  = 1*np.eye(n)
R  = 1*np.eye(d)
# Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0])
# R = np.diag([1.0, 10.0])
Qf = 1000*np.eye(n)

# =================================================================
# ======================== Subsection: Nonlinear MPC ==============
# # First solve the nonlinear optimal control problem as a Non-Linear Program (NLP)
printLevel = 1
# # # xub = np.array([15, 15, 15, 15])
# xub = np.array([15, 15, 15, 15, 15, 15])
# uub = np.array([10, 0.5])
# nlp = NLP(N, Q, R, Qf, goal, dt, xub, uub, printLevel)
# ut  = nlp.solve(x0)
#
# sys.reset_IC() # Reset initial conditions
# xPredNLP = []
# for t in range(0,maxTime): # Time loop
# 	xt = sys.x[-1]
# 	ut = nlp.solve(xt)
# 	xPredNLP.append(nlp.xPred)
# 	sys.applyInput(ut)
#
# x_cl_nlp = np.array(sys.x)
#
# for timeToPlot in [0, 10]:
# 	plt.figure()
# 	plt.plot(xPredNLP[timeToPlot][:,0], xPredNLP[timeToPlot][:,1], '--.b', label="Predicted trajectory at time $t = $"+str(timeToPlot))
# 	plt.plot(xPredNLP[timeToPlot][0,0], xPredNLP[timeToPlot][0,1], 'ok', label="$x_t$ at time $t = $"+str(timeToPlot))
# 	plt.xlabel('$x$')
# 	plt.ylabel('$y$')
# 	plt.xlim(-1,12)
# 	plt.ylim(-1,10)
# 	plt.legend()
#
# plt.figure()
# for t in range(0, maxTime):
# 	if t == 0:
# 		plt.plot(xPredNLP[t][:,0], xPredNLP[t][:,1], '--.b', label='Predicted trajectory at time $t$')
# 	else:
# 		plt.plot(xPredNLP[t][:,0], xPredNLP[t][:,1], '--.b')
# plt.plot(x_cl_nlp[:,0], x_cl_nlp[:,1], '-*r', label="Closed-loop trajectory")
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.xlim(-1,12)
# plt.ylim(-1,10)
# plt.legend()
# plt.show()

# # =================================================================
# # =========== Subsection: Sequential Quadratic Programming ========
# # State constraint set X = \{ x : F_x x \leq b_x \}
# Fx = np.vstack((np.eye(n), -np.eye(n)))
# bx = np.array([15,15,15,15]*(2))
Fx = np.array([[0., 0., 0., 0., 0., 1.],
	       [0., 0., 0., 0., 0., -1.]])
#
bx = np.array([[2.],
	       [2.]])
#
# Fu = np.kron(np.eye(2), np.array([1, -1])).T
# bu = np.array([[0.5],
# 			   [0.5],
# 			   [10.0],
# 			   [10.0]])

# # Input constraint set U = \{ u : F_u u \leq b_u \}
Fu = np.vstack((np.eye(d), -np.eye(d)))
bu = np.array([10, 0.5]*2)

# # Terminal constraint set
Ff = Fx
bf = bx

# printLevel = 1
uGuess = [np.array([10, 0.1])]*N

ftocp = FTOCP(N, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, dt, uGuess, goal, printLevel)
ftocp.solve(x0)

# print(ftocp.xPred)

plt.figure()
# plt.plot(xPredNLP[0][:,0], xPredNLP[0][:,1], '-*r', label='Solution from the NLP')
plt.plot(ftocp.xPred[:,0], ftocp.xPred[:,1], '--ob', label='Solution from one iteration of SQP')
plt.title('Predicted trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,12)
plt.ylim(-1,10)
plt.legend()
plt.show()
#
uGuess = []
for i in range(0, ftocp.N):
	uGuess.append(ftocp.uPred[i,:]) # Initialize input used for linearization using the optimal input from the first SQP iteration
ftocpSQP = FTOCP(N, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, dt, uGuess, goal, printLevel)
ftocpSQP.solve(x0)

plt.figure()
# plt.plot(xPredNLP[0][:,0], xPredNLP[0][:,1], '-*r', label='Solution from the NLP')
plt.plot(ftocp.xPred[:,0], ftocp.xPred[:,1], '--ob', label='Solution from one iteration of SQP')
plt.plot(ftocpSQP.xPred[:,0], ftocpSQP.xPred[:,1], '-.dk', label='Solution from two iterations of SQP')
plt.title('Predicted trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,12)
plt.ylim(-1,10)
plt.legend()

plt.show()
#
# # # =================================================================
# # =========== Subsection: NMPC using an SQP Approach  =============
sys.reset_IC() # Reset initial conditions
xPred = []
for t in range(0,maxTime):
	xt = sys.x[-1]
	ut = ftocpSQP.solve(xt)
	ftocpSQP.uGuessUpdate()
	xPred.append(ftocpSQP.xPred)
	sys.applyInput(ut)

x_cl = np.array(sys.x)
u_cl = np.array(sys.u)

for t in range(0, 6):
	plt.figure()
	# plt.plot(xPredNLP[t][:,0], xPredNLP[t][:,1], '-*r', label='Predicted trajectory using NLP at time $t = $'+str(t))
	plt.plot(xPred[t][:,0], xPred[t][:,1], '--.b', label='Predicted trajectory using one iteration of SQPat time $t = $'+str(t))
	plt.plot(xPred[t][0,0], xPred[t][0,1], 'ok', label="$x_t$ at time $t = $"+str(t))
	plt.xlabel('$x_1$')
	plt.ylabel('$x_2$')
	plt.xlim(-1,12)
	plt.ylim(-1,10)
	plt.legend()


plt.figure()
for t in range(0, maxTime):
	if t == 0:
		plt.plot(xPred[t][:,0], xPred[t][:,1], '--.b', label='Predicted trajectory using one iteration of SQP')
	else:
		plt.plot(xPred[t][:,0], xPred[t][:,1], '--.b')

plt.plot(x_cl[:,0], x_cl[:,1], '-*r', label='Closed-loop trajectory using one iteration of SQP')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,12)
plt.ylim(-1,10)
plt.legend()



plt.figure()
# plt.plot(x_cl_nlp[:,0], x_cl_nlp[:,1], '-*r', label='Closed-loop trajectory using NLP')
plt.plot(x_cl[:,0], x_cl[:,1], '-ob', label='Closed-loop trajectory using one iteration of SQP')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,12)
plt.ylim(-1,10)
plt.legend()
plt.show()
