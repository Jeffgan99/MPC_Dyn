import numpy as np
from utils import system
import pdb
import matplotlib.pyplot as plt
from dyn_ftocp import FTOCP


## Parameters initialization
N = 20
n = 4
d = 2
x0 = np.zeros(4)
dt = 0.1
sys = system(x0, dt)
maxTime = 14
xRef = np.array([10, 10, 0, np.pi/2])
uGuess = [np.array([10, 0.1])]*N

Q = 1*np.eye(n)
R = 1*np.eye(d)
Qf = 1000*np.eye(n)

## SQP
# State constraints
Fx = np.vstack((np.eye(n), -np.eye(n)))
bx = np.array([15, 15, 15, 15] * 2)

# Input constraints
Fu = np.vstack((np.eye(d), -np.eye(d)))
bu = np.array([10, 0.5]*2)

# Terminal constraints
Ff = Fx
bf = bx

ftocp = FTOCP(N, n, d, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, dt, uGuess, xRef)
ftocp.solve(x0)


plt.figure()
plt.plot(ftocp.xPred[:,0], ftocp.xPred[:,1], '--ob', label='Solution from one iteration of SQP')
plt.title('Predicted trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,12)
plt.ylim(-1,15)
plt.legend()
plt.show()

uGuess = []
for i in range(0, ftocp.N):
	uGuess.append(ftocp.uPred[i,:])
ftocpSQP = FTOCP(N, n, d, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, dt, uGuess, xRef)
ftocpSQP.solve(x0)

plt.figure()
plt.plot(ftocp.xPred[:,0], ftocp.xPred[:,1], '--ob', label='Solution from one iteration of SQP')
plt.plot(ftocpSQP.xPred[:,0], ftocpSQP.xPred[:,1], '-.dk', label='Solution from two iterations of SQP')
plt.title('Predicted trajectory')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,12)
plt.ylim(-1,15)
plt.legend()

plt.show()


## MPC
sys.reset_IC()
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
	plt.plot(xPred[t][:,0], xPred[t][:,1], '--.b', label='Predicted trajectory using one iteration of SQPat time $t = $'+str(t))
	plt.plot(xPred[t][0,0], xPred[t][0,1], 'ok', label="$x_t$ at time $t = $"+str(t))
	plt.xlabel('$x_1$')
	plt.ylabel('$x_2$')
	plt.xlim(-1,12)
	plt.ylim(-1,15)
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
plt.ylim(-1,15)
plt.legend()



plt.figure()
plt.plot(x_cl[:,0], x_cl[:,1], '-ob', label='Closed-loop trajectory using one iteration of SQP')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(-1,12)
plt.ylim(-1,15)
plt.legend()
plt.show()
