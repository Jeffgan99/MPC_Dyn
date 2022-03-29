import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
from dataclasses import dataclass, field
from casadi import sin, cos, SX, vertcat, Function, jacobian
import math


class FTOCP(object):

    def __init__(self, N, n, d, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, dt, uGuess, goal):
        self.N = N
        self.n = n
        self.d = d
        self.Fx = Fx
        self.bx = bx
        self.Fu = Fu
        self.bu = bu
        self.Ff = Ff
        self.bf = bf
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.dt = dt
        self.uGuess = uGuess
        self.goal = goal

        self.buildIneqConstr()
        self.buildAutomaticDifferentiationTree()
        self.buildCost()

        self.time = 0

    def simForward(self, x0, uGuess):
        self.xGuess = [x0]
        for i in range(0, self.N):
            xt = self.xGuess[i]
            ut = self.uGuess[i]
            self.xGuess.append(np.array(self.dyModel(xt, ut)))
            # self.xGuess.append(np.array(NonLinearBicycleModel(xt, ut)))

    def solve(self, x0):

        startTimer = datetime.datetime.now()
        self.simForward(x0, self.uGuess)
        self.buildEqConstr()
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        self.linearizationTime = deltaTimer

        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H, self.q, self.G_in, np.add(self.w_in, np.dot(self.E_in, x0)), self.G_eq, np.add(np.dot(self.E_eq, x0), self.C_eq))
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        self.xPred = np.vstack((x0, np.reshape((self.Solution[np.arange(self.n * (self.N))]), (self.N, self.n))))
        self.uPred = np.reshape((self.Solution[self.n * (self.N) + np.arange(self.d * self.N)]), (self.N, self.d))
        print("Predicted State Trajectory: ")
        print(self.xPred)
        print("Predicted Input Trajectory: ")
        print(self.uPred)
        print("Linearization + buildEqConstr() Time (s): ", self.linearizationTime.total_seconds())
        print("Solver Time (s): ", self.solverTime.total_seconds())

        self.time += 1

        return self.uPred[0, :]

    def uGuessUpdate(self):
        uPred = self.uPred
        for i in range(0, self.N - 1):
            self.uGuess[i] = [0, 0]
        self.uGuess[-1] = [0, 0]


    def buildIneqConstr(self):
        rep_a = [self.Fx] * (self.N - 1)
        Mat = linalg.block_diag(linalg.block_diag(*rep_a), self.Ff)
        Fxtot = np.vstack((np.zeros((self.Fx.shape[0], self.n * self.N)), Mat))
        bxtot = np.append(np.tile(np.squeeze(self.bx), self.N), self.bf)

        rep_b = [self.Fu] * (self.N)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.N)

        G_in = linalg.block_diag(Fxtot, Futot)
        E_in = np.zeros((G_in.shape[0], self.n))
        E_in[0:self.Fx.shape[0], 0:self.n] = -self.Fx
        w_in = np.hstack((bxtot, butot))

        self.G_in = sparse.csc_matrix(G_in)
        self.E_in = E_in
        self.w_in = w_in.T


    def buildCost(self):
        listQ = [self.Q] * (self.N - 1)
        barQ = linalg.block_diag(linalg.block_diag(*listQ), self.Qf)

        listTotR = [self.R] * (self.N)
        barR = linalg.block_diag(*listTotR)

        H = linalg.block_diag(barQ, barR)

        goal = self.goal
        z = np.dot(np.append(np.tile(goal, self.N), np.zeros(self.R.shape[0] * self.N)), linalg.block_diag(barQ, barR))
        q = -2 * z
        # print("q: ", q)

        self.q = q
        self.H = sparse.csc_matrix(2 * H)


    def buildEqConstr(self):
        Gx = np.eye(self.n * (self.N))
        Gu = np.zeros((self.n * (self.N), self.d * self.N))

        self.C = []
        E_eq = np.zeros((Gx.shape[0], self.n))
        for k in range(0, self.N):
            A, B, C = self.buildLinearizedMatrices(self.xGuess[k], self.uGuess[k])

            if k == 0:
                E_eq[0:self.n, :] = A
            else:
                Gx[(self.n + (k - 1) * self.n):(self.n + (k - 1) * self.n + self.n),
                ((k - 1) * self.n):((k - 1) * self.n + self.n)] = -A
                Gu[(self.n + (k - 1) * self.n):(self.n + (k - 1) * self.n + self.n),
                ((k - 1) * self.d):((k - 1) * self.d + self.d)] = -B
            self.C = np.append(self.C, C)

        G_eq = np.hstack((Gx, Gu))
        C_eq = self.C

        self.C_eq = C_eq
        self.G_eq = sparse.csc_matrix(G_eq)
        self.E_eq = E_eq


    def buildAutomaticDifferentiationTree(self):
        X = SX.sym('X', self.n)
        U = SX.sym('U', self.d)

        X_next = self.dyModel(X, U)
        self.constraint = []
        for i in range(0, self.n):
            self.constraint = vertcat(self.constraint, X_next[i])

        self.A_Eval = Function('A', [X, U], [jacobian(self.constraint, X)])
        self.B_Eval = Function('B', [X, U], [jacobian(self.constraint, U)])
        self.f_Eval = Function('f', [X, U], [self.constraint])


    def buildLinearizedMatrices(self, x, u):
        A_linearized = np.array(self.A_Eval(x, u))
        B_linearized = np.array(self.B_Eval(x, u))
        C_linearized = np.squeeze(np.array(self.f_Eval(x, u))) - np.dot(A_linearized, x) - np.dot(B_linearized, u)

        return A_linearized, B_linearized, C_linearized


    def osqp_solve_qp(self, P, q, G=None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """

        qp_A = vstack([G, A]).tocsc()
        l = -inf * ones(len(h))
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])

        self.osqp = OSQP()
        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)

        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0
            print("The FTOCP is not feasible at time t = ", self.time)

        self.Solution = res.x


    def dyModel(self, x, u):

        lf = 0.125 # 0.125
        lr = 0.125 # 0.125
        beta = np.arctan(lr / (lr + lf) * np.tan(u[1]))

        x_next = x[0] + self.dt * x[2] * np.cos(x[3] + beta)
        y_next = x[1] + self.dt * x[2] * np.sin(x[3] + beta)
        v_next = x[2] + self.dt * u[0]
        theta_next = x[3] + self.dt * x[2] / lr * np.sin(beta)

        state_next = [x_next, y_next, v_next, theta_next]

        return state_next


