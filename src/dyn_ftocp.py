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
from casadi import *


def HackyFunction(name, ins, outs):
    # Obtain all symbolic primitives present in the inputs
    Vs = symvar(veccat(*ins))

    # Construct a helper function with these symbols as inputs
    h = Function("helper", Vs, ins)

    # Assert dense inputs
    for i in ins:
        assert i.is_dense(), "input must be dense"

    # Obtain sparsity pattern of the helper function Jacobian
    # It should be a permutation of the unit matrix
    H = blockcat([[Sparsity(h.sparsity_jac(i, j)) for i in range(h.n_in())] for j in range(h.n_out())])

    assert H.size1() == H.size2(), "input must be one-to-one"
    assert H.colind() == range(H.size1() + 1), "input must be one-to-one"
    assert H.T.colind() == range(H.size1() + 1), "input must be one-to-one"

    # Missing assertion: check that only transpose, reshape, vertcat, slice, .. are used

    # Construct new MX symbols on which the original inputs will depend
    Vns = [MX.sym("V", i.size()) for i in ins]

    res = reverse(ins, Vs, [Vns], {"always_inline": True, "never_inline": False})

    # Substitute the original inputs
    return Function("f", Vns, substitute(outs, Vs, res[0]))


class FTOCP(object):
    """ Finite Time Optimal Control Problem (FTOCP)
    Methods:
        - solve: solves the FTOCP given the initial condition x0 and terminal contraints
        - buildNonlinearProgram: builds the ftocp program solved by the above solve method
        - model: given x_t and u_t computes x_{t+1} = f( x_t, u_t )

    """

    def __init__(self, N, Q, R, Qf, Fx, bx, Fu, bu, Ff, bf, dt, uGuess, goal, printLevel):
        # Define variables
        self.printLevel = printLevel

        self.N = N
        self.n = Q.shape[1]
        self.d = R.shape[1]
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
            # self.xGuess.append(np.array(self.dynamics(xt, ut)))

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state
        """
        startTimer = datetime.datetime.now()
        self.simForward(x0, self.uGuess)
        self.buildEqConstr()
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        self.linearizationTime = deltaTimer

        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H, self.q, self.G_in, np.add(self.w_in, np.dot(self.E_in, x0)), self.G_eq,
                           np.add(np.dot(self.E_eq, x0), self.C_eq))
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        # Unpack Solution
        self.unpackSolution(x0)
        self.time += 1

        return self.uPred[0, :]

    def uGuessUpdate(self):
        uPred = self.uPred
        for i in range(0, self.N - 1):
            self.uGuess[i] = [0, 0]  # Method 1
            # self.uGuess[i] = uPred[0,:]  # Method 2
        self.uGuess[-1] = [0, 0]  # Method 1
        # self.uGuess[-1] = uPred[self.N-1]  # Method 2

    def unpackSolution(self, x0):
        # Extract predicted state and predicted input trajectories
        self.xPred = np.vstack((x0, np.reshape((self.Solution[np.arange(self.n * (self.N))]), (self.N, self.n))))
        self.uPred = np.reshape((self.Solution[self.n * (self.N) + np.arange(self.d * self.N)]), (self.N, self.d))

        if self.printLevel >= 2:
            print("Predicted State Trajectory: ")
            print(self.xPred)

            print("Predicted Input Trajectory: ")
            print(self.uPred)

        if self.printLevel >= 1:
            print("Linearization + buildEqConstr() Time: ", self.linearizationTime.total_seconds(), " seconds.")
            print("Solver Time: ", self.solverTime.total_seconds(), " seconds.")

    def buildIneqConstr(self):
        # The inequality constraint is Gin z<= win + Ein x0
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

        if self.printLevel >= 2:
            print("G_in: ")
            print(G_in)
            print("E_in: ")
            print(E_in)
            print("w_in: ", w_in)

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
        # Hint: First construct a vector z_{goal} using the goal state and then leverage the matrix H
        z = np.dot(np.append(np.tile(goal, self.N), np.zeros(self.R.shape[0] * self.N)), linalg.block_diag(barQ, barR))
        q = -2 * z

        if self.printLevel >= 2:
            print("H: ")
            print(H)
            print("q: ", q)

        self.q = q
        self.H = sparse.csc_matrix(
            2 * H)  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def buildEqConstr(self):
        # Hint 1: The equality constraint is: [Gx, Gu]*z = E * x(t) + C
        # Hint 2: Write on paper the matrices Gx and Gu to see how these matrices are constructed
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

        if self.printLevel >= 2:
            print("G_eq: ")
            print(G_eq)
            print("E_eq: ")
            print(E_eq)
            print("C_eq: ", C_eq)

        self.C_eq = C_eq
        self.G_eq = sparse.csc_matrix(G_eq)
        self.E_eq = E_eq

    def buildAutomaticDifferentiationTree(self):
        # Define variables
        n = self.n
        d = self.d
        X = SX.sym('X', n)
        U = SX.sym('U', d)


        X_next = self.dyModel(X, U)

        # X_next = self.dynamics(X, U)
        self.constraint = []
        for i in range(0, n):
            self.constraint = vertcat(self.constraint, X_next[i])

        # X_next = SX.sym('X_next', 1)
        # z = SX.sym('z', n - 1)
        # self.constraint = [X_next,z]

        # self.A_Eval = Function('A', [X, U], [jacobian(constraint, X)])
        # self.B_Eval = Function('B', [X, U], [jacobian(constraint, U)])
        # self.f_Eval = Function('f', [X, U], [constraint])

        # self.constraint = SX.sym('self.constraint', self.constraint)

        self.A_Eval = Function('A', [X, U], [jacobian(self.constraint, X)])
        self.B_Eval = Function('B', [X, U], [jacobian(self.constraint, U)])
        self.f_Eval = Function('f', [X, U], [self.constraint])

    def buildLinearizedMatrices(self, x, u):
        # Give a linearization point (x, u) this function return an affine approximation of the nonlinear system dynamics
        A_linearized = np.array(self.A_Eval(x, u))
        B_linearized = np.array(self.B_Eval(x, u))
        C_linearized = np.squeeze(np.array(self.f_Eval(x, u))) - np.dot(A_linearized, x) - np.dot(B_linearized, u)

        if self.printLevel >= 3:
            print("Linearization x: ", x)
            print("Linearization u: ", u)
            print("Linearized A")
            print(A_linearized)
            print("Linearized B")
            print(B_linearized)
            print("Linearized C")
            print(C_linearized)

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

    # def dynamics(self, x, u):
    #     # state x = [x,y, vx, vy]
    #     x_next = x[0] + self.dt * cos(x[3]) * x[2]
    #     y_next = x[1] + self.dt * sin(x[3]) * x[2]
    #     v_next = x[2] + self.dt * u[0]
    #     theta_next = x[3] + self.dt * u[1]
    #
    #     state_next = [x_next, y_next, v_next, theta_next]
    #
    #     return state_next

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
        # 1st Version that has been commented.
        # deltaT = 0.001
        # x_next = np.zeros(x.shape[0])
        #
        # i = 0
        # while ((i + 1) * deltaT <= self.dt):
        #     alpha_f = u[0] - np.arctan2(x[1] + lf * x[2], x[0])
        #     alpha_r = - np.arctan2(x[1] - lf * x[2], x[0])
        #
        #     # Compute lateral force at front and rear tire
        #     Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
        #     Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))
        #
        #     # Propagate the dynamics of deltaT
        #     x_next[0] = x[0] + deltaT * (u[1] - 1 / m * Fyf * np.sin(u[0]) + x[2] * x[1])
        #     x_next[1] = x[1] + deltaT * (1 / m * (Fyf * np.cos(u[0]) + Fyr) - x[2] * x[0])
        #     x_next[2] = x[2] + deltaT * (1 / Iz * (lf * Fyf * np.cos(u[0]) - lr * Fyr))
        #     x_next[3] = x[3] + deltaT * (x[2])
        #     x_next[4] = x[4] + deltaT * (x[0] * np.cos(x[3]) - x[1] * np.sin(x[3]))
        #     x_next[5] = x[5] + deltaT * (x[0] * np.sin(x[3]) + x[1] * np.cos(x[3]))
        #
        #     x[0] = x_next[0]
        #     x[1] = x_next[1]
        #     x[2] = x_next[2]
        #     x[3] = x_next[3]
        #     x[4] = x_next[4]
        #     x[5] = x_next[5]
        #
        #     # Increment counter
        #     i = i + 1

        # i = 0
        # while ((i + 1) * deltaT <= self.dt):
        #     alpha_f = u[0] - np.arctan2(x[1] + lf * x[2], x[0])
        #     alpha_r = - np.arctan2(x[1] - lf * x[2], x[0])
        #
        #     Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
        #     Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))
        #
        #     vx_next = x[0] + deltaT * (u[1] - 1 / m * Fyf * np.sin(u[0]) + x[2] * x[1])
        #     vy_next = x[1] + deltaT * (1 / m * (Fyf * np.cos(u[0]) + Fyr) - x[2] * x[0])
        #     phi_next = x[2] + deltaT * (1 / Iz * (lf * Fyf * np.cos(u[0]) - lr * Fyr))
        #     ephi_next = x[3] + deltaT * (x[2])
        #     ey_next = x[4] + deltaT * (x[0] * np.cos(x[3]) - x[1] * np.sin(x[3]))
        #     s_next = x[5] + deltaT * (x[0] * np.sin(x[3]) + x[1] * np.cos(x[3]))
        #
        #     x[0] = vx_next
        #     x[1] = vy_next
        #     x[2] = phi_next
        #     x[3] = ephi_next
        #     x[4] = ey_next
        #     x[5] = s_next
        #
        #     i = i + 1

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
        #
        # x_next = vx_next * np.cos(u[0]) - vy_next * np.sin(u[0])
        # y_next = vx_next * np.sin(u[0]) + vy_next * np.cos(u[0])
        # the_next = phi_next

        # x_next = vx_next * np.cos(u[0]) - vy_next * np.sin(u[0])
        # y_next = vx_next * np.sin(u[0]) + vy_next * np.cos(u[0])
        # the_next = phi_next
        #
        # state_next = [x_next, y_next, the_next]


        state_next = [vx_next, vy_next, phi_next, ephi_next, ey_next, s_next]

        return state_next
