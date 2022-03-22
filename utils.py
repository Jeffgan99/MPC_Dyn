import numpy as np
import pdb

# class system(object):
# 	"""docstring for system"""
# 	def __init__(self, x0, dt):
# 		self.x 	   = [x0]
# 		self.u 	   = []
# 		self.w 	   = []
# 		self.x0    = x0
# 		self.dt    = dt
#
# 	def applyInput(self, ut):
# 		self.u.append(ut)
#
# 		xt = self.x[-1]
# 		x_next      = xt[0] + self.dt * np.cos(xt[3]) * xt[2]
# 		y_next      = xt[1] + self.dt * np.sin(xt[3]) * xt[2]
# 		v_next      = xt[2] + self.dt * ut[0]
# 		theta_next  = xt[3] + self.dt * ut[1]
#
# 		state_next = np.array([x_next, y_next, v_next, theta_next])
#
# 		self.x.append(state_next)
#
# 	def reset_IC(self):
# 		self.x = [self.x0]
# 		self.u = []
# 		self.w = []

class system(object):
	"""docstring for system"""

	def __init__(self, x0, dt):
		self.x = [x0]
		self.u = []
		self.w = []
		self.x0 = x0
		self.dt = dt

	def applyInput(self, ut):
		self.u.append(ut)

		xt = self.x[-1]
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
		# deltaT = 0.001
		#
		# x_next = np.zeros(xt.shape[0])
		#
		# i = 0
		# while ((i + 1) * deltaT <= self.dt):
		# 	alpha_f = ut[0] - np.arctan2(xt[1] + lf * xt[2], xt[0])
		# 	alpha_r = - np.arctan2(xt[1] - lf * xt[2], xt[0])
		#
		# 	# Compute lateral force at front and rear tire
		# 	Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
		# 	Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))
		#
		# 	# Propagate the dynamics of deltaT
		# 	x_next[0] = xt[0] + deltaT * (ut[1] - 1 / m * Fyf * np.sin(ut[0]) + xt[2] * xt[1])
		# 	x_next[1] = xt[1] + deltaT * (1 / m * (Fyf * np.cos(ut[0]) + Fyr) - xt[2] * xt[0])
		# 	x_next[2] = xt[2] + deltaT * (1 / Iz * (lf * Fyf * np.cos(ut[0]) - lr * Fyr))
		# 	x_next[3] = xt[3] + deltaT * (xt[2])
		# 	x_next[4] = xt[4] + deltaT * (xt[0] * np.cos(xt[3]) - xt[1] * np.sin(xt[3]))
		# 	x_next[5] = xt[5] + deltaT * (xt[0] * np.sin(xt[3]) + xt[1] * np.cos(xt[3]))
		#
		# 	xt[0] = x_next[0]
		# 	xt[1] = x_next[1]
		# 	xt[2] = x_next[2]
		# 	xt[3] = x_next[3]
		# 	xt[4] = x_next[4]
		# 	xt[5] = x_next[5]
		#
		# 	# Increment counter
		# 	i = i + 1

		# i = 0
		# while ((i + 1) * deltaT <= self.dt):
		# 	alpha_f = ut[0] - np.arctan2(xt[1] + lf * xt[2], xt[0])
		# 	alpha_r = - np.arctan2(xt[1] - lf * xt[2], xt[0])
		#
		# 	Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
		# 	Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))
		#
		# 	vx_next = xt[0] + deltaT * (ut[1] - 1 / m * Fyf * np.sin(ut[0]) + xt[2] * xt[1])
		# 	vy_next = xt[1] + deltaT * (1 / m * (Fyf * np.cos(ut[0]) + Fyr) - xt[2] * xt[0])
		# 	phi_next = xt[2] + deltaT * (1 / Iz * (lf * Fyf * np.cos(ut[0]) - lr * Fyr))
		# 	ephi_next = xt[3] + deltaT * (xt[2])
		# 	ey_next = xt[4] + deltaT * (xt[0] * np.cos(xt[3]) - xt[1] * np.sin(xt[3]))
		# 	s_next = xt[5] + deltaT * (xt[0] * np.sin(xt[3]) + xt[1] * np.cos(xt[3]))
		#
		# 	xt[0] = vx_next
		# 	xt[1] = vy_next
		# 	xt[2] = phi_next
		# 	xt[3] = ephi_next
		# 	xt[4] = ey_next
		# 	xt[5] = s_next
		#
		# 	i = i + 1

		alpha_f = ut[0] - np.arctan2(xt[1] + lf * xt[2], xt[0])
		alpha_r = - np.arctan2(xt[1] - lf * xt[2], xt[0])

		Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
		Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))

		vx_next = xt[0] + self.dt * (ut[1] - 1 / m * Fyf * np.sin(ut[0]) + xt[2] * xt[1])
		vy_next = xt[1] + self.dt * (1 / m * (Fyf * np.cos(ut[0]) + Fyr) - xt[2] * xt[0])
		phi_next = xt[2] + self.dt * (1 / Iz * (lf * Fyf * np.cos(ut[0]) - lr * Fyr))
		ephi_next = xt[3] + self.dt * (xt[2])
		ey_next = xt[4] + self.dt * (xt[0] * np.cos(xt[3]) - xt[1] * np.sin(xt[3]))
		s_next = xt[5] + self.dt * (xt[0] * np.sin(xt[3]) + xt[1] * np.cos(xt[3]))

		# x_next = vx_next * np.cos(ut[0]) - vy_next * np.sin(ut[0])
		# y_next = vx_next * np.sin(ut[0]) + vy_next * np.cos(ut[0])
		# the_next = phi_next
		#
		# state_next = np.array([x_next, y_next, the_next])


		state_next = np.array([vx_next, vy_next, phi_next, ephi_next, ey_next, s_next])
		# state_next = np.array(xt)

		self.x.append(state_next)

	def reset_IC(self):
		self.x = [self.x0]
		self.u = []
		self.w = []
