import numpy as np
from scipy.linalg import  expm


class Right_IEKF:

    def __init__(self, system):
        # Right_IEKF Construct an instance of this class
        #
        # Input:
        #   system:     system and noise models
        self.A = system['A']  # error dynamics matrix
        self.f = system['f']  # process model
        self.H = system['H']  # measurement error matrix
        # Note that measurement error matrix is a constant for this problem, cuz gravity duhh
        self.Q = system['Q']  # input noise covariance
        self.N = system['N']  # measurement noise covariance
        self.X = np.eye(3)  # state vector
        self.P = 0.1 * np.eye(3)  # state covariance

    def Ad(self, X):
        # Adjoint of SO3 Adjoint (R) = R
        return X

    def skew(self,x):
        # This is useful for the rotation matrix
        """
        :param x: Vector
        :return: R in skew form (NOT R_transpose)
        """
        # vector to skew R^3 -> so(3)
        matrix = np.array([[0, -x[2], x[1]],
                           [x[2], 0, -x[0]],
                           [-x[1], x[0], 0]], dtype=float)
        return matrix


    def prediction(self, u, dt):
        # EKF propagation (prediction) step
        """

        :param u: Note that u is actually the gyroscope reading
        :return:
        """
        u_lie = self.skew(u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + np.dot(np.dot(self.Ad(self.X), self.Q), self.Ad(self.X).T)
        self.X = self.f(self.X, u_lie, dt)

    def correction(self, Y, b):
        # Note that g is actually the measurement expected in global coordinate frame
        # RI-EKF correction Step
        # No need to stack measurments
        N = np.dot(np.dot(self.X, self.N), self.X.T) # no need to use blkdiag cuz not add zeros
        # filter gain
        H = self.H(b)
        S = np.dot(np.dot(H, self.P), H.T) + N
        L = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update state
        nu = np.dot(self.X, Y).reshape(-1,1) - b
        delta = self.skew(np.dot(L, nu))  # innovation in the spatial frame
        # I used the skew define here to move to lie algebra

        self.X = np.dot(expm(delta), self.X)

        # Update Covariance
        I = np.eye(np.shape(self.P)[0])
        temp = I - np.dot(L, H)
        self.P = np.dot(np.dot(temp, self.P), temp.T) + np.dot(np.dot(L, N), L.T)





