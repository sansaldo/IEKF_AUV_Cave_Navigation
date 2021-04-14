import numpy as np
from scipy.linalg import expm, block_diag


class Right_IEKF:

    def __init__(self, system):
        # Right_IEKF Construct an instance of this class
        #
        # Input:
        #   system:     system and noise models
        self.A = system['A']  # error dynamics matrix
        self.f = system['f']  # process model
        self.H = system['H']  # measurement error matrix
        # Note that measurement error matrix is a constant for this problem (gravity)
        self.Q = system['Q']  # input noise covariance
        if 'N' in system:
            self.N = system['N']  # <- example, if using only one sensor/unstacked measurements
        else:
            self.N_DVL = system['N_DVL']
            self.N_D = system['N_D']
            self.N_M = system['N_M']
        self.X = system['X'] if 'X' in system else np.eye(5) # state vector
        self.P = system['P'] if 'P' in system else 0.1 * np.eye(9)  # state covariance

    def Ad(self, X):
        # Adjoint of SO3 Adjoint (R) = R
        R = X[:3,:3]
        v = X[:3,3]
        p = X[:3,4]

        v_wedge = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
        p_wedge = np.array([[0, -p[2], p[1]],
                            [p[2], 0, -p[0]],
                            [-p[1], p[0], 0]])

        adj = np.zeros((9,9))
        adj[:3,:3] = R
        adj[3:6,3:6] = R
        adj[6:,6:] = R
        adj[3:6,:3] = np.matmul(v_wedge,R)
        adj[6:,:3] = np.matmul(p_wedge,R)
        return adj


    def skew(self,x):
        # This is useful for the rotation matrix
        """
        :param x: Vector
        :return: R in skew form (NOT R_transpose)
        """
        # vector to skew R^9 -> se(3)+vel
        # x = [ tx, ty, tz, vx, vy, vz, x, y, z]
        
        matrix = np.array([[0, -x[2], x[1], x[3], x[6]],
                           [x[2], 0, -x[0], x[4], x[7]],
                           [-x[1], x[0], 0, x[5], x[8]],
                           [0,0,0,0,0],
                           [0,0,0,0,0]], dtype=float)
        return matrix


    def prediction(self, u, dt, b_g=0):
        # EKF propagation (prediction) step
        """

        :param u: Note that u is actually the gyroscope reading
        :return:
        """
        u[0:3] = u[0:3] - b_g
        u_lie = self.skew(u)
        Phi = expm(self.A*dt)  # see iekf slide 31, though in this case we may not have. Likely this is approximately I for sufficiently small dt. (still needs to be very small - like < 100 Hz) 
        Qd = np.matmul(np.matmul(Phi,self.Q),Phi.T)*dt  # discretized process noise, need to do Phi*Qd*Phi^T if Phi is not the identity

        self.P = np.dot(np.dot(Phi, self.P), Phi.T) + np.dot(np.dot(self.Ad(self.X), Qd), self.Ad(self.X).T)
        self.X = self.f(self.X, u_lie, dt)

    def correction(self, Y, b):
        # RI-EKF correction Step
        N = np.dot(np.dot(self.X, self.N), self.X.T)  # Check how we use diagonals, if we need to, etc.

        # test change in performance when truncating H, nu, and N
        N = N[:3,:3]

        # filter gain
        H = self.H(b)
        S = np.dot(np.dot(H, self.P), H.T) + N
        L = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update state
        nu = np.dot(self.X, Y) - b

        # test change in performance when truncating H, nu, and N
        nu = nu[:3]  #0,1,2 are what we really care about

        delta = self.skew(np.dot(L, nu))  # innovation in the spatial frame
        # skew define here to move to lie algebra 

        self.X = np.dot(expm(delta), self.X)

        # Update Covariance
        I = np.eye(np.shape(self.P)[0])
        temp = I - np.dot(L, H)
        self.P = np.dot(np.dot(temp, self.P), temp.T) + np.dot(np.dot(L, N), L.T)

    def correction_stacked(self, Y, b):

        # RI-EKF correction Step with stacked measurements
        X_stacked = block_diag(self.X, self.X, self.X)
        N_DVL = np.dot(np.dot(self.X, self.N_DVL), self.X.T)  # Check how we use diagonals, if we need to, etc.
        N_D = np.dot(np.dot(self.X, self.N_D), self.X.T)
        N_M = np.dot(np.dot(self.X, self.N_M), self.X.T)
        N_stacked = block_diag(N_DVL[:3,:3],N_D[2,2],N_M[:3,:3])  # just want significant rows from covariances, in block form

        # filter gain
        H = self.H(b)
        S = np.dot(np.dot(H, self.P), H.T) + N_stacked
        L = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update state
        nu = np.dot(X_stacked, Y) - b

        # Take only the rows we care about

        nu = nu[[0,1,2,7,10,11,12]]
        delta = self.skew(np.dot(L, nu))  # innovation in the spatial frame
        # skew define here to move to lie algebra 

        self.X = np.dot(expm(delta), self.X)

        # Update Covariance
        I = np.eye(np.shape(self.P)[0])
        temp = I - np.dot(L, H)
        self.P = np.dot(np.dot(temp, self.P), temp.T) + np.dot(np.dot(L, N_stacked), L.T)




