import numpy as np
from scipy.linalg import  expm

def skew(omega):
    # Assume phi is a 3x1 vector
    return np.array([[        0, -omega[2],  omega[1]],
                     [ omega[2],         0, -omega[0]],
                     [-omega[1],  omega[0],         0]])

def gamma_0(phi):
    return expm(skew(phi))

def gamma_1(phi):
    nphi = np.linalg.norm(phi)
    sphi = skew(phi)
    return np.eye(3) + ((1-np.cos(nphi))/(nphi**2))*sphi + ((nphi-np.sin(nphi))/(nphi**3))*np.matmul(sphi,sphi)

def gamma_2(phi):
    nphi = np.linalg.norm(phi)
    sphi = skew(phi)
    return 0.5*np.eye(3) + ((nphi-np.sin(nphi))/(nphi**3))*sphi + ((nphi**2 + 2*np.cos(nphi) - 2)/(2*(nphi**4)))*np.matmul(sphi,sphi)

def imu_dynamics(state, inputs, dt):
    Rk = state[:3,:3]
    vk = state[:3,3]
    pk = state[:3,4]

    # Assume matrix form for inputs
    omega_k = inputs[:3,:3]
    ak = inputs[:3,3]

    g = np.array([[0], [0], [9.81]])  # may need to make negative

    Rk1 = np.matmul(Rk,gamma_0(omega_k*dt))
    vk1 = vk + np.matmul(np.matmul(Rk,gamma_1(omega_k*dt)),ak)*dt + g*dt
    pk1 = pk + vk*dt + np.matmul(np.matmul(Rk,gamma_2(omega_k*dt)),ak)*(dt**2) + 0.5*g*dt

    new_state = np.eye(5)
    new_state[:3,:3] = Rk1
    new_state[:3,3]  = vk1
    new_state[:3,4]  = pk1

    return new_state

def H_matrix(b): # Possibly nonconstant, but not likely we believe
    H = np.zeros((5,9))
    H[:3,3:6] = np.eye(3)
    return H