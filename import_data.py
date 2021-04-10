####### IMPORT LIBRARIES ####### 

from dataclasses import dataclass
import numpy as np
import os
import pandas as pd 


####### IMPORT DATA FILES ####### 

cwd = os.getcwd()

depth = np.array(pd.read_csv(os.path.join(cwd, 'sensor_data/depth_sensor.csv')))
dvl = np.array(pd.read_csv(os.path.join(cwd, 'sensor_data/dvl_linkquest.csv')))
imu = np.array(pd.read_csv(os.path.join(cwd, 'sensor_data/imu_adis_ros.csv')))
gt = np.array(pd.read_csv(os.path.join(cwd, 'sensor_data/odometry.csv')))
imu_bias = np.array(pd.read_csv(os.path.join(cwd, 'sensor_data/imu_adis.csv')))
#tf = np.array(pd.read_csv(os.path.join(cwd, 'sensor_data/tf.csv')))


####### DECLARE DATA CLASSES ####### 

class odom_data:
    l = len(gt)	
    time = np.zeros((1, len(gt)))
    # 7x1: position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w
    z = np.zeros((7, len(gt)))
	
    def __init__(self, gt):

        self.l = len(gt)
        self.time = np.zeros((1, len(gt)))
        self.z = np.zeros((7, len(gt)))

class dvl_data:
    l = len(dvl)
    time = np.zeros((1, len(dvl)))
    # 3x1 vector: field.velocityEarth0	field.velocityEarth1	field.velocityEarth2
    z = np.zeros((3, len(dvl)))
	
    def __init__(self, dvl):

        self.l = len(dvl)
        self.time = np.zeros((1, len(dvl)))
        self.z = np.zeros((3, len(dvl)))

class depth_data:
    l = len(depth)
    time = np.zeros((1, len(depth)))
    z = np.zeros((1, len(depth)))
	
    def __init__(self, depth):

        self.l = len(depth)
        self.time = np.zeros((1, len(depth)))
        self.z = np.zeros((1, len(depth)))

class imu_data:
    l = len(depth)
    time = np.zeros((1, len(imu)))
    # 9x1 vector: angular velocity x, angular velocity y, angular velocity z, linear accel x, linear accel y, linear accel z, x, y, z)
    z = np.zeros((9, len(imu)))
	
    def __init__(self, imu):

        self.l = len(imu)
        self.time = np.zeros((1, len(imu)))
        self.z = np.zeros((9, len(imu)))

class initial_pose:
    p = np.zeros((3))
    R = np.eye(3)

    def __init__(self, imu):
        self.p = np.zeros((3))
        self.R = np.eye(3)

class imu_bias_data:
    l = len(imu_bias)
    time = np.zeros((1, len(imu_bias)))
    # 3x1 vector: field.bx, field.by, field.bz
    z = np.zeros((3, len(imu_bias)))
	
    def __init__(self, imu_bias):

        self.l = len(imu_bias)
        self.time = np.zeros((1, len(imu_bias)))
        self.z = np.zeros((3, len(imu_bias)))

class mag_data:
    l = len(imu_bias)
    time = np.zeros((1, len(imu_bias)))
    # 3x1 vector: field.mx, field.my, field.mz
    z = np.zeros((3, len(imu_bias)))
	
    def __init__(self, imu_bias):

        self.l = len(imu_bias)
        self.time = np.zeros((1, len(imu_bias)))
        self.z = np.zeros((3, len(imu_bias)))

####### PROCESS DATA ####### 
for i in range(len(imu)):
    # t = rospy.Time.from_sec(imu[1,0])
    # seconds = t.to_sec() #floating point
    # nanoseconds = t.to_nsec()
    imu_data.time[:,i] = [imu[i,0]]
    # 9x1 vector: angular velocity x, angular velocity y, angular velocity z, linear accel x, linear accel y, linear accel z, x, y, z)
    imu_data.z[:,i] = [imu[i,16], imu[i,17], imu[i,18]+.00021,imu[i,28], imu[i,29], imu[i,30], 0, 0, 0]

for i in range(len(depth)):

    depth_data.time[:,i] = [depth[i,0]]
    depth_data.z[:,i] = [depth[i,3]]

for i in range(len(dvl)):
    # World frame
    dvl_data.time[:,i] = [dvl[i,0]]
    # 3x1 vector: field.velocityEarth0	field.velocityEarth1	field.velocityEarth2
    # dvl_data.z[:,i] = [dvl[i,27], dvl[i,28], dvl[i,29]]
    dvl_data.z[:,i] = [dvl[i,28], dvl[i,27], -dvl[i,29]]  # expected based on frames figure from paper

T = np.eye(4)
T[1,1] = -1
T[2,2] = -1
T[0,3] = 0
T[1,3] = 0.64
T[2,3] = 0.09

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.

    Courtesy of: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

for i in range(len(gt)):
    # World frame
    odom_data.time[:,i] = [gt[i,0]]
    # 7x1: position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w
    # odom_data.z[:,i] = [ gt[i,3], gt[i,4], gt[i,5], gt[i,6], gt[i,7], gt[i,8], gt[i,9] ]
    # odom_data.z[:,i] = [ -gt[i,4], -gt[i,3], gt[i,5], gt[i,6], gt[i,7], gt[i,8], gt[i,9] ] # black magic
    # odom_data.z[:,i] = [ gt[i,3], -gt[i,4], -gt[i,5], gt[i,6], gt[i,7], gt[i,8], gt[i,9] ] # expected based on frame figure from paper
    Q = [gt[i,9], gt[i,6], gt[i,7], gt[i,8]]
    R = quaternion_rotation_matrix(Q)
    Hc = np.eye(4)
    Hc[:3,:3] = R
    Hc[0,3] = gt[i,3]
    Hc[1,3] = gt[i,4]
    Hc[2,3] = gt[i,5]
    Hi = np.linalg.solve(T, np.matmul(Hc, T))

    odom_data.z[:,i] = [ Hi[0,3], Hi[1,3], Hi[2,3], 0, 0, 0, 0]  # dummy zeros because I don't want to convert quaternions rn

# initial_pose.p[:] = 0
qx,qy,qz,qw = imu[0,3:7]
initial_pose.R = quaternion_rotation_matrix([qw, qx, qy, qz])
initial_pose.p[2] = -depth_data.z[0,0]

for i in range(len(imu_bias)):

    imu_bias_data.time[:,i] = [imu_bias[i,0]]
    # 3x1 vector: field.bx, field.by, field.bz
    imu_bias_data.z[:,i] = [imu_bias[i,20], imu_bias[i,21], imu_bias[i,22]]

for i in range(len(imu_bias)):

    mag_data.time[:,i] = [imu_bias[i,0]]
    # 3x1 vector: field.mx, field.my, field.mz
    mag_data.z[:,i] = [imu_bias[i,11], imu_bias[i,12], imu_bias[i,13]]

	
####### RUN MAIN ####### 

if __name__ == "__main__":
    toy_example() 
    data_example() 
