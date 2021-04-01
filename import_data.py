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

class imu_bias_data:
    l = len(imu_bias)
    time = np.zeros((1, len(imu_bias)))
    # 3x1 vector: field.bx, field.by, field.bz
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
    imu_data.z[:,i] = [imu[i,16], imu[i,17], imu[i,18],imu[i,28], imu[i,29], imu[i,30], 0, 0, 0]

for i in range(len(depth)):

    depth_data.time[:,i] = [depth[i,0]]
    depth_data.z[:,i] = [depth[i,3]]

for i in range(len(dvl)):
    # World frame
    dvl_data.time[:,i] = [dvl[i,0]]
    # 3x1 vector: field.velocityEarth0	field.velocityEarth1	field.velocityEarth2
    dvl_data.z[:,i] = [dvl[i,27], dvl[i,28], dvl[i,29]]

for i in range(len(gt)):
    # World frame
    odom_data.time[:,i] = [gt[i,0]]
    # 7x1: position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w
    odom_data.z[:,i] = [ gt[i,3], gt[i,4], gt[i,5], gt[i,6], gt[i,7], gt[i,8], gt[i,9] ]

for i in range(len(imu_bias)):

    imu_bias_data.time[:,i] = [imu_bias[i,0]]
    # 3x1 vector: field.bx, field.by, field.bz
    imu_bias_data.z[:,i] = [imu_bias[i,20], imu_bias[i,21], imu_bias[i,22]]

