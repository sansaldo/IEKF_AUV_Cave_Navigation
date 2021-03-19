import rosbag
from std_msgs.msg import Int32, String
import numpy as np
from scipy.spatial.transform import Rotation as R
import progressbar

bag = rosbag.Bag('full_dataset.bag')
# Topics we want to use from the bag:
# - both IMUs: /imu_adis(_ros) and /imu_xsens_mti(_ros)
#   * gives us orientation as Euler angles (not global) and gyroscope (for correction), magnetometer
#   * roll/pitch/yaw + mx/my/mz for magnetometer + gx/gy/gz for gyroscope + bx/by/bz for gyroscope bias (?)
# - DVL: /dvl_linkquest
#   * use for prediction step
# - depth sensor: /depth_sensor
# - odometry: /odometry
#   * estimate of pose we can use as ground truth - why not use for estimation?

# NOTE: all covariances given in the ros bag are 0??

print(bag)
topics = ['/imu_adis', '/imu_adis_ros', '/imu_xsens_mti', '/imu_xsens_mti_ros', '/dvl_linkquest', '/depth_sensor', '/odometry']

# Get odometry estimates - vehicle position
odometry = []
print('Getting odometry estimates...')
bar = progressbar.ProgressBar(maxval=len([t for _, _, t in bag.read_messages(topics='/odometry')]))
bar.start()
k = 0
for topic, msg, t in bag.read_messages(topics='/odometry'):
    k += 1
    bar.update(k)    
    # print('----------------------------')
    # print(msg)

    # Convert odometry's position and rotation to an SE(3) matrix
    position = np.expand_dims(np.array([msg.pose.pose.position.z, msg.pose.pose.position.y, msg.pose.pose.position.z]), axis=1)
    rotation = R.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]).as_dcm()
    combined = np.concatenate((np.concatenate((rotation, position), axis=1), np.array([[0., 0., 0., 1.]])), axis=0)
    odometry.append((t, combined))
bar.finish()

# Get depth sensor readings
depth = []
print('Getting depth readings...')
bar = progressbar.ProgressBar(maxval=len([t for _, _, t in bag.read_messages(topics='/depth_sensor')]))
bar.start()
k = 0
for topic, msg, t in bag.read_messages(topics='/depth_sensor'):
    k += 1
    bar.update(k)    
    # print('----------------------------')
    # print(msg)

    depth.append((t, msg.depth))
bar.finish()

# Get DVL readings - vehicle velocities
dvl = []
print('Getting DVL readings...')
bar = progressbar.ProgressBar(maxval=len([t for _, _, t in bag.read_messages(topics='/dvl_linkquest')]))
bar.start()
k = 0
for topic, msg, t in bag.read_messages(topics='/dvl_linkquest'):
    k += 1
    bar.update(k)    
    # print('----------------------------')
    # print(msg)

    # Note that we can also get the beam angles and readings per beam from msg.altitudeBeam and msg.bottomVelocityBeam
    dvl.append((t, np.array(msg.velocityInst)))
bar.finish()

# Get outer IMU readings - vehicle accelerations and estimated pose
imu_euler1 = []
imu_gyro1 = []
imu_magneto1 = []
imu_acc1 = []
print('Getting IMU1 readings (ADIS sensor on bottom of AUV)...')
bar = progressbar.ProgressBar(maxval=len([t for _, _, t in bag.read_messages(topics='/imu_adis')]))
bar.start()
k = 0
for topic, msg, t in bag.read_messages(topics='/imu_adis'):
    k += 1
    bar.update(k)
    print('----------------------------')
    print(msg)
    imu_euler1.append((t, R.from_euler('xyz', [msg.roll, msg.pitch, msg.yaw], degrees=False).as_dcm()))
    imu_gyro1.append((t, (msg.gx, msg.gy, msg.gz)))
    imu_magneto1.append((t, (msg.mx, msg.my, msg.mz)))
    imu_acc1.append((t, np.array([msg.ax, msg.ay, msg.az])))
bar.finish()

# Get inner IMU readings - vehicle acceleration and estimated pose
imu_euler2 = []
imu_gyro2 = []
imu_magneto2 = []
imu_acc2 = []
print('Getting IMU2 readings (Xsens sensor inside AUV)...')
bar = progressbar.ProgressBar(maxval=len([t for _, _, t in bag.read_messages(topics='/imu_xsens_mti')]))
bar.start()
k = 0
for topic, msg, t in bag.read_messages(topics='/imu_xsens_mti'):
    k += 1
    bar.update(k)    
    # print('----------------------------')
    # print(msg)
    imu_euler2.append((t, R.from_euler('xyz', [msg.roll, msg.pitch, msg.yaw], degrees=False).as_dcm()))
    imu_gyro2.append((t, (msg.gx, msg.gy, msg.gz)))
    imu_magneto2.append((t, (msg.mx, msg.my, msg.mz)))
    imu_acc2.append((t, np.array([msg.ax, msg.ay, msg.az])))
bar.finish()




