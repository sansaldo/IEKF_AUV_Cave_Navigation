import sys
sys.path.append('/home/shane/catkin_ws/devel/lib/python2.7/dist-packages') # Need to import some custom ROS message classes from the caves dataset package
import rosbag
import rospy
from std_msgs.msg import Int32, String
from nav_msgs.msg import Odometry
from cirs_girona_cala_viuda.msg import Depth, LinkquestDvl, Imu
import numpy as np
from scipy.spatial.transform import Rotation as R
import progressbar
from datetime import datetime

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

verbose = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'verbose':
        print('Verbose mode active.')
        verbose = True

def log_message(type_str, t):
    print('[%s] * received %s message *' % (str(t.secs + t.nsecs/1e9), str(type_str)))

odometry = []
def odometry_callback(msg, id):
    # Get some message metadata first
    t = msg.header.stamp
    frame_id = msg.header.frame_id
    if verbose:
        log_message('odometry', t)
    # Get odometry estimates of vehicle position

    # Convert odometry's position and rotation to an SE(3) matrix
    position = np.expand_dims(np.array([msg.pose.pose.position.z, msg.pose.pose.position.y, msg.pose.pose.position.z]), axis=1)
    rotation = R.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]).as_dcm()
    combined = np.concatenate((np.concatenate((rotation, position), axis=1), np.array([[0., 0., 0., 1.]])), axis=0)
    odometry.append((t, combined))

# Get depth sensor readings
depth = []
def depth_callback(msg, id):
    # Get some message metadata first
    t = msg.header.stamp
    frame_id = msg.header.frame_id
    if verbose:
        log_message('depth', t)
    depth.append((t, msg.depth))

# Get DVL readings - vehicle velocities
dvl = []
def dvl_callback(msg, id):
    # Get some message metadata first
    t = msg.header.stamp
    frame_id = msg.header.frame_id
    if verbose:
        log_message('dvl', t)
    # Note that we can also get the beam angles and readings per beam from msg.altitudeBeam and msg.bottomVelocityBeam
    dvl.append((t, np.array(msg.velocityInst)))

# Get outer IMU readings - vehicle accelerations and estimated pose
imu_euler1 = []
imu_gyro1 = []
imu_magneto1 = []
imu_acc1 = []
def external_imu_callback(msg, id):
    # Get some message metadata first
    t = msg.header.stamp
    frame_id = msg.header.frame_id
    if verbose:
        log_message('external imu', t)
    imu_euler1.append((t, R.from_euler('xyz', [msg.roll, msg.pitch, msg.yaw], degrees=False).as_dcm()))
    imu_gyro1.append((t, (msg.gx, msg.gy, msg.gz)))
    imu_magneto1.append((t, (msg.mx, msg.my, msg.mz)))
    imu_acc1.append((t, np.array([msg.ax, msg.ay, msg.az])))

# Get inner IMU readings - vehicle acceleration and estimated pose
imu_euler2 = []
imu_gyro2 = []
imu_magneto2 = []
imu_acc2 = []
def internal_imu_callback(msg, id):
    # Get some message metadata first
    t = msg.header.stamp
    frame_id = msg.header.frame_id
    if verbose:
        log_message('internal imu', t)
    imu_euler2.append((t, R.from_euler('xyz', [msg.roll, msg.pitch, msg.yaw], degrees=False).as_dcm()))
    imu_gyro2.append((t, (msg.gx, msg.gy, msg.gz)))
    imu_magneto2.append((t, (msg.mx, msg.my, msg.mz)))
    imu_acc2.append((t, np.array([msg.ax, msg.ay, msg.az])))

print('Initializing node...')
rospy.init_node('cirs_girona_cala_viuda')

# topics = ['/imu_adis', '/imu_adis_ros', '/imu_xsens_mti', '/imu_xsens_mti_ros', '/dvl_linkquest', '/depth_sensor', '/odometry']

print('Subscribing to the topics we want...')
rospy.Subscriber('/odometry', Odometry, odometry_callback, 0)
rospy.Subscriber('/depth_sensor', Depth, depth_callback, 1)
rospy.Subscriber('/dvl_linkquest', LinkquestDvl, dvl_callback, 2)
rospy.Subscriber('/imu_adis', Imu, external_imu_callback, 3)
rospy.Subscriber('/imu_xsens_mti', Imu, internal_imu_callback, 4)

print('Spinning!')
rospy.spin()