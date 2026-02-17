from warnings import filters
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from rabeckes_proj4.kf import KalmanFilter

class KFNode(Node):
    def __init__(self):
        super().__init__('kf_node')
        self.dt = 0.05
        self.poses = []
        # Init Filters
        F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]])
        Q = np.eye(6) * 0.0001
        R = np.eye(3) * 0.001
        self.filters = {
                'kf' : KalmanFilter(F, H, Q, R, 6),
                'ekf' : None, 
                'ukf' : None
                }
        self.z = {
            'x' : 0.,
            'y' : 0.,
            'theta' : 0.}
        # Create Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.jointStateSub,
            20)
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imuSub,
            20)
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped,
            '/cmd_vel',
            self.cmdVelSub,
            10)
        # Create Publishers (could be better as a dict)
        self.pubs = {}
        for f in self.filters:
            self.pubs[f] = {
                'path' : self.create_publisher(
                    Path,
                    f'/localization_node/{f}/path',
                    20),
                'odom' : self.create_publisher(
                    Odometry,
                    f'/localization_node/{f}/odometry',
                    20)
            }
        # Timer callback
        self.timer = self.create_timer(self.dt, self.step)
        # Transform Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
    def jointStateSub(self, msg):
        try:
            tf = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f'Could not transform odom to base_link: {e}')
            return
        '''
        qx = tf.transform.rotation.x
        qy = tf.transform.rotation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w
        theta = self.quaternionToYaw(qx, qy, qz, qw)
        mx = msg.position[0]
        my = msg.position[1]
        self.z['x'] = np.cos(theta)*mx - np.sin(theta)*my + tf.transform.translation.x
        self.z['y'] = np.sin(theta)*mx + np.cos(theta)*my + tf.transform.translation.y
        '''
        self.z['x'] = tf.transform.translation.x
        self.z['y'] = tf.transform.translation.y
        #self.get_logger().info(f'{tf}')
        return

    def imuSub(self, msg):
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w
        yaw = self.quaternionToYaw(x, y, z, w)
        self.z['theta'] = yaw
        #self.get_logger().info(f'IMU: theta = {self.z['theta']}')
        return

    def cmdVelSub(self, msg):
        return

    def quaternionToYaw(self, x, y, z, w):
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def step(self):
        p0 = np.eye(6) * 0.1
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'odom'
        for f in self.filters:
            path_msg = Path()
            odom_msg = Odometry()
            # Headers
            path_msg.header = header
            odom_msg.header = header
            # Data
            if self.filters[f] == None:
                break
            z = np.array([self.z['x'], self.z['y'], self.z['theta']])
            x, P = self.filters[f].step(z, P=p0)
            self.get_logger().info(f'New state vector {x}')
            pose = Pose()
            pose.position.x, pose.position.y = x[0], x[1]
            pose.orientation.w = -np.cos(x[2]/2)
            pose.orientation.z = np.sin(x[2]/2)
            odom_msg.pose.pose = pose
            pose_s = PoseStamped()
            pose_s.pose = pose
            pose_s.header = header
            self.poses.append(pose_s)
            path_msg.poses = self.poses
            c = np.zeros([6,6])
            c[0:3, 0:3] = P[0:3, 0:3]
            odom_msg.pose.covariance = c.flatten()
            # Publish
            self.pubs[f]['path'].publish(path_msg)
            self.pubs[f]['odom'].publish(odom_msg)
        return

def main():
    rclpy.init()
    node = KFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
