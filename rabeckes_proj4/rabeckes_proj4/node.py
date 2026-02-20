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
from tf2_geometry_msgs import do_transform_pose

from rabeckes_proj4.filters import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter

class KFNode(Node):
    def __init__(self):
        super().__init__('kf_node')
        self.dt = 0.05
        self.poses = []
        # Init Filters
        # KF
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
        # EKF
        f = lambda x: np.array([
            x[0] + x[3]*self.dt,
            x[1] + x[4]*self.dt,
            x[2] + x[5]*self.dt,
            x[3],
            x[4],
            x[5]])
        F_jacobian = lambda x: np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
        H_jacobian = lambda x: np.array([
            [np.cos(x[2]), np.sin(x[2]), 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]])
        R_ekf = np.eye(2) * 0.5
        self.filters = {
                'kf' : None, #KalmanFilter(F, H, Q, R, 6),
                'ekf' : ExtendedKalmanFilter(f, F_jacobian, H_jacobian, Q, R_ekf, 6),
                'ukf' : None
                }
        self.wheel = {
            'phi_l' : 0.,
            'phi_r' : 0.,
            'radius' : 0.033,
            'base' : 0.16,
            }
        self.z = {
            'x' : 0.,
            'y' : 0.,
            'ds' : 0.,
            'theta' : 0.,
            'd_theta' : 0.
            }
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
        # Wheel Angle
        d_phi_l = msg.position[0] - self.wheel['phi_l']
        d_phi_r = msg.position[1] - self.wheel['phi_r']
        self.wheel['phi_l'] = msg.position[0]
        self.wheel['phi_r'] = msg.position[1]
        # Wheel Travel
        ds_l = d_phi_l * self.wheel['radius']
        ds_r = d_phi_r * self.wheel['radius']
        self.z['ds'] = (ds_l + ds_r) / 2
        self.z['d_theta'] = self.wheel['radius'] * (ds_r - ds_l) / self.wheel['base']
        # Update state
        return

    def imuSub(self, msg):
        try:
            tf = self.tf_buffer.lookup_transform('odom', 'imu_link', rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f'Could not transform odom to imu_link: {e}')
            return
        theta = msg.angular_velocity.z * self.dt
        pose = Pose()
        pose.orientation.w = -np.cos(theta/2)
        pose.orientation.z = np.sin(theta/2)
        new_pose = do_transform_pose(pose, tf)
        t_theta = self.quaternionToYaw(new_pose.orientation.x, new_pose.orientation.y, new_pose.orientation.z, new_pose.orientation.w)
        self.z['theta'] = self.z['theta'] + t_theta
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
            if self.filters[f] == None or self.filters[f] == 'kf':
                continue
            z = np.array([self.z['ds'], self.z['d_theta']])
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
