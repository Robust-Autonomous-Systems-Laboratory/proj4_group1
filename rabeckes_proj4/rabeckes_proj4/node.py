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
        self.poses = {
            'kf' : [],
            'ekf' : [],
            'ukf' : []
            }
        # Init Filters
        # KF
        F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
        H = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
        Q = np.eye(6) * 0.01
        R = np.eye(3) * 0.001
        # EKF
        f = lambda x, u: np.array([
            x[0] + x[3]*self.dt*np.cos(x[2]),
            x[1] + x[3]*self.dt*np.sin(x[2]),
            x[2] + x[4]*self.dt,
            x[3] + x[5]*self.dt,
            u[1],
            (x[3] + u[0])/self.dt
            ])
        F_jacobian = lambda x: np.array([
            [1, 0, -x[3]*np.sin(x[2])*self.dt, np.cos(x[2])*self.dt, 0, 0],
            [0, 1, x[3]*np.cos(x[2])*self.dt, np.sin(x[2])*self.dt,  0, 0],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1/self.dt, 0, 1]])
        H_jacobian = lambda x: np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
        R_ekf = np.eye(3) * 0.001
        f_ukf = lambda x: np.array([
            x[0] + x[3]*self.dt*np.cos(x[2]),
            x[1] + x[3]*self.dt*np.sin(x[2]),
            x[2] + x[4]*self.dt,
            x[3] + x[5]*self.dt,
            x[4],
            x[5]])
        self.filters = {
                'kf' :  KalmanFilter(F, H, Q, R, 6),
                'ekf' : ExtendedKalmanFilter(f, F_jacobian, H_jacobian, Q, R_ekf, 6),
                'ukf' : None #UnscentedKalmanFilter(6, 3, f_ukf, lambda x: np.array([x[3], x[4], x[5]]), Q, R_ekf)
                }
        self.wheel = {
            'phi_l' : 0.,
            'phi_r' : 0.,
            'radius' : 0.033,
            'base' : 0.16,
            }
        self.z = {
            'v' : 0.,
            'omega' : 0.,
            'acc' : 0.
            }
        self.u = {
            'v' : 0.,
            'omega' : 0.
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
        ds_l = (d_phi_l * self.wheel['radius'])/self.dt
        ds_r = (d_phi_r * self.wheel['radius'])/self.dt
        self.z['v'] = (ds_l + ds_r) / 2
        # Update state
        return

    def imuSub(self, msg):
        self.z['acc'] = msg.linear_acceleration.x
        self.z['omega'] = msg.angular_velocity.z
        #self.get_logger().info(f'IMU: theta = {self.z['theta']}')
        return

    def cmdVelSub(self, msg):
        self.u['v'] = msg.twist.linear.x
        self.u['omega'] = msg.twist.angular.z
        return

    def quaternionToYaw(self, x, y, z, w):
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def step(self):
        p0 = np.eye(6) * 0.1
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'odom'
        z = np.array([self.z['v'], self.z['omega'], self.z['acc']])
        u = np.array([self.u['v'], self.u['omega']])
        for f in self.filters:
            path_msg = Path()
            odom_msg = Odometry()
            # Headers
            path_msg.header = header
            odom_msg.header = header
            # Data
            if self.filters[f] == None:
                continue
            x, P = self.filters[f].step(z, u=u, P=p0)
            self.get_logger().info(f'{f} New state vector {x}')
            pose = Pose()
            pose.position.x, pose.position.y = x[0], x[1]
            pose.orientation.w = -np.cos(x[2]/2)
            pose.orientation.z = np.sin(x[2]/2)
            odom_msg.pose.pose = pose
            pose_s = PoseStamped()
            pose_s.pose = pose
            pose_s.header = header
            self.poses[f].append(pose_s)
            path_msg.poses = self.poses[f]
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
