import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path, Odometry

class KFNode(Node):
    def __init__(self):
        super().__init__('kf_node')
        # Inits
        self.filters = ['kf', 'ekf', 'ukf']
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
        self.timer = self.create_timer(0.05, self.step)
        self.i = 0
        # Init vars
        self.z = {
            'x' : 0,
            'y' : 0,
            'theta' : 0}

    def jointStateSub(self, msg):
        self.z['x'] = msg.position[0]
        self.z['y'] = msg.position[1]
        self.get_logger().info(f'Joint State: x = {self.z['x']}, y = {self.z['y']}')
        return

    def imuSub(self, msg):
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self.z['theta'] = yaw
        self.get_logger().info(f'IMU: theta = {self.z['theta']}')
        return

    def cmdVelSub(self, msg):
        return

    def step(self):
        for f in self.filters:
            path_msg = Path()
            odom_msg = Odometry()
            # Headers
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'odom'
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = 'odom'
            # Data

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
