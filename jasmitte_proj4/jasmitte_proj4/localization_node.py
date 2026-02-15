import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import numpy as np

class LocalizationNode(Node):
    """
    ROS 2 Node that implements three different state estimation filters:
    1. Linearized Kalman Filter (KF)
    2. Extended Kalman Filter (EKF)
    3. Unscented Kalman Filter (UKF)
    
    References:
    - Differential Kinematics: https://en.wikipedia.org/wiki/Differential_wheeled_robot
    - Kalman Filtering: https://en.wikipedia.org/wiki/Kalman_filter
    - Burgerbot Parameters: https://emanual.robotis.com/docs/en/platform/turtlebot3/features/
    """
    def __init__(self):
        super().__init__('localization_node')
        
        # ___Burgerbot Constants___
        self.R = 0.033
        self.L = 0.160
        
        # ___Filter States (x, y, theta)___
        self.x_kf = np.zeros(3)
        self.P_kf = np.eye(3) * 0.1 
        self.x_last_kf_wheels = np.copy(self.x_kf)
        self.x_last_kf_imu = np.copy(self.x_kf)
        
        self.x_ekf = np.zeros(3)
        self.P_ekf = np.eye(3) * 0.1 
        self.x_last_ekf_wheels = np.copy(self.x_ekf)
        self.x_last_ekf_imu = np.copy(self.x_ekf)
        
        self.x_ukf = np.zeros(3)
        self.P_ukf = np.eye(3) * 0.1 
        self.x_last_ukf_wheels = np.copy(self.x_ukf)
        self.x_last_ukf_imu = np.copy(self.x_ukf)

        # ___Noise Matrices___
        self.Q = np.diag([0.0001, 0.0001, 0.0001]) # Process noise for [x, y, theta]
        self.R_wheels = np.diag([0.001, 0.001]) # For [ds, dtheta]
        self.R_imu = np.array([[0.04]])         # For [dtheta]
        
        # ___UKF Parameters___
        self.n = 3
        self.alpha = 0.1 
        self.lambda_ukf = self.alpha**2 * (self.n + 0.0) - self.n
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lambda_ukf)))
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lambda_ukf)))
        self.Wc[0] = self.lambda_ukf / (self.n + self.lambda_ukf) + (1 - self.alpha**2 + 2.0)
        self.Wm[0] = self.lambda_ukf / (self.n + self.lambda_ukf)
        
        self.last_phi_l, self.last_phi_r = None, None
        self.v_cmd, self.omega_cmd = 0.0, 0.0
        self.last_time = None
        self.last_imu_time = None
        
        path_qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        # ___Publishers___
        self.kf_odom_pub = self.create_publisher(Odometry, '/localization_node/kf/odometry', 10)
        self.kf_path_pub = self.create_publisher(Path, '/localization_node/kf/path', path_qos)
        self.ekf_odom_pub = self.create_publisher(Odometry, '/localization_node/ekf/odometry', 10)
        self.ekf_path_pub = self.create_publisher(Path, '/localization_node/ekf/path', path_qos)
        self.ukf_odom_pub = self.create_publisher(Odometry, '/localization_node/ukf/odometry', 10)
        self.ukf_path_pub = self.create_publisher(Path, '/localization_node/ukf/path', path_qos)

        self.kf_analysis_pub = self.create_publisher(Float64MultiArray, '/localization_node/kf/analysis', 10)
        self.ekf_analysis_pub = self.create_publisher(Float64MultiArray, '/localization_node/ekf/analysis', 10)
        self.ukf_analysis_pub = self.create_publisher(Float64MultiArray, '/localization_node/ukf/analysis', 10)

        self.kf_path, self.ekf_path, self.ukf_path = Path(), Path(), Path()
        for p in [self.kf_path, self.ekf_path, self.ukf_path]: p.header.frame_id = 'odom'

        # ___Subscriptions___
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.cmd_vel_sub = self.create_subscription(TwistStamped, '/cmd_vel', self.cmd_vel_callback, 10)

        self.get_logger().info('Localization Node Started Strongly.')

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def f(self, x, dt, v, omega):
        x_new = np.copy(x)
        x_new[0] += v * math.cos(x[2]) * dt
        x_new[1] += v * math.sin(x[2]) * dt
        x_new[2] = self.normalize_angle(x_new[2] + omega * dt)
        return x_new

    def get_jacobian_f(self, x, dt, v):
        F = np.eye(3)
        F[0, 2] = -v * math.sin(x[2]) * dt
        F[1, 2] = v * math.cos(x[2]) * dt
        return F

    def h_wheels(self, x, x_prev):
        dx, dy = x[0] - x_prev[0], x[1] - x_prev[1]
        return np.array([math.sqrt(dx**2 + dy**2), self.normalize_angle(x[2] - x_prev[2])])

    def h_imu(self, x, x_prev):
        return np.array([self.normalize_angle(x[2] - x_prev[2])])

    def get_jacobian_h_wheels(self, x, x_prev):
        dx, dy = x[0] - x_prev[0], x[1] - x_prev[1]
        ds = math.sqrt(dx**2 + dy**2)
        H = np.zeros((2, 3))
        if ds > 1e-6:
            H[0, 0], H[0, 1] = dx / ds, dy / ds
        H[1, 2] = 1.0
        return H

    def predict_step(self, dt):
        # ___KF PREDICTION___
        x_nominal = np.zeros(3)
        F_kf = self.get_jacobian_f(x_nominal, dt, self.v_cmd)
        self.x_kf = self.f(self.x_kf, dt, self.v_cmd, self.omega_cmd)
        self.P_kf = F_kf @ self.P_kf @ F_kf.T + self.Q

        # ___EKF PREDICTION___
        F_ekf = self.get_jacobian_f(self.x_ekf, dt, self.v_cmd)
        self.x_ekf = self.f(self.x_ekf, dt, self.v_cmd, self.omega_cmd)
        self.P_ekf = F_ekf @ self.P_ekf @ F_ekf.T + self.Q

        # ___UKF PREDICTION___
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = self.x_ukf
        U = np.linalg.cholesky((self.n + self.lambda_ukf) * self.P_ukf)
        for k in range(self.n):
            sigmas[k+1], sigmas[self.n+k+1] = self.x_ukf + U[:, k], self.x_ukf - U[:, k]
        
        sigmas_f = np.array([self.f(s, dt, self.v_cmd, self.omega_cmd) for s in sigmas])
        self.x_ukf = np.dot(self.Wm, sigmas_f)
        self.x_ukf[2] = self.normalize_angle(self.x_ukf[2])
        
        self.P_ukf.fill(0)
        for i in range(2 * self.n + 1):
            dx = sigmas_f[i] - self.x_ukf
            dx[2] = self.normalize_angle(dx[2])
            self.P_ukf += self.Wc[i] * np.outer(dx, dx)
        self.P_ukf += self.Q

    def perform_update(self, z, sensor_type):
        # ___Select Configuration___
        if sensor_type == 'wheels':
            h_func = self.h_wheels
            H_gen = self.get_jacobian_h_wheels
            R = self.R_wheels
            angle_indices = [1]
            xp_kf, xp_ekf, xp_ukf = self.x_last_kf_wheels, self.x_last_ekf_wheels, self.x_last_ukf_wheels
        else: # IMU case
            h_func = self.h_imu
            H_gen = lambda x, xp: np.array([[0.0, 0.0, 1.0]])
            R = self.R_imu
            angle_indices = [0]
            xp_kf, xp_ekf, xp_ukf = self.x_last_kf_imu, self.x_last_ekf_imu, self.x_last_ukf_imu

        # ___KF UPDATE___
        H_kf = H_gen(np.zeros(3), np.zeros(3))
        kf_y = z - h_func(self.x_kf, xp_kf)
        for idx in angle_indices: kf_y[idx] = self.normalize_angle(kf_y[idx])
        S_kf = H_kf @ self.P_kf @ H_kf.T + R
        K_kf = self.P_kf @ H_kf.T @ np.linalg.inv(S_kf)
        self.x_kf = self.x_kf + K_kf @ kf_y
        self.x_kf[2] = self.normalize_angle(self.x_kf[2])
        self.P_kf = (np.eye(3) - K_kf @ H_kf) @ self.P_kf

        # ___EKF UPDATE___
        H_ekf = H_gen(self.x_ekf, xp_ekf)
        ekf_y = z - h_func(self.x_ekf, xp_ekf)
        for idx in angle_indices: ekf_y[idx] = self.normalize_angle(ekf_y[idx])
        S_ekf = H_ekf @ self.P_ekf @ H_ekf.T + R
        K_ekf = self.P_ekf @ H_ekf.T @ np.linalg.inv(S_ekf)
        self.x_ekf = self.x_ekf + K_ekf @ ekf_y
        self.x_ekf[2] = self.normalize_angle(self.x_ekf[2])
        self.P_ekf = (np.eye(3) - K_ekf @ H_ekf) @ self.P_ekf

        # ___UKF UPDATE___
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = self.x_ukf
        U = np.linalg.cholesky((self.n + self.lambda_ukf) * self.P_ukf)
        for k in range(self.n): sigmas[k+1], sigmas[self.n+k+1] = self.x_ukf + U[:, k], self.x_ukf - U[:, k]
            
        sigmas_h = np.array([h_func(s, xp_ukf) for s in sigmas])
        zp = np.dot(self.Wm, sigmas_h)
        
        dim_z = R.shape[0]
        St, Pxz = np.zeros((dim_z, dim_z)), np.zeros((self.n, dim_z))
        for i in range(2 * self.n + 1):
            dz, dx = sigmas_h[i] - zp, sigmas[i] - self.x_ukf
            for idx in angle_indices: dz[idx] = self.normalize_angle(dz[idx])
            dx[2] = self.normalize_angle(dx[2])
            St += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
            
        K_ukf = Pxz @ np.linalg.inv(St + R)
        ukf_y = z - zp
        for idx in angle_indices: ukf_y[idx] = self.normalize_angle(ukf_y[idx])
        
        self.x_ukf = self.x_ukf + K_ukf @ ukf_y
        self.x_ukf[2] = self.normalize_angle(self.x_ukf[2])
        self.P_ukf = self.P_ukf - K_ukf @ (St + R) @ K_ukf.T

        # ___Analysis Padding___
        if sensor_type == 'imu':
            kf_y_pub = np.array([0.0, kf_y[0]])
            ekf_y_pub = np.array([0.0, ekf_y[0]])
            ukf_y_pub = np.array([0.0, ukf_y[0]])
            self.x_last_kf_imu, self.x_last_ekf_imu, self.x_last_ukf_imu = np.copy(self.x_kf), np.copy(self.x_ekf), np.copy(self.x_ukf)
        else:
            kf_y_pub, ekf_y_pub, ukf_y_pub = kf_y, ekf_y, ukf_y
            self.x_last_kf_wheels, self.x_last_ekf_wheels, self.x_last_ukf_wheels = np.copy(self.x_kf), np.copy(self.x_ekf), np.copy(self.x_ukf)

        self.publish_analysis(kf_y_pub, ekf_y_pub, ukf_y_pub)

    def publish_analysis(self, kf_y, ekf_y, ukf_y):
        """ Publishes error residuals and variance diagonals for rqt_plot """
        for pub, y, P in [(self.kf_analysis_pub, kf_y, self.P_kf), 
                          (self.ekf_analysis_pub, ekf_y, self.P_ekf), 
                          (self.ukf_analysis_pub, ukf_y, self.P_ukf)]:
            msg = Float64MultiArray()
            msg.data = [float(y[0]), float(y[1]), float(P[0,0]), float(P[1,1]), float(P[2,2])]
            pub.publish(msg)

    def joint_state_callback(self, msg):
        try:
            l_idx, r_idx = msg.name.index('wheel_left_joint'), msg.name.index('wheel_right_joint')
        except ValueError: return
        
        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None: self.last_time = curr_time; return
        dt, self.last_time = curr_time - self.last_time, curr_time

        if self.last_phi_l is not None:
            ds = (self.R * (msg.position[l_idx] - self.last_phi_l) + self.R * (msg.position[r_idx] - self.last_phi_r)) / 2.0
            dtheta = (self.R * (msg.position[r_idx] - self.last_phi_r) - self.R * (msg.position[l_idx] - self.last_phi_l)) / self.L
            self.predict_step(dt)
            self.perform_update(np.array([ds, dtheta]), 'wheels')
            
        self.last_phi_l, self.last_phi_r = msg.position[l_idx], msg.position[r_idx]
        self.publish_all()

    def imu_callback(self, msg):
        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None: self.last_time = curr_time; self.last_imu_time = curr_time; return
        
        dt, self.last_time = curr_time - self.last_time, curr_time
        
        # Measurement interval for IMU integration
        if self.last_imu_time is None: self.last_imu_time = curr_time
        dt_imu = curr_time - self.last_imu_time
        self.last_imu_time = curr_time
        
        self.predict_step(dt)
        self.perform_update(np.array([msg.angular_velocity.z * dt_imu]), 'imu')
        self.publish_all()

    def cmd_vel_callback(self, msg):
        self.v_cmd, self.omega_cmd = msg.twist.linear.x, msg.twist.angular.z

    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.pub_filter(self.x_kf, self.P_kf, self.kf_odom_pub, self.kf_path_pub, self.kf_path, now)
        self.pub_filter(self.x_ekf, self.P_ekf, self.ekf_odom_pub, self.ekf_path_pub, self.ekf_path, now)
        self.pub_filter(self.x_ukf, self.P_ukf, self.ukf_odom_pub, self.ukf_path_pub, self.ukf_path, now)

    def pub_filter(self, x, P, odom_pub, path_pub, path_obj, now):
        o = Odometry()
        o.header.stamp, o.header.frame_id = now, 'odom'
        o.pose.pose.position.x, o.pose.pose.position.y = x[0], x[1]
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(x[2]/2), math.cos(x[2]/2)
        c = np.zeros(36); c[0], c[7], c[35] = P[0,0], P[1,1], P[2,2]
        o.pose.covariance = c.tolist()
        odom_pub.publish(o)
        path_obj.poses.append(PoseStamped(header=o.header, pose=o.pose.pose))
        path_obj.header.stamp = now
        path_pub.publish(path_obj)

def main():
    rclpy.init()
    node = LocalizationNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()