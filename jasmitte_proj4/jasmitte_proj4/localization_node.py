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
    ROS 2 Node for state estimation using KF, EKF, and UKF.
    """
    def __init__(self):
        super().__init__('localization_node')
        
        # ___Burgerbot Constants___
        self.R = 0.033
        self.L = 0.170 # Slightly larger than 0.160 but seems to correct overshoot
        self.tau = 0.8
        
        # ___Filter States (x, y, theta, v, omega)___
        self.n = 5
        self.x_kf = np.zeros(self.n)
        self.P_kf = np.eye(self.n) * 0.1 
        self.x_ekf = np.zeros(self.n)
        self.P_ekf = np.eye(self.n) * 0.1 
        self.x_ukf = np.zeros(self.n)
        self.P_ukf = np.eye(self.n) * 0.1 

        # ___Noise Matrices___
        self.Q = np.diag([1e-10, 1e-10, 1e-9, 0.005, 0.005]) 
        
        # IMU noise: [omega, ax]
        self.R_imu = np.diag([0.001, 1.0])
        
        # Wheel noise: [v, omega]
        self.R_wheels = np.diag([0.05, 0.05])
        
        # ___UKF Parameters___
        self.alpha = 0.1 
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ukf = self.alpha**2 * (self.n + self.kappa) - self.n
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lambda_ukf)))
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lambda_ukf)))
        self.Wm[0] = self.lambda_ukf / (self.n + self.lambda_ukf)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        
        self.last_phi_l, self.last_phi_r = None, None
        self.v_cmd, self.omega_cmd = 0.0, 0.0
        self.last_time = None
        self.last_wheels_time = None
        self.last_imu_time = None
        
        path_qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        # Publishers
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

        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.cmd_vel_sub = self.create_subscription(TwistStamped, '/cmd_vel', self.cmd_vel_callback, 10)

        self.get_logger().info('Localization Node Started. Smoother Velocity-based Updates Restored.')

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def make_pd(self, P):
        P = (P + P.T) / 2.0
        ee, ev = np.linalg.eigh(P)
        ee = np.maximum(ee, 1e-9)
        return ev @ np.diag(ee) @ ev.T

    def get_sigmas(self, x, P):
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x
        try:
            U = np.linalg.cholesky((self.n + self.lambda_ukf) * P)
        except np.linalg.LinAlgError:
            U = np.linalg.cholesky((self.n + self.lambda_ukf) * self.make_pd(P + np.eye(self.n) * 1e-7))
        for k in range(self.n):
            sigmas[k+1], sigmas[self.n+k+1] = x + U[:, k], x - U[:, k]
        return sigmas

    def f(self, x, dt, v_cmd, omega_cmd):
        x_new = np.copy(x)
        x_new[0] += x[3] * math.cos(x[2]) * dt
        x_new[1] += x[3] * math.sin(x[2]) * dt
        x_new[2] = self.normalize_angle(x[2] + x[4] * dt)
        x_new[3] += (v_cmd - x[3]) * (dt / self.tau)
        x_new[4] += (omega_cmd - x[4]) * (dt / self.tau)
        return x_new

    def get_jacobian_f(self, x, dt):
        F = np.eye(self.n)
        F[0, 2] = -x[3] * math.sin(x[2]) * dt
        F[0, 3] = math.cos(x[2]) * dt
        F[1, 2] = x[3] * math.cos(x[2]) * dt
        F[1, 3] = math.sin(x[2]) * dt
        F[2, 4] = dt
        F[3, 3] = 1.0 - (dt / self.tau)
        F[4, 4] = 1.0 - (dt / self.tau)
        return F

    def h_imu(self, x, v_cmd):
        """ IMU Prediction: [omega, ax] """
        ax_expected = (v_cmd - x[3]) / self.tau
        return np.array([x[4], ax_expected])

    def h_wheels(self, x):
        """ Wheel Prediction: [v, omega] """
        return np.array([x[3], x[4]])

    def predict_step(self, dt):
        self.x_kf = self.f(self.x_kf, dt, self.v_cmd, self.omega_cmd)
        F_kf = self.get_jacobian_f(self.x_kf, dt)
        self.P_kf = F_kf @ self.P_kf @ F_kf.T + self.Q
        self.x_ekf = self.f(self.x_ekf, dt, self.v_cmd, self.omega_cmd)
        F_ekf = self.get_jacobian_f(self.x_ekf, dt)
        self.P_ekf = F_ekf @ self.P_ekf @ F_ekf.T + self.Q
        sigmas = self.get_sigmas(self.x_ukf, self.P_ukf)
        sigmas_f = np.array([self.f(s, dt, self.v_cmd, self.omega_cmd) for s in sigmas])
        self.x_ukf = np.dot(self.Wm, sigmas_f)
        self.x_ukf[2] = self.normalize_angle(self.x_ukf[2])
        self.P_ukf.fill(0)
        for i in range(2 * self.n + 1):
            dx = sigmas_f[i] - self.x_ukf
            dx[2] = self.normalize_angle(dx[2])
            self.P_ukf += self.Wc[i] * np.outer(dx, dx)
        self.P_ukf = self.make_pd(self.P_ukf + self.Q)

    def perform_update(self, z, R, sensor_type):
        if sensor_type == 'imu':
            h_func = lambda x: self.h_imu(x, self.v_cmd)
            H = np.zeros((2, 5)); H[0, 4] = 1.0; H[1, 3] = -1.0 / self.tau
        else: # wheels
            h_func = lambda x: self.h_wheels(x)
            H = np.zeros((2, 5)); H[0, 3] = 1.0; H[1, 4] = 1.0

        # KF
        kf_y = z - h_func(self.x_kf)
        S_kf = H @ self.P_kf @ H.T + R
        K_kf = self.P_kf @ H.T @ np.linalg.inv(S_kf)
        self.x_kf += K_kf @ kf_y
        self.x_kf[2] = self.normalize_angle(self.x_kf[2])
        self.P_kf = (np.eye(self.n) - K_kf @ H) @ self.P_kf
        # EKF
        ekf_y = z - h_func(self.x_ekf)
        S_ekf = H @ self.P_ekf @ H.T + R
        K_ekf = self.P_ekf @ H.T @ np.linalg.inv(S_ekf)
        self.x_ekf += K_ekf @ ekf_y
        self.x_ekf[2] = self.normalize_angle(self.x_ekf[2])
        self.P_ekf = (np.eye(self.n) - K_ekf @ H) @ self.P_ekf
        # UKF
        sigmas = self.get_sigmas(self.x_ukf, self.P_ukf)
        sigmas_h = np.array([h_func(s) for s in sigmas])
        zp = np.dot(self.Wm, sigmas_h)
        dim_z = R.shape[0]
        St, Pxz = np.zeros((dim_z, dim_z)), np.zeros((self.n, dim_z))
        for i in range(2 * self.n + 1):
            dz, dx = sigmas_h[i] - zp, sigmas[i] - self.x_ukf
            dx[2] = self.normalize_angle(dx[2])
            St += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        K_ukf = Pxz @ np.linalg.inv(St + R)
        ukf_y = z - zp
        self.x_ukf += K_ukf @ ukf_y
        self.x_ukf[2] = self.normalize_angle(self.x_ukf[2])
        self.P_ukf = self.make_pd(self.P_ukf - K_ukf @ (St + R) @ K_ukf.T)
        self.publish_analysis(kf_y, ekf_y, ukf_y)

    def publish_analysis(self, kf_y, ekf_y, ukf_y):
        for pub, y, P in [(self.kf_analysis_pub, kf_y, self.P_kf), 
                          (self.ekf_analysis_pub, ekf_y, self.P_ekf), 
                          (self.ukf_analysis_pub, ukf_y, self.P_ukf)]:
            msg = Float64MultiArray()
            msg.data = [float(y[0]), float(y[1]), float(P[0,0]), float(P[1,1]), float(P[2,2]), float(P[3,3])]
            pub.publish(msg)

    def joint_state_callback(self, msg):
        try:
            l_idx, r_idx = msg.name.index('wheel_left_joint'), msg.name.index('wheel_right_joint')
        except ValueError: return
        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None: self.last_time = curr_time
        if self.last_wheels_time is None: 
            self.last_wheels_time = curr_time
            self.last_phi_l, self.last_phi_r = msg.position[l_idx], msg.position[r_idx]
            return
        dt, self.last_time = curr_time - self.last_time, curr_time
        self.predict_step(dt)
        dt_w = curr_time - self.last_wheels_time
        self.last_wheels_time = curr_time
        if dt_w < 1e-6: return
        v_l = self.R * (msg.position[l_idx] - self.last_phi_l) / dt_w
        v_r = self.R * (msg.position[r_idx] - self.last_phi_r) / dt_w
        self.perform_update(np.array([(v_l+v_r)/2.0, (v_r-v_l)/self.L]), self.R_wheels, 'wheels')
        self.last_phi_l, self.last_phi_r = msg.position[l_idx], msg.position[r_idx]
        self.publish_all()

    def imu_callback(self, msg):
        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None: self.last_time = curr_time
        if self.last_imu_time is None: self.last_imu_time = curr_time; return
        dt, self.last_time = curr_time - self.last_time, curr_time
        self.predict_step(dt)
        dt_imu = curr_time - self.last_imu_time
        self.last_imu_time = curr_time
        self.perform_update(np.array([msg.angular_velocity.z, msg.linear_acceleration.x]), self.R_imu, 'imu')
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