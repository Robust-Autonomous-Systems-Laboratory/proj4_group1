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
    KF: Linearized Kalman Filter (around nominal trajectory).
    EKF: Extended Kalman Filter (around current estimate).
    UKF: Unscented Kalman Filter (sigma point propagation).
    """
    def __init__(self):
        super().__init__('localization_node')
        
        # ___Burgerbot Constants___
        self.R = 0.033
        self.L = 0.165 # This is slightly wider than the actual robot specs 
        self.tau = 0.8 # Acceleration time constant. This was deemed necessary after much testing to get good results.
        
        # ___Filter States (x, y, theta, v, omega)___
        self.n = 5
        self.x_kf = np.zeros(self.n)
        self.P_kf = np.eye(self.n) * 0.1 
        self.x_ekf = np.zeros(self.n)
        self.P_ekf = np.eye(self.n) * 0.1 
        self.x_ukf = np.zeros(self.n)
        self.P_ukf = np.eye(self.n) * 0.1 

        self.kf_res_imu, self.kf_res_wheels = [0.0, 0.0], [0.0, 0.0]
        self.ekf_res_imu, self.ekf_res_wheels = [0.0, 0.0], [0.0, 0.0]
        self.ukf_res_imu, self.ukf_res_wheels = [0.0, 0.0], [0.0, 0.0]
        
        self.kf_s_imu, self.kf_s_wheels = [0.0, 0.0], [0.0, 0.0]
        self.ekf_s_imu, self.ekf_s_wheels = [0.0, 0.0], [0.0, 0.0]
        self.ukf_s_imu, self.ukf_s_wheels = [0.0, 0.0], [0.0, 0.0]

        # Trajectory for KF Linearization
        self.x_nom = np.zeros(self.n)

        # ___Noise Matrices___
        self.Q = np.diag([1e-10, 1e-10, 1e-9, 0.005, 0.005]) 
        self.R_imu = np.diag([0.001, 1.0])
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

        # Parameters
        self.declare_parameter('prediction_only', False)

        self.get_logger().info('Localization Node Started.')

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
        ax_expected = (v_cmd - x[3]) / self.tau
        return np.array([x[4], ax_expected])

    def h_wheels(self, x):
        return np.array([x[3], x[4]])

    def predict_step(self, dt):
        # 1. KF
        self.x_nom = self.f(self.x_nom, dt, self.v_cmd, self.omega_cmd)
        self.x_kf = self.f(self.x_kf, dt, self.v_cmd, self.omega_cmd)
        F_lkf = self.get_jacobian_f(self.x_nom, dt)
        self.P_kf = F_lkf @ self.P_kf @ F_lkf.T + self.Q

        # 2. EKF
        self.x_ekf = self.f(self.x_ekf, dt, self.v_cmd, self.omega_cmd)
        F_ekf = self.get_jacobian_f(self.x_ekf, dt)
        self.P_ekf = F_ekf @ self.P_ekf @ F_ekf.T + self.Q

        # 3. UKF Prediction
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
        if self.get_parameter('prediction_only').get_parameter_value().bool_value:
            # Skip updates but still publish analysis for covariance plotting
            dummy_y = np.zeros(2)
            dummy_S = np.eye(2)
            self.publish_analysis(dummy_y, dummy_y, dummy_y, dummy_S, dummy_S, dummy_S, sensor_type)
            return

        if sensor_type == 'imu':
            h_func = lambda x: self.h_imu(x, self.v_cmd)
            H = np.zeros((2, 5)); H[0, 4] = 1.0; H[1, 3] = -1.0 / self.tau
        else: # wheels
            h_func = lambda x: self.h_wheels(x)
            H = np.zeros((2, 5)); H[0, 3] = 1.0; H[1, 4] = 1.0

        # KF Update
        kf_y = z - h_func(self.x_kf)
        S_kf = H @ self.P_kf @ H.T + R
        K_kf = self.P_kf @ H.T @ np.linalg.inv(S_kf)
        self.x_kf += K_kf @ kf_y
        self.x_kf[2] = self.normalize_angle(self.x_kf[2])
        self.P_kf = (np.eye(self.n) - K_kf @ H) @ self.P_kf

        # EKF Update
        ekf_y = z - h_func(self.x_ekf)
        S_ekf = H @ self.P_ekf @ H.T + R
        K_ekf = self.P_ekf @ H.T @ np.linalg.inv(S_ekf)
        self.x_ekf += K_ekf @ ekf_y
        self.x_ekf[2] = self.normalize_angle(self.x_ekf[2])
        self.P_ekf = (np.eye(self.n) - K_ekf @ H) @ self.P_ekf

        # UKF Update
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
        S_ukf = St + R
        K_ukf = Pxz @ np.linalg.inv(S_ukf)
        ukf_y = z - zp
        self.x_ukf += K_ukf @ ukf_y
        self.x_ukf[2] = self.normalize_angle(self.x_ukf[2])
        self.P_ukf = self.make_pd(self.P_ukf - K_ukf @ S_ukf @ K_ukf.T)
        self.publish_analysis(kf_y, ekf_y, ukf_y, S_kf, S_ekf, S_ukf, sensor_type)

    def publish_analysis(self, kf_y, ekf_y, ukf_y, S_kf, S_ekf, S_ukf, sensor_type):
        s_kf_diag = 3.0 * np.sqrt(np.diag(S_kf))
        s_ekf_diag = 3.0 * np.sqrt(np.diag(S_ekf))
        s_ukf_diag = 3.0 * np.sqrt(np.diag(S_ukf))
        
        if sensor_type == 'imu':
            self.kf_res_imu, self.kf_s_imu = kf_y, s_kf_diag
            self.ekf_res_imu, self.ekf_s_imu = ekf_y, s_ekf_diag
            self.ukf_res_imu, self.ukf_s_imu = ukf_y, s_ukf_diag
        else:
            self.kf_res_wheels, self.kf_s_wheels = kf_y, s_kf_diag
            self.ekf_res_wheels, self.ekf_s_wheels = ekf_y, s_ekf_diag
            self.ukf_res_wheels, self.ukf_s_wheels = ukf_y, s_ukf_diag

        data_list = [
            (self.kf_analysis_pub, self.kf_res_imu, self.kf_res_wheels, self.kf_s_imu, self.kf_s_wheels, self.P_kf),
            (self.ekf_analysis_pub, self.ekf_res_imu, self.ekf_res_wheels, self.ekf_s_imu, self.ekf_s_wheels, self.P_ekf),
            (self.ukf_analysis_pub, self.ukf_res_imu, self.ukf_res_wheels, self.ukf_s_imu, self.ukf_s_wheels, self.P_ukf)
        ]

        for pub, res_imu, res_wheels, s_imu, s_wheels, P in data_list:
            msg = Float64MultiArray()
            msg.data = [
                float(res_imu[0]), float(res_imu[1]), float(res_wheels[0]), float(res_wheels[1]),
                float(s_imu[0]), float(s_imu[1]), float(s_wheels[0]), float(s_wheels[1]),
                -float(s_imu[0]), -float(s_imu[1]), -float(s_wheels[0]), -float(s_wheels[1]),
                float(P[0,0]), float(P[1,1]), float(P[2,2]), float(P[3,3]), float(P[4,4])
            ]
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
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        print("\n" + "="*50)
        print("2e. Ground Truth Evaluation Results")
        print("Note: Assumes 'Initial Position - Final Position'")
        print("-" * 50)
        print(f"{'Algorithm':<10} | {'Trans. Error (m)':<18} | {'Rot. Error (rad)':<18}")
        print("-" * 50)
        
        for name, state in [("KF", node.x_kf), ("EKF", node.x_ekf), ("UKF", node.x_ukf)]:
            trans_err = np.sqrt(state[0]**2 + state[1]**2)
            rot_err = abs(node.normalize_angle(state[2]))
            print(f"{name:<10} | {trans_err:<18.4f} | {rot_err:<18.4f}")
            
        print("="*50 + "\n")
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()