## Assignment Implementation: EE5531 Project 4
**Authors:** Anders Smitterberg and Reid Beckes

# Project 4: State Estimation (KF, EKF, UKF)
This repository contains a ROS 2 node that implements three distinct methods for robot state estimation: **Linearized Kalman Filter (KF)**, **Extended Kalman Filter (EKF)**, and **Unscented Kalman Filter (UKF)**. The node fuses data from commanded velocities (`/cmd_vel`), wheel encoders (`/joint_states`), and IMU measurements (`/imu`) to estimate the robot's pose and path.

### Prerequisites
* Ubuntu 24.04 or Linux Mint 22.3
* ROS 2 Jazzy
* `rosbag2_2026_02_03-15_36_14` recorded data folder

### Initial Setup and Testing
1. **Create and enter a new ROS 2 workspace:**
    ```bash
    mkdir -p ~/proj4_ws/src
    cd ~/proj4_ws/src
    ```
2. **Clone the repository:**
    ```bash
    git clone https://github.com/Robust-Autonomous-Systems-Laboratory/proj4_group1.
    git submodule update --init --recursive
    ```
3. **Install dependencies:**
    ```bash
    cd ~/proj4_ws
    rosdep install -i --from-path src --rosdistro jazzy -y --os=ubuntu:noble
    ```
4. **Build the workspace:**
    ```bash
    colcon build --symlink-install
    ```
5. **Source the environment:**
    ```bash
    source install/setup.bash
    ```
6. **Run the localization node:**
    ```bash
    ros2 run jasmitte_proj4 localization_node --ros-args -p use_sim_time:=true
    ```
7. **Play the rosbag:**
    * In a separate terminal, play the data with clock emulation to sync the node timestamps:
        ```bash
        ros2 bag play src/rosbag2_2026_02_03-15_36_14 --clock
        ```
8. **Visualization (RViz2):**
    * Open RViz2 using the provided configuration file to see the Path trajectories and Covariance ellipses:
        ```bash
        ros2 run rviz2 rviz2 -d install/jasmitte_proj4/share/jasmitte_proj4/rviz/proj4.rviz --ros-args -p use_sim_time:=true
        ```
9. **Analysis (rqt_plot):**
    * To view residuals and covariance stability for any filter (e.g., EKF):
        ```bash
        ros2 run rqt_plot rqt_plot /localization_node/ekf/analysis/data[0]:data[1]:data[2]
        ```

---

### Results and Analysis

#### Example Output

*Eventual description of our visual results*

#### 6b. Reducing Epistemic Uncertainty: Statistics of the Residual
*   **Linear Residuals:** 
*   **Angular Residuals:** 
*   **Comparison of Tuning:** 

#### 6c. Covariance Stability
*   **Behavior without measurements:** 
*   **KF Stability:** 
*   **EKF Stability:** 
*   **UKF Stability:** 

#### 6d. One Ground Truth Point
*   **Calculated Translation Error:** 
*   **Calculated Rotational Error:** 
*   **Best Performing Filter:** 

#### 7. Decision
*   **Selected Algorithm:** 
*   **Reasoning:** 

#### 8. Improvement
*   **Future Enhancements:** 

---

### References
*   Differential Kinematics: [Wikipedia: Differential wheeled robot](https://en.wikipedia.org/wiki/Differential_wheeled_robot)
*   Kalman Filtering: [Wikipedia: Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter)
*   Non-linear Filtering (EKF/UKF): [Wikipedia: Non-linear filters](https://en.wikipedia.org/wiki/Kalman_filter#Nonlinear_filters)

### AI Disclosure
Google Gemini was used to assist with the implementation of the Kalman Filter algorithms (KF, EKF, UKF), formatting the Python node structure, and establishing the ROS 2 launch and analysis topic configurations. Specifically, Gemini assisted in manual implementation of the UKF Sigma Point math and the Jacobian derivations for the EKF.
