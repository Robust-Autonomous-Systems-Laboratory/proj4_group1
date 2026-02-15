## Assignment Implementation: EE5531 Project 4
**Authors:** Anders Smitterberg and Reid Beckes

# Project 4: State Estimation (KF, EKF, UKF)
This repository contains a ROS 2 node that implements three distinct methods for robot state estimation: **Linearized Kalman Filter (KF)**, **Extended Kalman Filter (EKF)**, and **Unscented Kalman Filter (UKF)**. The node fuses data from commanded velocities (`/cmd_vel`), wheel encoders (`/joint_states`), and IMU measurements (`/imu`) to estimate the robot's pose and path.

### Prerequisites
* Ubuntu 24.04 or Linux Mint 22.3 or WSL (untested)
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
    git clone [https://github.com/Robust-Autonomous-Systems-Laboratory/proj4_group1](https://github.com/Robust-Autonomous-Systems-Laboratory/proj4_group1).
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
    * Each filter (KF, EKF, UKF) publishes a `Float64MultiArray` to its respective `/analysis` topic. To monitor a specific filter, add its topic to `rqt_plot`.
    * **Example (EKF):**
        ```bash
        ros2 run rqt_plot rqt_plot /localization_node/ekf/analysis/data[0]:data[1]
        ```
    * **Data Index Map:**
        * `data[0]`: Linear Residual (meters)
        * `data[1]`: Angular Residual (radians)
        * `data[2]`: Variance in X position ($\sigma^2_x$)
        * `data[3]`: Variance in Y position ($\sigma^2_y$)
        * `data[4]`: Variance in Heading ($\sigma^2_\theta$)

    * *Note: System asynchronous rates are handled via partial updates. When IMU measurements arrive, Distance Error (data[0]) is padded with 0.0 to maintain consistent array dimensions for plotting. This shows up in the plots.*
---

### Results and Analysis

#### Example Output
![Rviz Depiction of Three Paths](<results.png>)
*Three different robot paths, all determined from the same rosbag. Red is the kf, Green is the EKF, Blue is the UKF*

#### 6b. Reducing Uncertainty: Statistics of the Residual
* **Linear Residuals:** Monitored via `data[0]`. These represent the innovation between wheel odometry and the predicted state.
* **Angular Residuals:** Monitored via `data[1]`. These represent the fusion of the gyro and wheel encoders. 
* **Comparison of Tuning:** Observations on how $R_{wheels}$ vs $R_{imu}$ affects residual magnitude.

#### 6c. Covariance Stability
* **Behavior without measurements:** Expected growth of $P$ diagonals (`data[2:4]`) during prediction-only phases.
* **KF Stability:** Analysis of the linearized covariance behavior.
* **EKF Stability:** Analysis of the state-dependent Jacobian effects on $P$.
* **UKF Stability:** Observation of the sigma-point propagation of uncertainty.

#### 6d. One Ground Truth Point
* **Calculated Translation Error:** Difference between final estimated $(x,y)$ and ground truth.
* **Calculated Rotational Error:** Difference between final estimated $\theta$ and ground truth.
* **Best Performing Filter:** Comparison of final pose accuracy.

#### 7. Decision
* **Selected Algorithm:** * **Reasoning:** 

#### 8. Improvement
* **Future Enhancements:** ---

### References
* Differential Kinematics: [Wikipedia: Differential wheeled robot](https://en.wikipedia.org/wiki/Differential_wheeled_robot)
* Kalman Filtering: [Wikipedia: Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter)
* Non-linear Filtering (EKF/UKF): [Wikipedia: Non-linear filters](https://en.wikipedia.org/wiki/Kalman_filter#Nonlinear_filters)
* Robot Parameters : [Turtlebot3: Documentation](https://emanual.robotis.com/docs/en/platform/turtlebot3/features/)

### AI Disclosure
Google Gemini was used to assist with the implementation of the Kalman Filter algorithms (KF, EKF, UKF), formatting the Python node structure, and establishing the ROS 2 launch and analysis topic configurations. Specifically, Gemini assisted in the implementation of asynchronous sensor fusion using partial updates (1x1 IMU updates vs 2x2 wheel updates) to prevent state-estimate issues.