# Sensor Fusion with Extended Kalman Filter (EKF)

This repository contains a ROS (Robot Operating System) node that performs sensor fusion using an Extended Kalman Filter (EKF). The node synchronizes Odometry and LaserScan data to estimate the robot's state and transform LiDAR scan points into a consistent coordinate frame.

## Features

- **Sensor Fusion**: Subscribes to `/odom/filtered` (`nav_msgs/Odometry`) and `/scan` (`sensor_msgs/LaserScan`) topics.
- **Time Synchronization**: Utilizes `message_filters.ApproximateTimeSynchronizer` to match incoming odometry and scan messages.
- **Extended Kalman Filter (EKF)**: Fuses linear velocity, change in angle, and mean laser scan ranges/angles to maintain an accurate state estimate.
- **Scan Transformation**: Applies the estimated state transformations to laser scan points.
- **CSV Logging**: Outputs the transformed X and Y coordinates of the laser scan to a CSV file (`a.csv`).

## Prerequisites

- ROS (Tested on Noetic/Melodic)
- Python 3
- `rospy`
- `numpy`
- `sensor_msgs`
- `nav_msgs`
- `message_filters`

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BavanPrabahar/sensorfusion.git
   cd sensorfusion
   ```

2. **Ensure your ROS environment is sourced:**
   ```bash
   source /opt/ros/<distro>/setup.bash
   ```

3. **Run the node:**
   Ensure you have a ROS master running (`roscore`) and publishers for `/odom/filtered` and `/scan`.
   ```bash
   python3 map.py
   ```

## Output

- The script will output estimated state logs to the console using `rospy.loginfo`.
- Transformed laser scan coordinates will be saved to `a.csv` in the current working directory.

## License

This project is open-source. Please see the repository for more details.
