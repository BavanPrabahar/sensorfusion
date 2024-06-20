User
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import csv

class EKFNode:
    def __init__(self):
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
        self.R = np.diag([0.1, 0.1])
        self.P = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
        self.x_hat = np.zeros((5, 1))
        self.odom_sub = Subscriber('/odom/filtered', Odometry)
        self.scan_sub = Subscriber('/scan', LaserScan)
        self.ts_msg = ApproximateTimeSynchronizer([self.odom_sub, self.scan_sub], 10,0.1)
        self.ts_msg.registerCallback(self.process_msg)
        self.cumulative_displacement_x = 0.0
        self.cumulative_displacement_y = 0.0
        self.previous_time = rospy.Time.now()

    def calc_angle_change(self, angular_z, delta_time):
        return angular_z * delta_time

    def predict(self, F):
        self.x_hat = np.dot(F, self.x_hat)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, mean_range, mean_angle,lin_x,lin_y,change_in_angle):
        H =  np.eye(5)
        z = np.array([[lin_x],[lin_y],[change_in_angle],[mean_range],
                      [mean_angle]])
        y = z - np.dot(H, self.x_hat)
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        self.x_hat = self.x_hat + np.dot(K, y)
        self.P = np.dot(np.eye(5) - np.dot(K, H), self.P)

    def process_msg(self, odom_msg, scan_msg):
        lin_x = odom_msg.twist.twist.linear.x
        lin_y = odom_msg.twist.twist.linear.y
        ang_z = odom_msg.twist.twist.angular.z
        current_time = rospy.Time.now()
        delta_time = (current_time - self.previous_time).to_sec()
       
        self.previous_time = current_time

        displacement_x = odom_msg.twist.twist.linear.x * delta_time
        displacement_y = odom_msg.twist.twist.linear.y * delta_time
        change_in_angle = self.calc_angle_change(ang_z, delta_time)

        self.cumulative_displacement_x += displacement_x
        self.cumulative_displacement_y += displacement_y

        lin_x = self.cumulative_displacement_x
        lin_y = self.cumulative_displacement_y
     
       
        F = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])
        ranges = scan_msg.ranges
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment) + change_in_angle
        ranges1, angles1 = self.apply_transform(ranges, angles, change_in_angle, lin_x, lin_y)
        mean_range = np.mean(ranges1)
        mean_angle = np.mean(angles1)
        self.predict(F)
        self.update(mean_range, mean_angle,lin_x,lin_y,change_in_angle)
        if abs(mean_range - self.x_hat[3][0]) > 0.01: 
            diff_range = self.x_hat[3][0] - mean_range
            ranges3 = [r + diff_range for r in ranges]
        else:
            ranges3 = ranges
   
        if abs(mean_angle - self.x_hat[4][0]) > 0.01: 
            diff_angle = self.x_hat[4][0] - mean_angle
            angles3 = [a + diff_angle for a in angles]
        else:
            angles3 = angles
        x_coords = [ranges3[i] * np.cos(angles3[i]) for i in range(len(ranges3))]
        y_coords = [ranges3[i] * np.sin(angles3[i]) for i in range(len(ranges3))]
        csv_file_path = "a.csv"
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y'])
            for x, y in zip(x_coords, y_coords):
                writer.writerow([x, y])
        csvfile.close()    
        rospy.loginfo("Estimated State: lin_x={}, lin_y={}, change_in_angle={}, mean_range={}, mean_angle={}".format(
            self.x_hat[0][0], self.x_hat[1][0], self.x_hat[2][0], self.x_hat[3][0], self.x_hat[4][0]))

    def apply_transform(self, ranges, angles, change_in_angle, linear_x, linear_y):
        transform_matrix = np.array([[np.cos(change_in_angle), -np.sin(change_in_angle), linear_x],
                                     [np.sin(change_in_angle), np.cos(change_in_angle), linear_y],
                                     [0, 0, 1]])
        homogenous_coords = np.vstack((ranges * np.cos(angles), ranges * np.sin(angles), np.ones_like(ranges)))
        transformed_coords = np.dot(transform_matrix, homogenous_coords)
        ranges1 = transformed_coords[0]
        angles1 = transformed_coords[1]
        return ranges1, angles1

    def subscriber_node(self):
        rospy.init_node('subscriber_node', anonymous=True)
        try:    
            self.ts_msg
        except AttributeError as e:
            rospy.logerr("Error subscribing to topics: {}".format(e))
            return
    rospy.spin()

if __name__ == '__main__':
    try:
        node = EKFNode()
        node.subscriber_node()
    except rospy.ROSInterruptException:
        pass

