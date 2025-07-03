#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class WallFollower(Node):

    def __init__(self):
        super().__init__('wall_follower_node')

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)
        
        self.timer = self.create_timer(0.1, self.timer_cb)

        # Define initial state and region dictionary
        self.regions_ = {
            'front': 0.0,
            'fleft': 0.0,
            'left': 0.0,
            'fright': 0.0,
            'right': 0.0,
        }

        self.state_ = 0
        self.state_dict_ = {
            0: 'find the wall',
            1: 'turn left',
            2: 'follow the wall',
        }

    def scan_cb(self, msg: LaserScan):
    
        front_ranges = msg.ranges[-15:] + msg.ranges[:15]
        front_filtered = [r for r in front_ranges if r <= 10]
        if len(front_filtered) == 0:
            front_dist = 10.0
        else:
            front_dist = min(front_filtered)
        
        fleft_ranges = msg.ranges[30:60]
        fleft_dist = sum(fleft_ranges) / len(fleft_ranges)

        left_ranges = msg.ranges[60:120]
        left_dist = sum(left_ranges) / len(left_ranges)
        
        fright_ranges = msg.ranges[-60:-30]
        fright_dist = sum(fright_ranges) / len(fright_ranges)
        
        right_ranges = msg.ranges[-120:-60]
        right_dist = sum(right_ranges) / len(right_ranges)
        
        self.regions_ = {
            'front':  front_dist,
            'fleft':  fleft_dist,
            'left':   left_dist,
            'fright': fright_dist,
            'right':  right_dist,
        }

        self.get_logger().info(f'Distance to the right wall: {self.regions_["right"]:.2f} m')

        self.take_action()

    def change_state(self, state):
        if state != self.state_:
            self.get_logger().info(f'Wall follower - [{state}] - {self.state_dict_[state]}')
            self.state_ = state

    def take_action(self):
        regions = self.regions_
        d = 0.7

        if regions['front'] > d and regions['fleft'] > d and regions['fright'] > d:
            self.change_state(0)
        elif regions['front'] < d and regions['fleft'] > d and regions['fright'] > d:
            self.change_state(1)
        elif regions['front'] > d and regions['fleft'] > d and regions['fright'] < d:
            self.change_state(2)
        elif regions['front'] > d and regions['fleft'] < d and regions['fright'] > d:
            self.change_state(0)
        elif regions['front'] < d and regions['fleft'] > d and regions['fright'] < d:
            self.change_state(1)
        elif regions['front'] < d and regions['fleft'] < d and regions['fright'] > d:
            self.change_state(1)
        elif regions['front'] < d and regions['fleft'] < d and regions['fright'] < d:
            self.change_state(1)
        elif regions['front'] > d and regions['fleft'] < d and regions['fright'] < d:
            self.change_state(0)
        else:
            self.get_logger().info('Unknown case')
            self.get_logger().info(str(regions))

    def find_wall(self):
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = -0.3
        return msg

    def turn_left(self):
        msg = Twist()
        msg.angular.z = 0.3
        return msg

    def follow_the_wall(self):
        msg = Twist()
        msg.linear.x = 0.5
        return msg

    def timer_cb(self):
        # Publish the message based on the current state
        if self.state_ == 0:
            msg = self.find_wall()
        elif self.state_ == 1:
            msg = self.turn_left()
        elif self.state_ == 2:
            msg = self.follow_the_wall()
        else:
            self.get_logger().error('Unknown state!')
            msg = Twist()
        
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()