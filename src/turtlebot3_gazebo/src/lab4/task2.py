#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Quaternion
from std_msgs.msg import Header

import yaml
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import heapq
import os
import math

from scipy.ndimage import binary_dilation

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        """
        Compute the PID control output.
        """
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class Navigation(Node):
    def __init__(self, node_name='Navigation'):
        super().__init__(node_name)

        # File paths for the map and yaml (adjust paths as necessary)
        self.yaml_path = "/home/santaslug/sim_ws/src/turtlebot3_gazebo/maps/sync_classroom_map.yaml"
        self.pgm_path = "/home/santaslug/sim_ws/src/turtlebot3_gazebo/maps/sync_classroom_map.pgm"

        self.goal_pose = None
        self.ttbot_pose = None
        self.map_params = None
        self.occupancy_grid = None

        # PID controllers for linear and angular speeds
        self.linear_pid = PIDController(kp=0.5, ki=0.0, kd=0.01)  # Tune gains
        self.angular_pid = PIDController(kp=0.5, ki=0.0, kd=0.01)  # Tune gains


        # Subscribers
        self.create_subscription(
            PoseStamped, 
            '/move_base_simple/goal', 
            self.__goal_pose_cbk, 
            10
        )
        self.create_subscription(
            PoseWithCovarianceStamped, 
            '/amcl_pose', 
            self.__ttbot_pose_cbk, 
            10
        )

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.rate = self.create_rate(10)

        # Load map parameters and occupancy grid
        self.map_params = self.load_map_parameters(self.yaml_path, self.pgm_path)
        self.occupancy_grid = self.create_occupancy_grid(self.pgm_path)

        # Flip the occupancy grid so (0,0) is bottom-left
        self.occupancy_grid = np.flipud(self.occupancy_grid)

        # Inflate (dilate) the occupancy grid obstacles
        obstacle_mask = (self.occupancy_grid == 0)
        structure = np.ones((7,7))
        inflated_obstacles = binary_dilation(obstacle_mask, structure=structure)
        self.occupancy_grid = np.where(inflated_obstacles, 0, 1)

    def load_map_parameters(self, yaml_path, pgm_path):
        if not os.path.exists(yaml_path) or not os.path.exists(pgm_path):
            self.get_logger().error(f"Map files not found at {yaml_path} and {pgm_path}")
            raise FileNotFoundError("Map files not found.")

        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        x_offset = data['origin'][0]
        y_offset = data['origin'][1]
        resolution = data['resolution']

        with Image.open(pgm_path) as img:
            yaml_width, yaml_height = img.size

        world_width = yaml_width * resolution
        world_height = yaml_height * resolution

        self.get_logger().info(
            f"Map loaded: width={world_width}, height={world_height}, resolution={resolution}"
        )

        return {
            "x_offset": x_offset,
            "y_offset": y_offset,
            "world_width": world_width,
            "world_height": world_height,
            "yaml_width": yaml_width,
            "yaml_height": yaml_height,
            "resolution": resolution
        }

    def create_occupancy_grid(self, pgm_path):
        with Image.open(pgm_path) as img:
            img_data = np.array(img)

        unique_values = np.unique(img_data)
        self.get_logger().info(f"Unique pixel values in PGM: {unique_values}")

        # 0 = obstacle, 1 = free
        threshold = 100
        grid = np.where(img_data < threshold, 0, 1)
        self.get_logger().info("Occupancy grid created.")
        return grid

    def world_to_map(self, world_x, world_y):
        x_offset = self.map_params["x_offset"]
        y_offset = self.map_params["y_offset"]
        world_width = self.map_params["world_width"]
        world_height = self.map_params["world_height"]
        yaml_width = self.map_params["yaml_width"]
        yaml_height = self.map_params["yaml_height"]

        x_map = (world_x - x_offset) * yaml_width / world_width
        y_map = (world_y - y_offset) * yaml_height / world_height
        return int(round(x_map)), int(round(y_map))

    def map_to_world(self, map_x, map_y):
        x_offset = self.map_params["x_offset"]
        y_offset = self.map_params["y_offset"]
        world_width = self.map_params["world_width"]
        world_height = self.map_params["world_height"]
        yaml_width = self.map_params["yaml_width"]
        yaml_height = self.map_params["yaml_height"]

        x_world = (map_x * world_width / yaml_width) + x_offset
        y_world = (map_y * world_height / yaml_height) + y_offset
        return x_world, y_world

    def __goal_pose_cbk(self, data):
        self.goal_pose = data
        self.get_logger().info(
            'Received goal pose: x={:.4f}, y={:.4f}'.format(
                self.goal_pose.pose.position.x, 
                self.goal_pose.pose.position.y
            )
        )

    def __ttbot_pose_cbk(self, data):
        self.ttbot_pose = data.pose.pose
        self.get_logger().info(
            'Current robot pose: x={:.4f}, y={:.4f}'.format(
                self.ttbot_pose.position.x, 
                self.ttbot_pose.position.y
            )
        )

    def heuristic(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def neighbors(self, x, y, grid):
        # 8-connected movement
        moves = [
            (x-1, y), (x+1, y),
            (x, y-1), (x, y+1),
            (x-1, y-1), (x-1, y+1),
            (x+1, y-1), (x+1, y+1)
        ]
        valid_moves = []
        for nx, ny in moves:
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                if grid[ny, nx] == 1:  # free
                    valid_moves.append((nx, ny))
        return valid_moves

    def a_star_search(self, grid, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        self.get_logger().info("Starting A* search...")

        while frontier:
            current_priority, current = heapq.heappop(frontier)

            if current == goal:
                self.get_logger().info("Goal reached in A* search.")
                break

            for nxt in self.neighbors(current[0], current[1], grid):
                new_cost = cost_so_far[current] + self.heuristic(current, nxt)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.heuristic(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            self.get_logger().info("No path found!")
            return []

        # Reconstruct path
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def quaternion_to_yaw(self, quat):
        # Converts a quaternion to a yaw angle
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def create_waypoints(self, path, step=5):

        if not path:
            return []
        waypoints = []
        for i in range(0, len(path), step):
            waypoints.append(path[i])
        # Make sure the final goal is included
        if waypoints[-1] != path[-1]:
            waypoints.append(path[-1])
        self.get_logger().info(f"Waypoints created: {len(waypoints)} waypoints.")
        return waypoints

    def waypoints_to_ros_path(self, waypoints):

        ros_path = Path()
        ros_path.header = Header()
        ros_path.header.stamp = self.get_clock().now().to_msg()
        ros_path.header.frame_id = 'map'

        for (mx, my) in waypoints:
            wx, wy = self.map_to_world(mx, my)
            pose = PoseStamped()
            pose.header = ros_path.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            ros_path.poses.append(pose)

        return ros_path

    def follow_waypoints(self, waypoints):

        if not waypoints:
            self.get_logger().info("No waypoints to follow.")
            return

        distance_tolerance = 0.15
        angle_tolerance = 0.3
        max_linear_speed = 0.4  # m/s
        max_angular_speed = 1.5  # rad/s

        for idx, (mx, my) in enumerate(waypoints):
            goal_x, goal_y = self.map_to_world(mx, my)
            self.get_logger().info(f"Heading to waypoint {idx+1}/{len(waypoints)}: ({goal_x:.2f}, {goal_y:.2f})")

            while rclpy.ok():
                if self.ttbot_pose is None:
                    rclpy.spin_once(self, timeout_sec=0.1)
                    continue

                # Current robot position and orientation
                current_x = self.ttbot_pose.position.x
                current_y = self.ttbot_pose.position.y
                current_yaw = self.quaternion_to_yaw(self.ttbot_pose.orientation)

                # Calculate distance and heading to the waypoint
                dx = goal_x - current_x
                dy = goal_y - current_y
                distance = math.sqrt(dx**2 + dy**2)
                desired_yaw = math.atan2(dy, dx)
                yaw_error = desired_yaw - current_yaw
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))  # Normalize to [-pi, pi]

                if distance < distance_tolerance:
                    self.get_logger().info(f"Reached waypoint {idx+1}.")
                    break

                # Compute PID outputs
                linear_speed = self.linear_pid.compute(distance, dt=0.1)
                linear_speed = max(-max_linear_speed, min(max_linear_speed, linear_speed))

                angular_speed = self.angular_pid.compute(yaw_error, dt=0.1)
                angular_speed = max(-max_angular_speed, min(max_angular_speed, angular_speed))

                # Create twist
                twist = Twist()
                if abs(yaw_error) > angle_tolerance:
                    # Turn in place if heading error is large
                    twist.linear.x = 0.0
                    twist.angular.z = angular_speed
                else:
                    # Move forward and correct heading
                    twist.linear.x = linear_speed
                    twist.angular.z = angular_speed

                self.cmd_vel_pub.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.1)

        # Stop after final waypoint
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        self.get_logger().info("All waypoints reached. Robot stopped.")

    def visualize_path(self, grid, path, start, goal):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(grid, cmap='gray', origin='lower')
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'r-', linewidth=2, label='Path')

        ax.plot(start[0], start[1], 'go', label='Start')
        ax.plot(goal[0], goal[1], 'bo', label='Goal')
        ax.legend()
        plt.title("A* Path on Occupancy Grid")
        plt.show()

    def visualize_initial_positions(self, start, goal):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(self.occupancy_grid, cmap='gray', origin='lower')
        ax.plot(start[0], start[1], 'go', label='Start')
        ax.plot(goal[0], goal[1], 'bo', label='Goal')
        ax.legend()
        plt.title("Occupancy Grid with Start and Goal")
        plt.show()

    def path_to_ros_path(self, path):
        ros_path = Path()
        ros_path.header = Header()
        ros_path.header.stamp = self.get_clock().now().to_msg()
        ros_path.header.frame_id = 'map'

        for (mx, my) in path:
            wx, wy = self.map_to_world(mx, my)
            pose = PoseStamped()
            pose.header = ros_path.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            ros_path.poses.append(pose)

        return ros_path

    def run(self):
        self.get_logger().info("Waiting for both start and goal poses...")
        while rclpy.ok() and (self.ttbot_pose is None or self.goal_pose is None):
            self.get_logger().info("Waiting for poses...")
            rclpy.spin_once(self, timeout_sec=0.5)

        self.get_logger().info("Both poses received! Ready to plan a path.")

        start_x = self.ttbot_pose.position.x
        start_y = self.ttbot_pose.position.y
        goal_x = self.goal_pose.pose.position.x
        goal_y = self.goal_pose.pose.position.y

        start_map = self.world_to_map(start_x, start_y)
        goal_map = self.world_to_map(goal_x, goal_y)

        self.get_logger().info(f"Start (world): ({start_x}, {start_y}) -> Start (map): {start_map}")
        self.get_logger().info(f"Goal (world): ({goal_x}, {goal_y}) -> Goal (map): {goal_map}")

        # Optional visualization
        self.visualize_initial_positions(start_map, goal_map)

        grid_height, grid_width = self.occupancy_grid.shape
        if not (0 <= start_map[0] < grid_width and 0 <= start_map[1] < grid_height):
            self.get_logger().error("Start position is out of occupancy grid bounds!")
            return
        if not (0 <= goal_map[0] < grid_width and 0 <= goal_map[1] < grid_height):
            self.get_logger().error("Goal position is out of occupancy grid bounds!")
            return

        try:
            start_occupied = self.occupancy_grid[start_map[1], start_map[0]]
            goal_occupied = self.occupancy_grid[goal_map[1], goal_map[0]]
        except IndexError:
            self.get_logger().error("Start or goal position indexing failed!")
            return

        self.get_logger().info(f"Start occupancy: {start_occupied}, Goal occupancy: {goal_occupied}")

        if start_occupied == 0:
            self.get_logger().error("Start position is occupied!")
            return
        if goal_occupied == 0:
            self.get_logger().error("Goal position is occupied!")
            return

        # Compute path using A*
        path = self.a_star_search(self.occupancy_grid, start_map, goal_map)
        self.get_logger().info(f"Path length: {len(path)}")

        if path:
            # Optional visualization of path
            self.visualize_path(self.occupancy_grid, path, start_map, goal_map)

            # Publish the full global plan as a ROS path
            ros_full_path = self.path_to_ros_path(path)
            self.path_pub.publish(ros_full_path)
            self.get_logger().info("Published the full global plan.")

            # Generate waypoints from the path
            waypoints = self.create_waypoints(path, step=5)  

            # Publish the waypoint-based global plan
            ros_waypoints = self.waypoints_to_ros_path(waypoints)
            self.path_pub.publish(ros_waypoints)
            self.get_logger().info("Published the waypoint-based global plan.")

            # Follow the computed waypoints
            self.follow_waypoints(waypoints)
        else:
            self.get_logger().info("No valid path found.")

        # Keep node alive if needed
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            self.rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    nav = Navigation()

    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()