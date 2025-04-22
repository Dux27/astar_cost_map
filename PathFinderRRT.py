from CostMap import Start_point, Finish_point, points_with_cost
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add the RRT module directory to the Python path
repo_path = r"C:\Users\filip\Desktop\infinity\PythonRobotics\PathPlanning\RRT"
sys.path.append(repo_path)

from rrt import RRT

print("Generating path...")

start = Start_point
goal = Finish_point

class ConstantDistanceRRT(RRT):
    def __init__(self, cost_points, step_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_points = points_with_cost 
        self.step_size = step_size 

    def get_cost(self, point):
        """Returns the cost of a given point by finding the closest cost point"""
        distances = np.linalg.norm(self.cost_points[:, :2] - point, axis=1)
        nearest_index = np.argmin(distances)
        return self.cost_points[nearest_index, 2]  # The cost is stored in the third column

    def is_valid(self, point):
        """Checks if the point is valid by making sure it exists in the cost_points"""
        distances = np.linalg.norm(self.cost_points[:, :2] - point, axis=1)
        return np.any(distances < 0.5)  # Tolerance for close points

    def generate_random_point(self):
        """Generate a random point from the given cost points"""
        return self.cost_points[np.random.randint(len(self.cost_points))]

    def planning(self, animation=False):
        path = super().planning(animation)
        if path:
            # Enforce constant distance between waypoints
            path = self.enforce_constant_distance(path)
            total_cost = sum(self.get_cost(p) for p in path)
            print(f"Total path cost: {total_cost}")
        return path

    def enforce_constant_distance(self, path):
        """Ensure that the distance between consecutive points is constant"""
        new_path = [path[0]]  # Start with the first point
        for i in range(1, len(path)):
            current_point = np.array(new_path[-1])  # Ensure this is a numpy array
            next_point = np.array(path[i])  # Ensure this is a numpy array
            distance = np.linalg.norm(current_point - next_point)

            # If the distance is too large, break it down into smaller segments with constant step size
            while distance > self.step_size:
                num_intermediate_points = int(np.floor(distance / self.step_size))
                for j in range(1, num_intermediate_points + 1):
                    interp_point = current_point + (next_point - current_point) * (j / (num_intermediate_points + 1))
                    new_path.append(interp_point)
                # Recalculate distance with the new intermediate point added
                current_point = new_path[-1]
                distance = np.linalg.norm(current_point - next_point)
            # Add the next point to the path if it's close enough
            new_path.append(next_point)
        return np.array(new_path)

    def expand(self, current_node):
        """Override to limit expansion to points in cost_points"""
        # Get possible next points from cost_points
        distances = np.linalg.norm(self.cost_points[:, :2] - current_node, axis=1)
        closest_idx = np.argmin(distances)
        next_point = self.cost_points[closest_idx, :2]
        return next_point

# Example Usage
if __name__ == "__main__":
    # Example: Your previous setup for points_with_cost, start, goal, etc.
    # Example points_with_cost (2D array with [x, y, cost])
    cost_points = np.array(points_with_cost)  # Assuming points_with_cost is already defined

    start = Start_point
    goal = Finish_point

    step_size = 2.0  # Set the constant distance between points (you can change this value)

    # Initialize the RRT with constant distance between waypoints
    rrt = ConstantDistanceRRT(
        cost_points=cost_points,
        step_size=step_size,
        start=start,
        goal=goal,
        rand_area=[0, 10],
        obstacle_list=[],  # Not needed as we're using cost points
        max_iter=300
    )

    # Plan the path
    path = rrt.planning(animation=False)

    # Visualize the path on the cost map
    plt.scatter(cost_points[:, 0], cost_points[:, 1], c=cost_points[:, 2], cmap="viridis", label='Cost Points')
    if path is not None:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='blue', label='Path')
    plt.scatter(*start, color='green', label='Start')
    plt.scatter(*goal, color='red', label='Goal')
    plt.legend()
    plt.show()