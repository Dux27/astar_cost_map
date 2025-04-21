import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the RRT module directory to the Python path
repo_path = r"C:\Users\filip\Desktop\infinity\PythonRobotics\PathPlanning\RRT"
sys.path.append(repo_path)

# Import the RRT module
from rrt import RRT

# Define the cost map
grid_size = 100
cost_map = np.random.randint(1, 10, size=(grid_size, grid_size))  # Cost between 1 and 10

# Start and goal points
start = (5, 5)
goal = (95, 95)

# Cost-aware RRT extension
class CostAwareRRT(RRT):
    def __init__(self, cost_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_map = cost_map

    def get_cost(self, point):
        x, y = point
        return self.cost_map[int(y)][int(x)]

    def planning(self, animation=False):
        path = super().planning(animation)
        if path:
            total_cost = sum(self.get_cost(p) for p in path)
            print(f"Total path cost: {total_cost}")
        return path

# Initialize RRT with cost-awareness
rrt = CostAwareRRT(
    cost_map=cost_map,
    start=start,
    goal=goal,
    rand_area=[0, 100],
    obstacle_list=[],  # Not needed for cost maps
    max_iter=300
)

# Plan the path
path = rrt.planning(animation=True)

# Visualize the cost map and path
plt.imshow(cost_map, cmap='Greys', origin='lower')
if path:
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, color='blue', linewidth=2)
plt.scatter(*start, color='green', label='Start')
plt.scatter(*goal, color='red', label='Goal')
plt.legend()
plt.show()
