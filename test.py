import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

# Example data points
red_class = np.array([(1, 1), (3, 3), (2, 5)])  # Red buoys
green_class = np.array([(4, 2), (6, 4), (5, 6)])  # Green buoys

# Reference point (e.g., channel entrance)
reference_point = np.array([0, 0])

# Function to find the nearest neighbor path
def nearest_neighbor_path(points, start_point):
    points = points.copy()
    path = [start_point]
    while len(points) > 0:
        distances = cdist([path[-1]], points).flatten()
        nearest_idx = distances.argmin()
        path.append(points[nearest_idx])
        points = np.delete(points, nearest_idx, axis=0)
    return np.array(path)

# Find the starting buoy for each class (nearest to reference point)
red_start_idx = cdist([reference_point], red_class).argmin()
green_start_idx = cdist([reference_point], green_class).argmin()

red_start = red_class[red_start_idx]
green_start = green_class[green_start_idx]

# Remove the starting points from the datasets
red_class = np.delete(red_class, red_start_idx, axis=0)
green_class = np.delete(green_class, green_start_idx, axis=0)

# Build paths for each class
red_path = nearest_neighbor_path(red_class, red_start)
green_path = nearest_neighbor_path(green_class, green_start)

# Plot the points
plt.scatter(red_path[:, 0], red_path[:, 1], color='red', label='Red Buoys')
plt.scatter(green_path[:, 0], green_path[:, 1], color='green', label='Green Buoys')

# Plot the paths
plt.plot(red_path[:, 0], red_path[:, 1], color='red', linestyle='-', label='Red Path')
plt.plot(green_path[:, 0], green_path[:, 1], color='green', linestyle='-', label='Green Path')

# Show the graph
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Navigation Channel Paths')
plt.legend()
plt.grid(True)
plt.show()
