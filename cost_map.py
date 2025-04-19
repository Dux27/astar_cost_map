import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
import math
import numpy as np
import os
import csv
import time  # Added for timing the execution

GRID_SIZE = 0.1            # Size of each grid cell
DISTANCE_RATE = 2.5        # Rate at which distance affects cost
MAX_COST = 5.0              # Maximum cost value
MAX_OBSTACLE_DISTANCE = 3.0 # Maximum distance to consider an obstacle (adjust this as needed)

def load_obstacles(file_path):
    """Load obstacle data from a CSV file"""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        exit(1)
    
    obstacles = []
    with open(file_path, mode="r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            try:
                x = float(row[0])
                y = float(row[1])
                obstacle_type = row[2]
                obstacles.append([x, y, obstacle_type])
            except ValueError as e:
                print(f"Skipping invalid row: {row}, Error: {e}")
    return obstacles

def compute_convex_hull(coordinates):
    """Compute the convex hull for the given set of coordinates"""
    hull = ConvexHull(coordinates)
    hull_points = coordinates[hull.vertices]
    return hull, hull_points

def generate_grid(hull_points, grid_size):
    """Generate grid points inside the convex hull"""
    min_x, min_y = np.min(hull_points, axis=0)
    max_x, max_y = np.max(hull_points, axis=0)
    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(min_y, max_y + grid_size, grid_size)
    xv, yv = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])
    path = Path(hull_points)
    return grid_points[path.contains_points(grid_points)]

def calculate_cost(distance):
    return max(0.0, MAX_COST - (distance * DISTANCE_RATE))  

def add_costs(obstacles, points_inside_convex):
    # Initialize the points with costs
    points_with_cost = np.zeros((points_inside_convex.shape[0], 3))  
    points_with_cost[:, :2] = points_inside_convex  
    
    obstacle_positions = np.array([(obstacle[0], obstacle[1]) for obstacle in obstacles])

    # Calculate the distance from each point to nearby obstacles and apply cost
    for i in range(len(points_with_cost)):
        # Find obstacles within a certain distance range
        distances = np.linalg.norm(obstacle_positions - points_with_cost[i, :2], axis=1)
        nearby_obstacles = distances[distances <= MAX_OBSTACLE_DISTANCE]  # Only consider obstacles within range
        if nearby_obstacles.size > 0:
            min_distance = np.min(nearby_obstacles)
            cost = calculate_cost(min_distance)
            points_with_cost[i, 2] = cost

    return points_with_cost

def plot_obstacles_and_hull(obstacles, coordinates, hull, points_with_cost):
    """Plot the obstacles, convex hull, and the cost map with gradient color scale"""
    fig, ax = plt.subplots()  # Create figure and axes objects

    # Plot the convex hull boundaries
    for edges in hull.simplices:
        ax.plot(coordinates[edges, 0], coordinates[edges, 1], color='black', linewidth=1)

    # Normalize the cost values to fit within the 0 to 1 range
    norm = plt.Normalize(vmin=0, vmax=MAX_COST) 

    # Create a colormap from dark blue to light grey
    cmap = LinearSegmentedColormap.from_list("cyan_orange", ["cyan", "black"])

    # Plot points inside convex hull with color gradient based on cost values
    for point in points_with_cost:
        x, y, cost = point
        color = cmap(norm(cost))  # RdYlGn_r colormap (red to green)
        ax.scatter(x, y, color=color, s=5)

    # Plot obstacles as blue dots
    for obstacle in obstacles:
        if obstacle[2] == "red_buoy":
            ax.plot(obstacle[0], obstacle[1], 'ro', label='Obstacle')
        elif obstacle[2] == "green_buoy":
            ax.plot(obstacle[0], obstacle[1], 'go', label='Obstacle')
        elif obstacle[2] == "yellow_buoy":
            ax.plot(obstacle[0], obstacle[1], 'yo', label='Obstacle')

    # Colorbar setup
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Set an empty array to the ScalarMappable
    fig.colorbar(sm, ax=ax, label='Cost')

    end_time = time.time()  # End timing
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Cost Map')
    plt.show()

if __name__ == "__main__":
    start_time = time.time()  # Start timing

    main_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(main_dir, "config", "obstacles.csv")
    
    obstacles = load_obstacles(file_path)
    coordinates = np.array([[x, y] for x, y, _ in obstacles])
    hull, hull_points = compute_convex_hull(coordinates)
    points_inside_convex = generate_grid(hull_points, GRID_SIZE)
    points_with_cost = add_costs(obstacles, points_inside_convex)

    plot_obstacles_and_hull(obstacles, coordinates, hull, points_with_cost)
