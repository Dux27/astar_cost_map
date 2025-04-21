import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import csv
import time  

GRID_SIZE = 0.2             # Size of each grid cell
DISTANCE_RATE = 2.5         # Rate at which distance affects cost
MAX_COST = 5.0              # Maximum cost value
MAX_OBSTACLE_DISTANCE = 4.0 # Maximum distance to consider an obstacle in cost calculation

"""Start point is where Task2 starts, and finish point is dynamic and is calculated based on the last red and green buoys detected"""
Start_point = np.array([0, 0]) # Point where Task2 starts (hopefully I guess)
Finish_point = np.array([0, 0]) # Point where Task2 ends (probably I guess)

def load_obstacles(file_path):
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
    
    # Create a grid of points within the bounding box of the convex hull
    x_coords = np.arange(min_x, max_x + grid_size, grid_size)
    y_coords = np.arange(min_y, max_y + grid_size, grid_size)
    
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])
    
    path = Path(hull_points)
    return grid_points[path.contains_points(grid_points)]

def calculate_cost(distance):
    return max(0.0, MAX_COST - (distance * DISTANCE_RATE))  

def add_costs(obstacles, points_inside_convex):
    """ Initialize the points with costs"""
    points_with_cost = np.zeros((points_inside_convex.shape[0], 3))  
    points_with_cost[:, :2] = points_inside_convex  
    
    obstacle_positions = np.array([(obstacle[0], obstacle[1]) for obstacle in obstacles])

    # Calculate the distance from each point to nearby obstacles and apply cost
    for i in range(len(points_with_cost)):
        # Create a list of distances to all obstacles
        dx = obstacle_positions[:, 0] - points_with_cost[i, 0]
        dy = obstacle_positions[:, 1] - points_with_cost[i, 1]
        distances = np.sqrt(dx**2 + dy**2) # Euclidean distance between gridpoints and obstacles
        
        nearby_obstacles = distances[distances <= MAX_OBSTACLE_DISTANCE] # Filter distances to nearby obstacles
        if nearby_obstacles.size > 0:
            min_distance = np.min(nearby_obstacles)
            cost = calculate_cost(min_distance)
            points_with_cost[i, 2] = cost

    return points_with_cost

def nearest_neighbor_path(points, start_point):
    """Create a path using the nearest neighbor algorithm (cdist)"""
    points = points.copy()
    path = [start_point]
    while len(points) > 0:
        distances = cdist([path[-1]], points).flatten()
        nearest_idx = distances.argmin()
        path.append(points[nearest_idx])
        points = np.delete(points, nearest_idx, axis=0)
    return np.array(path)

def create_path_from_buoys(obstacles, buoy_type: str):
    """Create a path from buoys of a specific type"""
    buoy_class = np.array([(obstacle[0], obstacle[1]) for obstacle in obstacles if obstacle[2] == buoy_type])
    # Find the starting buoy for each class (nearest to reference point)
    buoy_start_idx = cdist([Start_point], buoy_class).argmin()
    buoy_start = buoy_class[buoy_start_idx]
    # Remove the starting points from the datasets
    buoy_class = np.delete(buoy_class, buoy_start_idx, axis=0)
    # Build paths for each class
    buoy_path = nearest_neighbor_path(buoy_class, buoy_start)
    
    return buoy_path

def find_finish_gate():
    red_path = create_path_from_buoys(obstacles, "red_buoy")
    green_path = create_path_from_buoys(obstacles, "green_buoy")
    last_red = red_path[-1]
    last_green = green_path[-1]
    
    x = (last_red[0] + last_green[0])/2.0
    y = (last_red[1] + last_green[1])/2.0
    
    return x, y

def plot_obstacles_and_hull(obstacles, coordinates, hull, points_with_cost):
    """Plot the obstacles, convex hull, and the cost map with gradient color scale"""
    fig, ax = plt.subplots()  
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
        color = cmap(norm(cost)) # Convert cost to color
        SCALE_FACTOR = 50
        marker_size = int(5 + (GRID_SIZE * SCALE_FACTOR))
        ax.scatter(x, y, color=color, s=marker_size)
    # Plot obstacles with different colors based on their type
    for obstacle in obstacles:
        if obstacle[2] == "red_buoy":
            ax.plot(obstacle[0], obstacle[1], 'ro', label='Obstacle')
        elif obstacle[2] == "green_buoy":
            ax.plot(obstacle[0], obstacle[1], 'go', label='Obstacle')
        elif obstacle[2] == "yellow_buoy":
            ax.plot(obstacle[0], obstacle[1], 'yo', label='Obstacle')
    # Colorbar setup
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  
    fig.colorbar(sm, ax=ax, label='Cost')
    # Plot connection lines between red and green buoys
    red_path = create_path_from_buoys(obstacles, "red_buoy")
    green_path = create_path_from_buoys(obstacles, "green_buoy")
    ax.plot(red_path[:, 0], red_path[:, 1], color='red', linestyle='--', label='Red Path', linewidth=2)
    ax.plot(green_path[:, 0], green_path[:, 1], color='green', linestyle='--', label='Green Path', linewidth=2)
    # Plot finish gate
    finish_x, finish_y = find_finish_gate()
    Finish_point = np.array([finish_x, finish_y]) # Global variable assignment
    ax.plot(finish_x, finish_y, 'bo', label='Finish Gate', markersize=10)

    end_time = time.time()  # End timing for execution time measurement
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Cost Map')
    plt.show()

if __name__ == "__main__":
    start_time = time.time()  # Start timing for execution time measurement

    main_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(main_dir, "config", "obstacles_2.csv")
    
    obstacles = load_obstacles(file_path)
    coordinates = np.array([[x, y] for x, y, _ in obstacles])
    hull, hull_points = compute_convex_hull(coordinates)
    points_inside_convex = generate_grid(hull_points, GRID_SIZE)
    points_with_cost = add_costs(obstacles, points_inside_convex)

    plot_obstacles_and_hull(obstacles, coordinates, hull, points_with_cost)
