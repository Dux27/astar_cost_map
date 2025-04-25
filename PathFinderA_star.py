import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from CostMap import Start_point, Goal_point, points_with_cost, GRID_SIZE, MAX_COST
import time

### CONSTANTS
MAX_STEP_SIZE = 1.0 # Maximum step size for the pathfinding algorithm

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def find_nearest_point(point, points_with_cost):
    """Find the nearest point in the cost points array for start and finish point to be on the grid"""
    distances = np.linalg.norm(points_with_cost[:, :2] - point, axis=1)
    nearest_index = np.argmin(distances)
    return points_with_cost[nearest_index, :2]

def a_star(start, goal, points_with_cost, max_distance):
    # Define open and closed lists (priority queue)
    open_list = []
    closed_list = set()

    # The dictionary to store the path
    came_from = {}

    # The dictionary to store the cost of each point
    g_score = {tuple(point[:2]): float('inf') for point in points_with_cost}
    g_score[tuple(start)] = 0

    # Heuristic cost from start to goal
    f_score = {tuple(point[:2]): float('inf') for point in points_with_cost}
    f_score[tuple(start)] = euclidean_distance(start, goal)

    # Add the start point to the open list
    heapq.heappush(open_list, (f_score[tuple(start)], tuple(start)))

    while open_list:
        # Get the point in open list with the lowest f score
        _, current = heapq.heappop(open_list)

        # If the goal is reached, reconstruct the path
        if np.array_equal(current, goal):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path
        
        closed_list.add(tuple(current))

        # Check neighbors (all points in points_with_cost)
        for neighbor in points_with_cost[:, :2]:
            neighbor_tuple = tuple(neighbor)  # Convert neighbor to tuple
            
            # Skip already visited points and those that are too far away
            if neighbor_tuple in closed_list or euclidean_distance(current, neighbor) > max_distance:
                continue  # Ignore already visited points or those outside the max distance range
            
            # Find the full cost point, which includes [x, y, cost]
            neighbor_row = points_with_cost[np.all(points_with_cost[:, :2] == neighbor, axis=1)][0]
            neighbor_cost = neighbor_row[2]  # Cost is in the third column
            
            # Calculate tentative g score for the neighbor
            tentative_g_score = g_score[tuple(current)] + euclidean_distance(current, neighbor)

            if tentative_g_score < g_score[neighbor_tuple]:
                # This is a better path to the neighbor
                came_from[neighbor_tuple] = current
                g_score[neighbor_tuple] = tentative_g_score
                f_score[neighbor_tuple] = g_score[neighbor_tuple] + euclidean_distance(neighbor, goal) + neighbor_cost
                
                # Add the neighbor to the open list if not already in open list
                if not any(neighbor_tuple == n[1] for n in open_list):  # Avoid duplicates
                    heapq.heappush(open_list, (f_score[neighbor_tuple], neighbor_tuple))

    return None  # No path found

def plot_cost_map_with_path(points_with_cost, path=None):
    # Plot the path
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='red', label='Path')
    else:
        print("No path found.")
    
    # Setup color map
    cmap = LinearSegmentedColormap.from_list("cyan_black", ["cyan", "black"])
    # Normalize the cost values to fit within the 0 to 1 range
    norm = plt.Normalize(vmin=0, vmax=MAX_COST) 
    
    # Plot colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  
    ax = plt.gca()  
    cbar = plt.colorbar(sm, ax=ax, label='Cost', orientation='vertical')
    cbar.set_ticks([0, MAX_COST])  # Set ticks at the minimum and maximum values
    cbar.set_ticklabels(['Low', 'High'])  # Label the ticks as "Low" and "High"

    SCALE_FACTOR = 50
    marker_size = int(5 + (GRID_SIZE * SCALE_FACTOR))

    # Plot the cost map
    plt.scatter(points_with_cost[:, 0], points_with_cost[:, 1], c=points_with_cost[:, 2], cmap=cmap, label='Cost Points', s=marker_size)
    plt.scatter(*start_nearest, color='red', label='Start')
    plt.scatter(*goal_nearest, color='red', label='Goal')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Cost Map with A* Path')
    plt.show()
    
if __name__ == "__main__":
    start_time = time.time()  # Start timing for execution time measurement
    
    if MAX_STEP_SIZE <= GRID_SIZE:
        raise ValueError("MAX_STEP_SIZE must be greater than GRID_SIZE to ensure proper pathfinding.")

    print(f"Start: [{Start_point[0]}, {Start_point[1]}], Goal: [{Goal_point[0]}, {Goal_point[1]}]")
    print("Running A* algorithm...")    
    
    # Find nearest points in the cost points array for start and goal
    start_nearest = find_nearest_point(Start_point, points_with_cost)
    goal_nearest = find_nearest_point(Goal_point, points_with_cost)
    print(f"Start point adjusted to: {start_nearest}")
    print(f"Goal point adjusted to: {goal_nearest}")

    path = a_star(start_nearest, goal_nearest, points_with_cost, MAX_STEP_SIZE)
    
    end_time = time.time()  # End timing for execution time measurement
    print("A* algorithm completed.")
    print(f"Path finding execution time: {end_time - start_time:.3f} seconds")
    
    plot_cost_map_with_path(points_with_cost, path)
