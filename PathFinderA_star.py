import numpy as np
import heapq
import matplotlib.pyplot as plt
from CostMap import Start_point, Finish_point, points_with_cost  # Importing the cost map, start, and finish points

# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Find the nearest point in the cost points array
def find_nearest_point(point, points_with_cost):
    distances = np.linalg.norm(points_with_cost[:, :2] - point, axis=1)
    nearest_index = np.argmin(distances)
    return points_with_cost[nearest_index, :2]  # Return the nearest [x, y] coordinates

# Function to adjust the cost: Boost the obstacles by multiplying the cost by a large factor
def adjust_cost(cost, obstacle_threshold=10, boost_factor=1000):
    if cost > obstacle_threshold:  # If the cost is above a certain threshold, treat it as an obstacle
        return boost_factor * cost  # Significantly increase the cost for obstacles
    return cost  # Return the original cost if it's a free space

# A* algorithm to find a path with maximum distance constraint
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
            
            # Adjust the cost to boost obstacles
            adjusted_cost = adjust_cost(neighbor_cost)
            
            # Calculate tentative g score for the neighbor
            tentative_g_score = g_score[tuple(current)] + euclidean_distance(current, neighbor)

            if tentative_g_score < g_score[neighbor_tuple]:
                # This is a better path to the neighbor
                came_from[neighbor_tuple] = current
                g_score[neighbor_tuple] = tentative_g_score
                f_score[neighbor_tuple] = g_score[neighbor_tuple] + euclidean_distance(neighbor, goal) + adjusted_cost
                
                # Add the neighbor to the open list if not already in open list
                if not any(neighbor_tuple == n[1] for n in open_list):  # Avoid duplicates
                    heapq.heappush(open_list, (f_score[neighbor_tuple], neighbor_tuple))

    return None  # Return None if no path is found

# Example usage of the A* algorithm
if __name__ == "__main__":
    # Use the imported points_with_cost, Start_point, and Finish_point from the CostMap
    start = Start_point
    goal = Finish_point

    print(f"Start: {start}, Goal: {goal}")
    print("Running A* algorithm...")

    # Find the nearest point in points_with_cost for the start and goal
    start_nearest = find_nearest_point(start, points_with_cost)
    goal_nearest = find_nearest_point(goal, points_with_cost)

    print(f"Start point adjusted to: {start_nearest}")
    print(f"Goal point adjusted to: {goal_nearest}")

    # Define the maximum allowed distance between consecutive nodes
    max_distance = 1.0  # Adjust this value based on your requirements

    # Find the path using A* algorithm
    path = a_star(start_nearest, goal_nearest, points_with_cost, max_distance)

    # Visualize the path
    if path:
        print(f"Path found: {path}")
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='blue', label='Path')
    else:
        print("No path found.")

    # Visualize all points and the start/goal positions
    plt.scatter(points_with_cost[:, 0], points_with_cost[:, 1], c=points_with_cost[:, 2], cmap="viridis", label='Cost Points')
    plt.scatter(*start_nearest, color='green', label='Start')
    plt.scatter(*goal_nearest, color='red', label='Goal')
    plt.legend()
    plt.show()
