import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import heapq  # Import heapq for A* algorithm

# Define the cost map (as you provided earlier)
grid_size = 100
cost_map = np.random.randint(1, 10, size=(grid_size, grid_size))  # Cost between 1 and 10

# Start and goal points
start = (5, 5)
goal = (95, 95)

# A* Pathfinding class (Modified to work with cost_map)
class AStarPathfinding:
    def __init__(self, cost_map, start, goal, step_size):
        self.cost_map = cost_map  # The cost map (a 2D array)
        self.start = start  # Start point
        self.goal = goal  # Goal point
        self.step_size = step_size  # Maximum distance between consecutive points

    def heuristic(self, point):
        """Estimate the cost to reach the goal from the current point (Euclidean distance)"""
        return np.linalg.norm(np.array(point) - np.array(self.goal))

    def get_neighbors(self, current_point):
        """Find the valid neighbors from the current point (points within step size)"""
        neighbors = []
        x, y = current_point
        # Check adjacent points within step size range (4 possible neighbors for grid movement)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(self.cost_map) and 0 <= ny < len(self.cost_map[0]):
                neighbors.append((nx, ny))
        return neighbors

    def plan(self):
        """Run A* algorithm to find the shortest path from start to goal"""
        open_list = []
        heapq.heappush(open_list, (0 + self.heuristic(self.start), 0, self.start, []))  # (f, g, point, path)

        closed_list = set()

        while open_list:
            _, g, current_point, path = heapq.heappop(open_list)

            # If we've already processed this point, continue
            if current_point in closed_list:
                continue
            closed_list.add(current_point)

            # Add current point to the path
            path = path + [current_point]

            # If we reached the goal, return the path
            if current_point == self.goal:
                return path, g

            # Explore neighbors
            for neighbor in self.get_neighbors(current_point):
                if neighbor not in closed_list:
                    cost = self.cost_map[neighbor[1], neighbor[0]]  # Get the cost of the neighbor
                    heapq.heappush(open_list, (
                        g + cost + self.heuristic(neighbor), g + cost, neighbor, path
                    ))

        return None, float('inf')  # No path found

# Initialize A* pathfinder
step_size = 1  # Step size between consecutive points
astar = AStarPathfinding(cost_map=cost_map, start=start, goal=goal, step_size=step_size)

# Plan the path using A*
path, total_cost = astar.plan()

if path:
    print(f"Path found: {path}")
    print(f"Total cost: {total_cost}")
else:
    print("No path found.")

# Visualize the cost map and the planned path
plt.imshow(cost_map, cmap='Greys', origin='lower')
if path:
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, color='blue', linewidth=2)
plt.scatter(*start, color='green', label='Start')
plt.scatter(*goal, color='red', label='Goal')
plt.legend()
plt.show()
