"""
Path Planning Visualizer

An interactive Python application using Tkinter to demonstrate and compare
two fundamental path planning algorithms based on space decomposition:
1.  Uniform Grid (Homogeneous Decomposition)
2.  Quadtree (Adaptive Decomposition)

Users can interactively create maps with obstacles (rectangles, circles, triangles),
set start and goal positions, and visualize the path found by the A* algorithm
on the resulting graph.

The application uses multithreading to perform heavy computations (graph generation
and A* search) in the background, preventing the GUI from freezing.
"""

import math
import heapq
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import random
import threading
import queue

# --- 1. Global Constants ---
MAP_WIDTH = 800
MAP_HEIGHT = 600
MAP_BOUNDS = (0, 0, MAP_WIDTH, MAP_HEIGHT)
DEFAULT_RECT_SIZE = 40
DEFAULT_CIRCLE_RADIUS = 20
DEFAULT_TRIANGLE_SIZE = 40


# --- 2. A* (A-star) Pathfinding Algorithm ---

def heuristic(a, b):
    """
    Calculate the heuristic (Euclidean distance) between two points.
    Used by A* to estimate the cost to reach the goal.
    Args:
        a (tuple): The first point (x, y).
        b (tuple): The second point (x, y).
    Returns:
        float: The Euclidean distance.
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def a_star_search(graph, start, goal):
    """
    Finds the shortest path from start to goal in a graph using the A* algorithm.
    Args:
        graph (dict): An adjacency list representation of the graph.
                      {node: [neighbor1, neighbor2, ...]}
        start (tuple): The starting node (x, y).
        goal (tuple): The goal node (x, y).
    Returns:
        list: A list of nodes (tuples) representing the path from start to goal,
              or None if no path is found.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))  # Priority queue (f_score, node)

    came_from = {}  # Stores the previous node in the optimal path

    # g_score: Cost from start to the current node
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    # f_score: Estimated total cost (g_score + heuristic)
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Path found, reconstruct it
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Return reversed path

        for neighbor in graph[current]:
            # Calculate tentative g_score for this path
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if tentative_g_score < g_score[neighbor]:
                # This path to the neighbor is better than the previous one
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if (f_score[neighbor], neighbor) not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found


# --- 3. Geometry Helpers and Obstacle Classes ---

def _line_intersects_line(p1, p2, p3, p4):
    """Checks if line segment (p1, p2) intersects line segment (p3, p4)."""

    def on_segment(p, q, r):
        # Check if point q lies on segment pr
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    def orientation(p, q, r):
        # Find orientation of ordered triplet (p, q, r)
        val = ((q[1] - p[1]) * (r[0] - q[0]) -
               (q[0] - p[0]) * (r[1] - q[1]))
        if val == 0: return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    # Find orientations
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases (collinear)
    if o1 == 0 and on_segment(p1, p3, p2): return True
    if o2 == 0 and on_segment(p1, p4, p2): return True
    if o3 == 0 and on_segment(p3, p1, p4): return True
    if o4 == 0 and on_segment(p3, p2, p4): return True

    return False


class Rectangle:
    """Represents a rectangular obstacle."""

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def collides_with_area(self, bounds):
        """Checks if the rectangle (self) intersects with a given bounding box (bounds)."""
        x_min, y_min, x_max, y_max = bounds
        # Standard AABB (Axis-Aligned Bounding Box) collision check
        return (self.x < x_max and self.x + self.width > x_min and
                self.y < y_max and self.y + self.height > y_min)

    def is_clicked(self, x, y):
        """Checks if a point (x, y) is inside the rectangle."""
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def draw(self, canvas):
        """Draws the rectangle on the Tkinter canvas."""
        canvas.create_rectangle(self.x, self.y, self.x + self.width, self.y + self.height,
                                fill='gray', outline='black')


class Circle:
    """Represents a circular obstacle."""

    def __init__(self, x, y, radius):
        self.center_x = x
        self.center_y = y
        self.radius = radius

    def collides_with_area(self, bounds):
        """Checks if the circle intersects with a given bounding box (bounds)."""
        x_min, y_min, x_max, y_max = bounds

        # Find the closest point on the bounding box to the circle's center
        closest_x = max(x_min, min(self.center_x, x_max))
        closest_y = max(y_min, min(self.center_y, y_max))

        # Calculate the distance from the center to this closest point
        distance_sq = (self.center_x - closest_x) ** 2 + (self.center_y - closest_y) ** 2

        # If the distance is less than the radius squared, they intersect
        return distance_sq < (self.radius ** 2)

    def is_clicked(self, x, y):
        """Checks if a point (x, y) is inside the circle."""
        distance_sq = (x - self.center_x) ** 2 + (y - self.center_y) ** 2
        return distance_sq <= self.radius ** 2

    def draw(self, canvas):
        """Draws the circle on the Tkinter canvas."""
        canvas.create_oval(self.center_x - self.radius, self.center_y - self.radius,
                           self.center_x + self.radius, self.center_y + self.radius,
                           fill='gray', outline='black')


class Triangle:
    """Represents a triangular obstacle."""

    def __init__(self, p1, p2, p3):
        self.p1 = p1  # Vertex 1 (x, y)
        self.p2 = p2  # Vertex 2 (x, y)
        self.p3 = p3  # Vertex 3 (x, y)

    def _sign(self, p1, p2, p3):
        """Helper function to determine orientation using cross-product."""
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    def _point_in_triangle(self, px, py):
        """Checks if a point (px, py) is inside the triangle using barycentric coordinates."""
        p = (px, py)
        d1 = self._sign(p, self.p1, self.p2)
        d2 = self._sign(p, self.p2, self.p3)
        d3 = self._sign(p, self.p3, self.p1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        # The point is inside if all signs are the same (all positive or all negative)
        return not (has_neg and has_pos)

    def is_clicked(self, x, y):
        """Checks if a point (x, y) is inside the triangle."""
        return self._point_in_triangle(x, y)

    def collides_with_area(self, bounds):
        """Complex AABB vs. Triangle collision check."""
        x_min, y_min, x_max, y_max = bounds

        # 1. Check if any triangle vertex is inside the bounds
        for p in [self.p1, self.p2, self.p3]:
            if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max:
                return True

        # 2. Check if any bounds corner is inside the triangle
        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        for c in corners:
            if self._point_in_triangle(c[0], c[1]):
                return True

        # 3. Check if any triangle edge intersects any bounds edge
        tri_edges = [(self.p1, self.p2), (self.p2, self.p3), (self.p3, self.p1)]
        box_edges = [(corners[0], corners[1]), (corners[1], corners[2]),
                     (corners[2], corners[3]), (corners[3], corners[0])]

        for te in tri_edges:
            for be in box_edges:
                if _line_intersects_line(te[0], te[1], be[0], be[1]):
                    return True

        return False

    def draw(self, canvas):
        """Draws the triangle on the Tkinter canvas."""
        canvas.create_polygon(self.p1, self.p2, self.p3,
                              fill='gray', outline='black')


# --- 4. Graph Generation Logic (Grid & Quadtree) ---

def create_grid_graph(obstacles, map_bounds, resolution):
    """
    Creates a graph based on a uniform grid decomposition.

    Args:
        obstacles (list): List of obstacle objects.
        map_bounds (tuple): (min_x, min_y, max_x, max_y) of the map.
        resolution (int): The number of cells in one dimension (e.g., 20 for 20x20).

    Returns:
        tuple: (graph, grid_nodes, cell_width, cell_height)
               - graph: Adjacency list for A*.
               - grid_nodes: 2D list of cell centers.
               - cell_width, cell_height: Dimensions of a single cell.
    """
    graph = {}
    grid_nodes = [[None for _ in range(resolution)] for _ in range(resolution)]
    min_x, min_y, max_x, max_y = map_bounds
    cell_width = (max_x - min_x) / resolution
    cell_height = (max_y - min_y) / resolution
    free_cells = []

    # Step 1: Classify all cells as FREE or OCCUPIED
    for i in range(resolution):
        for j in range(resolution):
            x_min = min_x + i * cell_width
            y_min = min_y + j * cell_height
            x_max = x_min + cell_width
            y_max = y_min + cell_height

            cell_bounds = (x_min, y_min, x_max, y_max)
            is_occupied = False
            for obs in obstacles:
                if obs.collides_with_area(cell_bounds):
                    is_occupied = True
                    break

            if not is_occupied:
                cell_center = (x_min + cell_width / 2, y_min + cell_height / 2)
                grid_nodes[i][j] = cell_center
                graph[cell_center] = []
                free_cells.append((i, j, cell_center))

    # Step 2: Connect adjacent FREE cells in the graph
    for i, j, center in free_cells:
        # Check 4 neighbors (up, down, left, right)
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < resolution and 0 <= nj < resolution and grid_nodes[ni][nj]:
                neighbor_center = grid_nodes[ni][nj]
                graph[center].append(neighbor_center)

    return graph, grid_nodes, cell_width, cell_height


def find_grid_node(pos, map_bounds, resolution):
    """Finds the center of the grid cell that contains a given (x, y) position."""
    min_x, min_y, max_x, max_y = map_bounds
    cell_width = (max_x - min_x) / resolution
    cell_height = (max_y - min_y) / resolution

    i = int((pos[0] - min_x) / cell_width)
    j = int((pos[1] - min_y) / cell_height)

    # Clamp to bounds
    i = max(0, min(i, resolution - 1))
    j = max(0, min(j, resolution - 1))

    x_min = min_x + i * cell_width
    y_min = min_y + j * cell_height
    return (x_min + cell_width / 2, y_min + cell_height / 2)


class QuadtreeNode:
    """Represents a node in the Quadtree."""

    def __init__(self, bounds, depth):
        self.bounds = bounds  # (x_min, y_min, x_max, y_max)
        self.depth = depth
        self.children = None  # List of 4 child nodes [NW, NE, SW, SE]
        self.status = 'EMPTY'  # EMPTY, FREE, OCCUPIED, MIXED
        self.center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)

    def subdivide(self):
        """Divides the node into four equal child nodes."""
        x_min, y_min, x_max, y_max = self.bounds
        x_mid = self.center[0]
        y_mid = self.center[1]

        # Define bounds for the 4 children
        nw_bounds = (x_min, y_mid, x_mid, y_max)
        ne_bounds = (x_mid, y_mid, x_max, y_max)
        sw_bounds = (x_min, y_min, x_mid, y_mid)
        se_bounds = (x_mid, y_min, x_max, y_mid)

        self.children = [
            QuadtreeNode(nw_bounds, self.depth + 1),
            QuadtreeNode(ne_bounds, self.depth + 1),
            QuadtreeNode(sw_bounds, self.depth + 1),
            QuadtreeNode(se_bounds, self.depth + 1)
        ]


def check_node_status(node, obstacles):
    """Checks if a node is FREE (no obstacles) or OCCUPIED (one or more obstacles)."""
    for obs in obstacles:
        if obs.collides_with_area(node.bounds):
            return 'OCCUPIED'
    return 'FREE'


def build_quadtree(node, obstacles, max_depth):
    """
    Recursively builds the Quadtree.

    Args:
        node (QuadtreeNode): The current node to process.
        obstacles (list): List of all obstacle objects.
        max_depth (int): The maximum allowed recursion depth.
    """
    node_status = check_node_status(node, obstacles)

    if node_status == 'FREE':
        node.status = 'FREE'
        return

    if node.depth == max_depth:
        # Reached max depth, force it to be a leaf
        # If it has any obstacle, it's OCCUPIED. Otherwise, it's FREE.
        node.status = 'OCCUPIED' if node_status == 'OCCUPIED' else 'FREE'
        return

    # If we are here, the node is not FREE and not at max depth,
    # so it must be MIXED and needs to be subdivided.
    node.status = 'MIXED'
    node.subdivide()
    for child in node.children:
        build_quadtree(child, obstacles, max_depth)


def get_free_leaves(node, free_leaves_list):
    """Recursively traverses the tree and collects all 'FREE' leaf nodes."""
    if node.status == 'FREE':
        free_leaves_list.append(node)
    elif node.status == 'MIXED':
        for child in node.children:
            get_free_leaves(child, free_leaves_list)


def are_adjacent(node1, node2):
    """Checks if two Quadtree nodes (AABBs) are adjacent (share an edge)."""
    b1 = node1.bounds
    b2 = node2.bounds

    # Check for X-axis adjacency (touching on left/right)
    touch_x = (abs(b1[2] - b2[0]) < 1e-5 or abs(b1[0] - b2[2]) < 1e-5)
    overlap_y = (b1[1] < b2[3] and b1[3] > b2[1])  # Check for Y-overlap

    # Check for Y-axis adjacency (touching on top/bottom)
    touch_y = (abs(b1[3] - b2[1]) < 1e-5 or abs(b1[1] - b2[3]) < 1e-5)
    overlap_x = (b1[0] < b2[2] and b1[2] > b2[0])  # Check for X-overlap

    return (touch_x and overlap_y) or (touch_y and overlap_x)


def create_quadtree_graph(root_node):
    """
    Creates an adjacency-list graph from the FREE leaves of a Quadtree.

    Note: This function is the primary performance bottleneck for deep
    Quadtrees, as it uses an O(n^2) check for adjacency between all
    n FREE leaves.

    Returns:
        tuple: (graph, free_leaves)
    """
    free_leaves = []
    get_free_leaves(root_node, free_leaves)

    graph = {leaf.center: [] for leaf in free_leaves}

    # O(n^2) adjacency check.
    for i in range(len(free_leaves)):
        for j in range(i + 1, len(free_leaves)):
            node1 = free_leaves[i]
            node2 = free_leaves[j]
            if are_adjacent(node1, node2):
                graph[node1.center].append(node2.center)
                graph[node2.center].append(node1.center)

    return graph, free_leaves


def find_leaf_node(pos, root_node):
    """Finds the leaf node in the Quadtree that contains a given (x, y) position."""
    node = root_node
    while node.status == 'MIXED':
        # Traverse down the tree
        x_mid, y_mid = node.center
        if pos[0] < x_mid:  # West
            if pos[1] < y_mid:  # South-West
                node = node.children[2]
            else:  # North-West
                node = node.children[0]
        else:  # East
            if pos[1] < y_mid:  # South-East
                node = node.children[3]
            else:  # North-East
                node = node.children[1]

    # Return the center of the leaf node if it's FREE, otherwise None
    return node.center if node.status == 'FREE' else None


# --- 5. Main Tkinter GUI Application Class ---

class PathPlannerApp:
    """
    The main application class that builds and manages the Tkinter GUI.
    """

    def __init__(self, root, method, param):
        self.root = root
        self.method = method  # 'grid' or 'quadtree'
        self.param = param  # resolution (for grid) or max_depth (for quadtree)
        self.root.title(f"Path Planner - Method: {method.capitalize()} ({param})")

        # Map state
        self.obstacles = []
        self.start_pos = None
        self.end_pos = None
        self.path = None
        self.decomposition = None  # Stores drawing data for grid/quadtree lines

        # Threading state
        self.path_queue = queue.Queue()  # For communication from worker thread
        self.is_calculating = False  # Flag to prevent edits during calculation

        # Control Panel (Left)
        self.control_frame = tk.Frame(root, width=200, bg='lightgray')
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Map Canvas (Right)
        self.canvas = tk.Canvas(root, width=MAP_WIDTH, height=MAP_HEIGHT, bg='white')
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Tool selection variable
        self.current_tool = tk.StringVar(value="set_start")

        self.create_controls()

        # Bind the click event
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def create_controls(self):
        """Creates the buttons and radio buttons in the control panel."""
        label = tk.Label(self.control_frame, text="Tools", font=("Arial", 16), bg='lightgray')
        label.pack(pady=10)

        tools = [
            ("Set Start", "set_start"),
            ("Set Goal", "set_end"),
            ("Add Rectangle", "add_rect"),
            ("Add Circle", "add_circle"),
            ("Add Triangle", "add_triangle"),
            ("Delete Object", "delete")
        ]

        for text, tool in tools:
            rb = tk.Radiobutton(self.control_frame, text=text, variable=self.current_tool,
                                value=tool, bg='lightgray', anchor='w')
            rb.pack(fill='x', padx=10, pady=2)

        sep = ttk.Separator(self.control_frame, orient='horizontal')
        sep.pack(fill='x', pady=10, padx=10)

        # Action Buttons
        self.btn_find_path = tk.Button(self.control_frame, text="FIND PATH",
                                       command=self.find_path, bg='green', fg='white',
                                       font=("Arial", 10, "bold"))
        self.btn_find_path.pack(fill='x', padx=10, pady=5)

        # Loading label (initially hidden)
        self.loading_label = tk.Label(self.control_frame, text="Calculating...",
                                      font=("Arial", 10, "italic"), bg='lightgray', fg='blue')
        # We use .pack_forget() later to hide it

        self.btn_random = tk.Button(self.control_frame, text="Generate Random Map",
                                    command=self.generate_random_map)
        self.btn_random.pack(fill='x', padx=10, pady=5)

        self.btn_clear = tk.Button(self.control_frame, text="Clear Map",
                                   command=self.clear_all, bg='red', fg='white')
        self.btn_clear.pack(fill='x', padx=10, pady=5, side=tk.BOTTOM)

    def on_canvas_click(self, event):
        """Handles mouse clicks on the canvas based on the selected tool."""
        if self.is_calculating: return  # Disallow edits while calculating

        x, y = event.x, event.y
        tool = self.current_tool.get()

        if tool == "set_start":
            self.start_pos = (x, y)
        elif tool == "set_end":
            self.end_pos = (x, y)
        elif tool == "add_rect":
            rect = Rectangle(x - DEFAULT_RECT_SIZE / 2, y - DEFAULT_RECT_SIZE / 2,
                             DEFAULT_RECT_SIZE, DEFAULT_RECT_SIZE)
            self.obstacles.append(rect)
        elif tool == "add_circle":
            circle = Circle(x, y, DEFAULT_CIRCLE_RADIUS)
            self.obstacles.append(circle)
        elif tool == "add_triangle":
            # Add an equilateral triangle
            size = DEFAULT_TRIANGLE_SIZE
            h = (math.sqrt(3) / 2) * size
            p1 = (x, y - h / 2)
            p2 = (x - size / 2, y + h / 2)
            p3 = (x + size / 2, y + h / 2)
            self.obstacles.append(Triangle(p1, p2, p3))
        elif tool == "delete":
            self.delete_object_at(x, y)

        self.redraw_canvas()

    def delete_object_at(self, x, y):
        """Finds and deletes the topmost obstacle at (x, y)."""
        # Iterate in reverse to safely remove items
        for i in range(len(self.obstacles) - 1, -1, -1):
            obs = self.obstacles[i]
            if obs.is_clicked(x, y):
                self.obstacles.pop(i)
                return  # Delete only one object per click

    def generate_random_map(self):
        """Generates a random set of obstacles."""
        if self.is_calculating: return
        self.clear_all()

        num_obstacles = random.randint(5, 15)
        for _ in range(num_obstacles):
            x = random.randint(50, MAP_WIDTH - 50)
            y = random.randint(50, MAP_HEIGHT - 50)

            choice = random.randint(1, 3)
            if choice == 1:
                # Add Rectangle
                w = random.randint(20, 100)
                h = random.randint(20, 100)
                self.obstacles.append(Rectangle(x - w / 2, y - h / 2, w, h))
            elif choice == 2:
                # Add Circle
                r = random.randint(10, 50)
                self.obstacles.append(Circle(x, y, r))
            else:
                # Add Triangle
                size = random.randint(20, 80)
                h = (math.sqrt(3) / 2) * size
                p1 = (x, y - h / 2)
                p2 = (x - size / 2, y + h / 2)
                p3 = (x + size / 2, y + h / 2)
                self.obstacles.append(Triangle(p1, p2, p3))

        self.redraw_canvas()

    def clear_all(self):
        """Clears the entire map (obstacles, start, goal, path)."""
        if self.is_calculating: return
        self.obstacles.clear()
        self.start_pos = None
        self.end_pos = None
        self.path = None
        self.decomposition = None
        self.redraw_canvas()

    def redraw_canvas(self):
        """Redraws the entire canvas from scratch."""
        self.canvas.delete("all")

        # Draw the decomposition (grid lines or quadtree boxes) first
        if self.decomposition:
            for item in self.decomposition:
                if item[0] == 'rect':
                    # (type, (x1,y1,x2,y2), fill_color, outline_color)
                    self.canvas.create_rectangle(*item[1], fill=item[2], outline=item[3], width=0.5)
                elif item[0] == 'line':
                    # (type, (x1,y1,x2,y2), fill_color)
                    self.canvas.create_line(*item[1], fill=item[2], width=0.5)

        # Draw obstacles on top
        for obs in self.obstacles:
            obs.draw(self.canvas)

        # Draw the found path
        if self.path:
            for i in range(len(self.path) - 1):
                p1 = self.path[i]
                p2 = self.path[i + 1]
                self.canvas.create_line(p1, p2, fill='blue', width=3)

        # Draw Start (green circle)
        if self.start_pos:
            self.canvas.create_oval(self.start_pos[0] - 5, self.start_pos[1] - 5,
                                    self.start_pos[0] + 5, self.start_pos[1] + 5,
                                    fill='green', outline='black', width=2)
        # Draw Goal (red square)
        if self.end_pos:
            self.canvas.create_rectangle(self.end_pos[0] - 5, self.end_pos[1] - 5,
                                         self.end_pos[0] + 5, self.end_pos[1] + 5,
                                         fill='red', outline='black', width=2)

    # --- Threading Functions ---

    def find_path(self):
        """
        [UI THREAD]
        Starts the pathfinding calculation in a separate worker thread.
        Disables UI elements and shows a "Calculating..." message.
        """
        if self.is_calculating: return  # Calculation already in progress

        if not self.start_pos or not self.end_pos:
            messagebox.showerror("Error", "Please set a Start and Goal point.")
            return

        # Set loading state
        self.is_calculating = True
        self.btn_find_path.config(state=tk.DISABLED, text="Calculating...")
        self.loading_label.pack(fill='x', padx=10, pady=5)  # Show loading label

        # Clear old path/decomposition
        self.path = None
        self.decomposition = None
        self.redraw_canvas()

        # Start the worker thread
        threading.Thread(target=self._threaded_find_path, daemon=True).start()

        # Start polling the queue for results
        self.check_path_queue()

    def _threaded_find_path(self):
        """
        [WORKER THREAD]
        Performs all heavy calculations (graph build + A*) in the background.
        Puts the result into the 'path_queue' when done.
        """
        graph = {}
        start_node = None
        end_node = None
        path = None
        decomposition = []
        error = None

        try:
            if self.method == 'grid':
                resolution = self.param
                graph, _, cell_w, cell_h = create_grid_graph(self.obstacles, MAP_BOUNDS, resolution)
                start_node = find_grid_node(self.start_pos, MAP_BOUNDS, resolution)
                end_node = find_grid_node(self.end_pos, MAP_BOUNDS, resolution)

                # Generate drawing data for the grid lines
                for i in range(resolution + 1):
                    x = i * cell_w
                    y = i * cell_h
                    decomposition.append(('line', (x, 0, x, MAP_HEIGHT), 'lightgray'))
                    decomposition.append(('line', (0, y, MAP_WIDTH, y), 'lightgray'))

            elif self.method == 'quadtree':
                max_depth = self.param
                root_node = QuadtreeNode(MAP_BOUNDS, 0)
                build_quadtree(root_node, self.obstacles, max_depth)
                graph, free_leaves = create_quadtree_graph(root_node)
                start_node = find_leaf_node(self.start_pos, root_node)
                end_node = find_leaf_node(self.end_pos, root_node)

                # Generate drawing data for the Quadtree cells
                def get_quad_drawing_data(node):
                    if node is None: return
                    x_min, y_min, x_max, y_max = node.bounds
                    if node.status == 'OCCUPIED':
                        decomposition.append(('rect', (x_min, y_min, x_max, y_max), '#ffdddd', 'gray'))
                    elif node.status == 'FREE':
                        decomposition.append(('rect', (x_min, y_min, x_max, y_max), '#ddffdd', 'gray'))
                    if node.children:
                        for child in node.children:
                            get_quad_drawing_data(child)

                get_quad_drawing_data(root_node)

            # Run A*
            if start_node not in graph:
                error = "Start point is in a blocked area!"
            elif end_node not in graph:
                error = "Goal point is in a blocked area!"
            else:
                path = a_star_search(graph, start_node, end_node)

        except Exception as e:
            error = f"An unexpected error occurred: {e}"

        # Put the complete result bundle into the queue
        result = {'path': path, 'decomposition': decomposition, 'error': error}
        self.path_queue.put(result)

    def check_path_queue(self):
        """
        [UI THREAD]
        Polls the 'path_queue' to see if the worker thread is done.
        If a result is found, it calls 'on_path_found'.
        If not, it schedules itself to check again in 100ms.
        """
        try:
            result = self.path_queue.get(block=False)
            # If we get here, a result is ready
            self.on_path_found(result)
        except queue.Empty:
            # No result yet, check again soon
            self.root.after(100, self.check_path_queue)

    def on_path_found(self, result):
        """
        [UI THREAD]
        Handles the result received from the worker thread.
        Updates the GUI, shows messages, and redraws the canvas.
        """
        # Reset loading state
        self.is_calculating = False
        self.btn_find_path.config(state=tk.NORMAL, text="FIND PATH")
        self.loading_label.pack_forget()  # Hide loading label

        # Show any errors
        if result['error']:
            messagebox.showerror("Path Error", result['error'])
        elif not result['path']:
            messagebox.showinfo("Result", "No valid path found.")

        # Store results and redraw
        self.path = result['path']
        self.decomposition = result['decomposition']
        self.redraw_canvas()


# --- 6. Setup Window (Entry Point) ---

def show_setup_window():
    """
    Displays the initial setup window to choose the method and parameter.
    """
    setup_root = tk.Tk()
    setup_root.title("Path Planner Setup")

    method_var = tk.StringVar(value="grid")
    param_var = tk.StringVar(value="25")  # Default value

    main_frame = tk.Frame(setup_root, padx=20, pady=20)
    main_frame.pack()

    tk.Label(main_frame, text="Choose Method:", font=("Arial", 12)).pack(anchor='w')

    rb_grid = tk.Radiobutton(main_frame, text="Uniform Grid", variable=method_var, value="grid")
    rb_grid.pack(anchor='w', padx=10)
    rb_quad = tk.Radiobutton(main_frame, text="Quadtree", variable=method_var, value="quadtree")
    rb_quad.pack(anchor='w', padx=10)

    tk.Label(main_frame, text="Parameter (Resolution / Depth):", font=("Arial", 12)).pack(anchor='w', pady=(10, 0))
    entry_param = tk.Entry(main_frame, textvariable=param_var)
    entry_param.pack(fill='x', padx=10)

    def start_app():
        """Validates input and launches the main application window."""
        method = method_var.get()
        try:
            param = int(param_var.get())
            if param <= 0: raise ValueError

            # Warn user if Quadtree depth is very high, as O(n^2) graph
            # creation can be extremely slow.
            if method == 'quadtree' and param > 8:
                if not messagebox.askyesno("Warning",
                                           f"A depth of {param} is very high and may take a long time to compute.\nContinue?"):
                    return

            # Close setup window and open main app
            setup_root.destroy()
            main_root = tk.Tk()
            app = PathPlannerApp(main_root, method, param)
            main_root.mainloop()

        except ValueError:
            messagebox.showerror("Invalid Input", "The parameter must be a positive integer.",
                                 parent=setup_root)

    btn_start = tk.Button(main_frame, text="Start Application", command=start_app, bg='blue', fg='white')
    btn_start.pack(pady=20, fill='x')

    setup_root.mainloop()


# --- 7. Script Entry Point ---
if __name__ == "__main__":
    show_setup_window()