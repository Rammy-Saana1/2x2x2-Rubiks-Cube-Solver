import time
import tracemalloc
import random
from collections import deque
from heapq import heappush, heappop
import copy
import sys
import pygame

# Increase recursion limit to prevent stack overflow in DFS
sys.setrecursionlimit(10000)

class Cube:
    def __init__(self):
        # 2x2x2 cube: 6 faces (U, D, F, B, L, R), each with 4 stickers (0-3)
        # Colors: W (white), Y (yellow), G (green), B (blue), O (orange), R (red)
        self.state = {
            'U': ['W'] * 4,  # Up face: white
            'D': ['Y'] * 4,  # Down face: yellow
            'F': ['G'] * 4,  # Front face: green
            'B': ['B'] * 4,  # Back face: blue
            'L': ['O'] * 4,  # Left face: orange
            'R': ['R'] * 4   # Right face: red
        }
        self.moves = []  # Track applied moves
        self.solved_state = copy.deepcopy(self.state)  # Store solved state
    
    def get_state_string(self):
        # Convert state to a string for hashing (visited set)
        return ''.join([''.join(face) for face in self.state.values()])

    def is_solved(self):
        # Check if each face has uniform color
        return all(len(set(face)) == 1 for face in self.state.values())

    def possible_moves(self):
        # Moves: U (up), R (right), F (front), clockwise (no prime), counterclockwise (prime)
        return ["U", "U'", "R", "R'", "F", "F'"]

    def apply_move(self, move):
        new_cube = Cube()
        new_cube.state = copy.deepcopy(self.state)
        new_cube.moves = self.moves + [move]
        
        # Move logic for 2x2x2 (affects stickers on adjacent faces)
        if move == "U":
            # Rotate up face clockwise
            new_cube.state['U'] = [new_cube.state['U'][3], new_cube.state['U'][0], 
                                  new_cube.state['U'][1], new_cube.state['U'][2]]
            # Update adjacent faces (F, R, B, L)
            temp = new_cube.state['F'][0:2]
            new_cube.state['F'][0:2] = new_cube.state['R'][0:2]
            new_cube.state['R'][0:2] = new_cube.state['B'][0:2]
            new_cube.state['B'][0:2] = new_cube.state['L'][0:2]
            new_cube.state['L'][0:2] = temp
        elif move == "U'":
            # Rotate up face counterclockwise
            new_cube.state['U'] = [new_cube.state['U'][1], new_cube.state['U'][2], 
                                  new_cube.state['U'][3], new_cube.state['U'][0]]
            temp = new_cube.state['F'][0:2]
            new_cube.state['F'][0:2] = new_cube.state['L'][0:2]
            new_cube.state['L'][0:2] = new_cube.state['B'][0:2]
            new_cube.state['B'][0:2] = new_cube.state['R'][0:2]
            new_cube.state['R'][0:2] = temp
        elif move == "R":
            # Rotate right face clockwise
            new_cube.state['R'] = [new_cube.state['R'][3], new_cube.state['R'][0], 
                                  new_cube.state['R'][1], new_cube.state['R'][2]]
            temp = [new_cube.state['F'][1], new_cube.state['F'][3]]
            new_cube.state['F'][1] = new_cube.state['U'][1]
            new_cube.state['F'][3] = new_cube.state['U'][3]
            new_cube.state['U'][1] = new_cube.state['B'][0]
            new_cube.state['U'][3] = new_cube.state['B'][2]
            new_cube.state['B'][0] = new_cube.state['D'][1]
            new_cube.state['B'][2] = new_cube.state['D'][3]
            new_cube.state['D'][1] = temp[0]
            new_cube.state['D'][3] = temp[1]
        elif move == "R'":
            # Rotate right face counterclockwise
            new_cube.state['R'] = [new_cube.state['R'][1], new_cube.state['R'][2], 
                                  new_cube.state['R'][3], new_cube.state['R'][0]]
            temp = [new_cube.state['F'][1], new_cube.state['F'][3]]
            new_cube.state['F'][1] = new_cube.state['D'][1]
            new_cube.state['F'][3] = new_cube.state['D'][3]
            new_cube.state['D'][1] = new_cube.state['B'][0]
            new_cube.state['D'][3] = new_cube.state['B'][2]
            new_cube.state['B'][0] = new_cube.state['U'][1]
            new_cube.state['B'][2] = new_cube.state['U'][3]
            new_cube.state['U'][1] = temp[0]
            new_cube.state['U'][3] = temp[1]
        elif move == "F":
            # Rotate front face clockwise
            new_cube.state['F'] = [new_cube.state['F'][3], new_cube.state['F'][0], 
                                  new_cube.state['F'][1], new_cube.state['F'][2]]
            temp = [new_cube.state['U'][2], new_cube.state['U'][3]]
            new_cube.state['U'][2] = new_cube.state['L'][1]
            new_cube.state['U'][3] = new_cube.state['L'][3]
            new_cube.state['L'][1] = new_cube.state['D'][0]
            new_cube.state['L'][3] = new_cube.state['D'][1]
            new_cube.state['D'][0] = new_cube.state['R'][0]
            new_cube.state['D'][1] = new_cube.state['R'][2]
            new_cube.state['R'][0] = temp[0]
            new_cube.state['R'][2] = temp[1]
        elif move == "F'":
            # Rotate front face counterclockwise
            new_cube.state['F'] = [new_cube.state['F'][1], new_cube.state['F'][2], 
                                  new_cube.state['F'][3], new_cube.state['F'][0]]
            temp = [new_cube.state['U'][2], new_cube.state['U'][3]]
            new_cube.state['U'][2] = new_cube.state['R'][0]
            new_cube.state['U'][3] = new_cube.state['R'][2]
            new_cube.state['R'][0] = new_cube.state['D'][0]
            new_cube.state['R'][2] = new_cube.state['D'][1]
            new_cube.state['D'][0] = new_cube.state['L'][1]
            new_cube.state['D'][1] = new_cube.state['L'][3]
            new_cube.state['L'][1] = temp[0]
            new_cube.state['L'][3] = temp[1]
        return new_cube

    def visualize(self):
        # Print a 2D net of the cube
        u = self.state['U']
        d = self.state['D']
        f = self.state['F']
        b = self.state['B']
        l = self.state['L']
        r = self.state['R']
        print("    +-----+")
        print(f"    | {u[0]} {u[1]} | U")
        print(f"    | {u[2]} {u[3]} |")
        print("    +-----+")
        print(f"+-----+-----+-----+-----+")
        print(f"| {l[0]} {l[1]} | {f[0]} {f[1]} | {r[0]} {r[1]} | {b[0]} {b[1]} | L F R B")
        print(f"| {l[2]} {l[3]} | {f[2]} {f[3]} | {r[2]} {r[3]} | {b[2]} {b[3]} |")
        print(f"+-----+-----+-----+-----+")
        print("    +-----+")
        print(f"    | {d[0]} {d[1]} | D")
        print(f"    | {d[2]} {d[3]} |")
        print("    +-----+")

    def visualize_pygame(self, solutions):
        # Initialize Pygame
        pygame.init()
        base_width, base_height = 600, 400
        screen = pygame.display.set_mode((base_width, base_height), pygame.RESIZABLE)
        pygame.display.set_caption("2x2x2 Rubik's Cube Solver")
        clock = pygame.time.Clock()

        # Color mapping
        color_map = {
            'W': (255, 255, 255),  # White
            'Y': (255, 255, 0),    # Yellow
            'G': (0, 128, 0),      # Green
            'B': (0, 0, 255),      # Blue
            'O': (255, 165, 0),    # Orange
            'R': (255, 0, 0)       # Red
        }

        # Base face positions (x, y) for top-left corner of each 2x2 face
        base_face_positions = {
            'U': (200, 50),
            'L': (100, 150),
            'F': (200, 150),
            'R': (300, 150),
            'B': (400, 150),
            'D': (200, 250)
        }

        # Font
        font = pygame.font.SysFont('arial', 20)

        # Animation state
        current_cube = Cube()
        current_cube.state = copy.deepcopy(self.state)
        current_algorithm = 'DFS'
        move_index = 0
        paused = False
        last_update = pygame.time.get_ticks()
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        current_algorithm = 'DFS'
                        move_index = 0
                        current_cube.state = copy.deepcopy(self.state)
                        paused = False
                    elif event.key == pygame.K_2:
                        current_algorithm = 'BFS'
                        move_index = 0
                        current_cube.state = copy.deepcopy(self.state)
                        paused = False
                    elif event.key == pygame.K_3:
                        current_algorithm = 'A*'
                        move_index = 0
                        current_cube.state = copy.deepcopy(self.state)
                        paused = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_LEFT and paused:
                        if move_index > 0:
                            move_index -= 1
                            current_cube.state = copy.deepcopy(self.state)
                            for i in range(move_index):
                                current_cube = current_cube.apply_move(solutions[current_algorithm][i])
                    elif event.key == pygame.K_RIGHT and paused:
                        if move_index < len(solutions[current_algorithm]):
                            current_cube = current_cube.apply_move(solutions[current_algorithm][move_index])
                            move_index += 1
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            # Get current window size
            width, height = screen.get_size()
            scale_x = width / base_width
            scale_y = height / base_height
            face_positions = {k: (v[0] * scale_x, v[1] * scale_y) for k, v in base_face_positions.items()}
            sticker_size = 50 * min(scale_x, scale_y)

            # Update animation (if not paused)
            if not paused and solutions[current_algorithm] and move_index < len(solutions[current_algorithm]):
                current_time = pygame.time.get_ticks()
                if current_time - last_update > 500:  # 500ms per move
                    current_cube = current_cube.apply_move(solutions[current_algorithm][move_index])
                    move_index += 1
                    last_update = current_time

            # Draw background
            screen.fill((200, 200, 200))  # Light gray

            # Draw each face
            for face, (x_base, y_base) in face_positions.items():
                for i in range(4):
                    x = x_base + (i % 2) * sticker_size
                    y = y_base + (i // 2) * sticker_size
                    color = color_map[current_cube.state[face][i]]
                    pygame.draw.rect(screen, color, (x, y, sticker_size, sticker_size))
                    pygame.draw.rect(screen, (0, 0, 0), (x, y, sticker_size, sticker_size), 2)
                # Add face label
                label = font.render(face, True, (0, 0, 0))
                screen.blit(label, (x_base + sticker_size / 2, y_base - 20 * scale_y))

            # Display status
            move_text = f"Move: {solutions[current_algorithm][move_index-1] if move_index > 0 else 'Start'}"
            move_label = font.render(move_text, True, (0, 0, 0))
            screen.blit(move_label, (20 * scale_x, 20 * scale_y))
            
            algo_label = font.render(f"Algorithm: {current_algorithm}", True, (0, 0, 0))
            screen.blit(algo_label, (20 * scale_x, 50 * scale_y))
            
            status_label = font.render("Paused" if paused else "Playing", True, (0, 0, 0))
            screen.blit(status_label, (20 * scale_x, 80 * scale_y))
            
            controls = [
                "Controls:",
                "1: DFS, 2: BFS, 3: A*",
                "SPACE: Pause/Resume",
                "LEFT/RIGHT: Step (when paused)",
                "ESC: Exit"
            ]
            for i, text in enumerate(controls):
                label = font.render(text, True, (0, 0, 0))
                screen.blit(label, (20 * scale_x, (100 + i * 20) * scale_y))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def count_mismatched_corners(self):
        # Count corners not in correct position/color
        corners = [
            (self.state['U'][0], self.state['F'][0], self.state['L'][0]),  # UFL
            (self.state['U'][1], self.state['R'][0], self.state['F'][1]),  # URF
            (self.state['U'][2], self.state['B'][1], self.state['R'][3]),  # UBR
            (self.state['U'][3], self.state['L'][2], self.state['B'][0]),  # ULB
        ]
        solved_corners = [
            ('W', 'G', 'O'),  # UFL
            ('W', 'R', 'G'),  # URF
            ('W', 'B', 'R'),  # UBR
            ('W', 'O', 'B'),  # ULB
        ]
        return sum(1 for c in corners if c not in solved_corners)

    def count_mismatched_edges(self):
        # 2x2x2 has no edge pieces
        return 0

def pattern_database_heuristic(cube):
    # Admissible heuristic: number of mismatched corners
    return cube.count_mismatched_corners() / 1.5  # Adjusted for broader exploration

def dfs(cube, depth, max_depth, visited, last_move=None):
    try:
        if cube.is_solved():
            return cube.moves
        if depth > max_depth:
            return None
        visited.add(cube.get_state_string())
        for move in cube.possible_moves():
            # Prune only inverse moves (e.g., U after U')
            if last_move and (
                (move == last_move + "'" and last_move[-1] != "'") or 
                (move + "'" == last_move)
            ):
                continue
            new_cube = cube.apply_move(move)
            if new_cube.get_state_string() not in visited:
                result = dfs(new_cube, depth + 1, max_depth, visited, move)
                if result:
                    return [move] + result
        return None
    except RecursionError:
        print("DFS: Recursion depth exceeded")
        return None

def bfs(cube, visited):
    try:
        queue = deque([(cube, [])])
        visited.add(cube.get_state_string())
        while queue:
            current_cube, moves = queue.popleft()
            if current_cube.is_solved():
                return moves
            for move in current_cube.possible_moves():
                new_cube = current_cube.apply_move(move)
                if new_cube.get_state_string() not in visited:
                    visited.add(new_cube.get_state_string())
                    queue.append((new_cube, moves + [move]))
        return None
    except MemoryError:
        print("BFS: Memory limit exceeded")
        return None

def a_star(cube, visited):
    try:
        heap = [(0, 0, cube, [])]  # (f_score, tie-breaker, cube, moves)
        visited.add(cube.get_state_string())
        counter = 0  # Tie-breaker for heap
        while heap:
            _, _, current_cube, moves = heappop(heap)
            if current_cube.is_solved():
                return moves
            for move in current_cube.possible_moves():
                new_cube = current_cube.apply_move(move)
                if new_cube.get_state_string() not in visited:
                    visited.add(new_cube.get_state_string())
                    g_score = len(moves) + 1
                    h_score = pattern_database_heuristic(new_cube)
                    f_score = g_score + h_score
                    counter += 1
                    heappush(heap, (f_score, counter, new_cube, moves + [move]))
        return None
    except MemoryError:
        print("A*: Memory limit exceeded")
        return None

def scramble_cube(cube, num_moves):
    moves = cube.possible_moves()
    for _ in range(num_moves):
        cube = cube.apply_move(random.choice(moves))
    return cube

def run_algorithm(algorithm, cube, max_depth=8):
    try:
        tracemalloc.start()
        start_time = time.time()
        visited = set()
        if algorithm == dfs:
            result = algorithm(cube, 0, max_depth, visited, None)
        else:
            result = algorithm(cube, visited)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Algorithm: {algorithm.__name__}")
        print(f"Solution: {result if result else 'None'}")
        print(f"Runtime: {end_time - start_time:.2f} seconds")
        print(f"States Explored: {len(visited)}")
        print(f"Memory Used: {peak / 1024 / 1024:.2f} MB")
        return result
    except Exception as e:
        print(f"Error in {algorithm.__name__}: {str(e)}")
        return None

if __name__ == "__main__":
    print("Testing 2x2x2 Rubik's Cube")
    cube = Cube()
    cube = scramble_cube(cube, 5)  # 5-move scramble
    print(f"Initial State: {cube.get_state_string()}")
    cube.visualize()  # Text-based visualization
    
    # Run algorithms and store solutions
    solutions = {}
    print("\nDFS Results:")
    solutions['DFS'] = run_algorithm(dfs, cube, max_depth=8)
    
    print("\nBFS Results:")
    solutions['BFS'] = run_algorithm(bfs, cube)
    
    print("\nA* Results:")
    solutions['A*'] = run_algorithm(a_star, cube)
    
    # Run Pygame visualization with all solutions
    if any(solutions.values()):
        cube.visualize_pygame(solutions)