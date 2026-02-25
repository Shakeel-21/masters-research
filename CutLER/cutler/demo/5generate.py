import random
import time
import math
import os
import json
import numpy as np
from PIL import Image

BASE_DIR = "Generation\\mario9 test"
TILES_DIR = os.path.join(BASE_DIR, "mario_9")
RULES_FILE = os.path.join(TILES_DIR, "adjacency_rules.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, "22.png")

SKY_WEIGHT = 50

GRID_WIDTH = 12
GRID_HEIGHT = 7
TIMEOUT = 5

def load_data():
    """Loads images, calculates tile sizes (in grid units), and loads rules."""
    print("Loading tiles and rules...")

    with open(RULES_FILE, 'r') as f:
        adjacencies = json.load(f)
    
    adjacencies = inject_background_rules(adjacencies)

    images = {}
    tile_sizes = {} 
    

    valid_extensions = {".png"}
    found_files = [f for f in os.listdir(TILES_DIR) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not found_files:
        raise FileNotFoundError("No images found in core3 folder!")

    temp_dims = []
    for fname in found_files:
        img_path = os.path.join(TILES_DIR, fname)
        img = Image.open(img_path).convert("RGBA")
        images[fname] = img
        temp_dims.append(img.size) 
    

    min_w = min(d[0] for d in temp_dims)
    min_h = min(d[1] for d in temp_dims)
    CELL_SIZE = (min_w, min_h) 
    print(f"Detected Base Cell Size: {CELL_SIZE}")

    for fname, img in images.items():
        w_units = img.width // CELL_SIZE[0]
        h_units = img.height // CELL_SIZE[1]
        tile_sizes[fname] = (h_units, w_units) 


    for special in ["B", "P"]:
        tile_sizes[special] = (1, 1)

    return adjacencies, images, tile_sizes, CELL_SIZE
# Direction Mapping: 0:Top, 1:Bottom, 2:Left, 3:Right
DIR_MAP = {0: "top", 1: "bottom", 2: "left", 3: "right"}

def get_all_tile_names(adjacencies):
    """Extracts all valid tile names, explicitly adding 'B'."""
    keys = list(adjacencies.keys())
    if "B" not in keys:
        keys.append("B")
    return [k for k in keys if k != "P"]

def inject_background_rules(adjacencies):
    """
    If 'B' is not a key in adjacencies, create it by inverting 
    the rules of all other tiles that point to 'B'.
    """
    if "B" not in adjacencies:
        adjacencies["B"] = {
            "top": [], "bottom": [], "left": [], "right": []
        }

    # standard inverse mapping
    inverse_dir = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

    # 1. Allow B to connect to B (Background acts like air)
    # Give it a high weight so large empty spaces are encouraged
    for d in ["top", "bottom", "left", "right"]:
        if ["B", SKY_WEIGHT] not in adjacencies["B"][d]:
             adjacencies["B"][d].append(["B", SKY_WEIGHT])

    for d in ["top", "bottom", "left", "right"]:
        if ["P", 10] not in adjacencies["B"][d]:
             adjacencies["B"][d].append(["P", 10])

    # 2. Scan all other tiles to see who connects to B
    for tile_name, rules in adjacencies.items():
        if tile_name == "B": continue
        
        for direction, neighbors in rules.items():
            # Check if this tile connects to B in this direction
            # e.g. tile_01["top"] contains "B"
            for neighbor_data in neighbors:
                n_name, n_weight = neighbor_data
                
                if n_name == "B":
                    # If tile_01 has B on TOP, then B has tile_01 on BOTTOM
                    inv_d = inverse_dir[direction]
                    
                    # Add tile_01 to B's allowed neighbors
                    # We reuse the weight defined in the tile
                    if [tile_name, n_weight] not in adjacencies["B"][inv_d]:
                        adjacencies["B"][inv_d].append([tile_name, n_weight])
    
    print("Injected rules for 'B' (Background).")
    return adjacencies

def calculate_global_ratios(adjacencies):
    """
    Calculates the global frequency/weight of each tile based on 
    how often it appears in the adjacency rules.
    """
    counts = {}
    for tile, directions in adjacencies.items():
        if tile not in counts: counts[tile] = 0
        # Summing the weights found in the adjacency lists
        for direction in directions.values():
            for neighbor_data in direction:
                # format is [name, weight]
                n_name, n_weight = neighbor_data
                if n_name not in counts: counts[n_name] = 0
                counts[n_name] += n_weight
    return counts

def valid_check(grid, adjacencies):
    for y in range(len(grid) - 1): 
        for x in range(len(grid[0])):
            tile = grid[y][x]

            if not isinstance(tile, str): 
                continue

            up = down = left = right = -1
            
            # Check Down
            if y + 1 < len(grid):
                down = grid[y+1][x]
                if down == tile: down = -1 
                if down == tile and isinstance(down, str): down = -1

            # Check Up
            if y - 1 >= 0:
                up = grid[y-1][x]
                if up == tile: up = -1
            
            # Check Left
            if x - 1 >= 0:
                left = grid[y][x-1]
                if left == tile: left = -1

            # Check Right
            if x + 1 < len(grid[0]):
                right = grid[y][x+1]
                if right == tile: right = -1
        
            neighbors = [up, down, right, left] 
            check_dirs = [up, down, left, right] 
            
            for direction_idx in range(4):
                neighbor_tile = check_dirs[direction_idx]

                if isinstance(neighbor_tile, str):   
                    if not is_valid_neighbor(neighbor_tile, tile, direction_idx, adjacencies):
                        print(f"{tile} is not allowed in direction {direction_idx} of {neighbor_tile}")
                        
                        
                        collapse_cell(grid, y, x, adjacencies, tile_sizes={}, ratios={}) 

def initialize_grid(height, width, all_tiles):
    # Initialize with a set of ALL valid tile names
    return [[set(all_tiles) for _ in range(width)] for _ in range(height)]

def pad_grid(grid, pad_value="P"):
    width = len(grid[0])
    # The padding row is fixed strings, not sets
    new_row = [pad_value] * width
    grid.append(new_row)
    return grid

def get_min_entropy_cell(grid, adjacencies):
    min_entropy = float('inf')
    min_cells = []
    
    # Iterate excluding the padding row
    for y in range(len(grid) - 2, -1, -1):
        for x in range(len(grid[0])):
            if len(grid[y][x]) == 0:
                return (-1, -1) 

            if isinstance(grid[y][x], set):
                is_above_padding = (y + 1 == len(grid) - 1)
                if is_above_padding:
                    filtered = set()
                    for t in grid[y][x]:
                        valid_bottoms = [n[0] for n in adjacencies[t]["bottom"]]
                        if "P" in valid_bottoms:
                            filtered.add(t)
                    grid[y][x] = filtered
                    

                    if len(grid[y][x]) == 0:
                        return (-1, -1)

                entropy = len(grid[y][x])
                
                if 1 < entropy < min_entropy:
                    min_entropy = entropy
                    min_cells = [(y, x)]
                elif entropy == min_entropy:
                    min_cells.append((y, x))
                elif entropy == 1:
                    min_cells.append((y, x))
    
    if min_cells:
        return random.choice(min_cells)
    
    return None

def collapse_cell(grid, y, x, adjacencies, tile_sizes, ratios):
    possible_tiles = list(grid[y][x])    
    if not possible_tiles:
        return None

    weights = []

    current_counts = {t: 0 for t in ratios.keys()}
    for row in grid:
        for cell in row:
            if isinstance(cell, str) and cell in current_counts:
                current_counts[cell] += 1
    
    total_grid_cells = len(grid) * len(grid[0])

    for tile in possible_tiles:
        # Default weight if not found
        tile_ratio = ratios.get(tile, 1)

        total_ratio_sum = sum(ratios.values())
        if total_ratio_sum == 0: total_ratio_sum = 1
        
        desired_count = (tile_ratio / total_ratio_sum) * total_grid_cells
        
        if current_counts.get(tile, 0) < desired_count:
            weights.append(tile_ratio)
        else:
            weights.append(0.01) 
    

    if sum(weights) == 0:
        weights = [1] * len(weights)

    # Normalize
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    is_above_padding = (y + 1 == len(grid) - 1)
    

    if is_above_padding:
        weights = []
        for tile in possible_tiles:
            valid_bottoms = adjacencies[tile]["bottom"]
            p_weight = next((item[1] for item in valid_bottoms if item[0] == "P"), 0)
            
            if p_weight > 0:
                weights.append(p_weight)
            else:
                weights.append(0) 

        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(possible_tiles)] * len(possible_tiles)

    if not possible_tiles: return None
    
    chosen_tile = random.choices(possible_tiles, weights=weights, k=1)[0]

    height, width = tile_sizes.get(chosen_tile, (1, 1))
    

    if (y + height > len(grid)) or (x + width > len(grid[0])):        
        return None

    area_free = True
    for dy in range(height):
        for dx in range(width):
            if not (0 <= y - dy < len(grid) and 0 <= x + dx < len(grid[0])):
                area_free = False
                break
            if not isinstance(grid[y - dy][x + dx], set):
                area_free = False
                break
        if not area_free: break
        
    if area_free:
        for dy in range(height):
            for dx in range(width):
                grid[y-dy][x+dx] = chosen_tile
        return chosen_tile
    
    return None

def propagate(grid, y, x, adjacencies, tile_sizes):
    stack = [(y, x)]
    while stack:
        cy, cx = stack.pop()

        if isinstance(grid[cy][cx], set):
            continue
            
        tile = grid[cy][cx]
        height, width = tile_sizes.get(tile, (1, 1))
        
        # Define bounds of the current tile placement
        y_min = cy - height + 1
        y_max = cy     
        y_bottom = cy
        y_top = cy - height + 1
        x_left = cx
        x_right = cx + width - 1

        
        for dy in range(-1, height + 1):
            for dx in range(-1, width + 1):
                
                ny, nx = cy - dy, cx + dx 
                
                # Bounds check
                if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
                    if isinstance(grid[ny][nx], set):
                        updated = update_cell(grid, ny, nx, cy, cx, y_top, y_bottom, x_left, x_right, adjacencies)
                        if updated:
                            stack.append((ny, nx))

def update_cell(grid, y, x, prev_y, prev_x, y_top, y_bottom, x_left, x_right, adjacencies):
    original_len = len(grid[y][x])
    direction = get_direction(y, x, y_top, y_bottom, x_left, x_right)
    
    if direction is None: return False

    prev_tile = grid[prev_y][prev_x] 
    
    to_remove = set()
    for potential_tile in grid[y][x]:
        
        if not is_valid_neighbor(prev_tile, potential_tile, direction, adjacencies):
            to_remove.add(potential_tile)
    
    if to_remove:
        grid[y][x] -= to_remove
        return len(grid[y][x]) < original_len
    return False

def get_direction(y, x, y_top, y_bottom, x_left, x_right):
    if y < y_top: return 0    # Target is above the block
    if y > y_bottom: return 1 # Target is below the block
    if x < x_left: return 2   # Target is left of the block
    if x > x_right: return 3  # Target is right of the block
    return None

def is_valid_neighbor(source_tile, target_tile, direction_idx, adjacencies):
    # Convert index to string key
    dir_key = DIR_MAP.get(direction_idx)
    if not dir_key: return False
    
   
    if source_tile not in adjacencies:
        return False 

    allowed_neighbors = adjacencies[source_tile][dir_key] # List of [name, freq]
    
    # Extract just the names
    allowed_names = {n[0] for n in allowed_neighbors}
    
    return target_tile in allowed_names



    

class TimeoutException(Exception):
    def __init__(self, grid):
        self.grid = grid

class RetryException(Exception):
    def __init__(self, grid):
        self.grid = grid

def debug_display(grid):
    print("\n--- FAILED GRID STATE ---")
    for row in grid:
        line = []
        for cell in row:
            if isinstance(cell, str):
                line.append(f"[{cell[-7:-4]:^3}]")
            elif isinstance(cell, set):
                
                line.append(f"<{len(cell)} >") 
            else:
                line.append("[ ? ]")
        print("".join(line))
    print("-------------------------\n")

def generate_level(height, width, adjacencies, tile_sizes, timeout):
    # Prepare global data
    all_tiles = get_all_tile_names(adjacencies)
    ratios = calculate_global_ratios(adjacencies)
    print(ratios)
    
    
    def check_timeout(start_t, current_grid):
        if time.time() - start_t > timeout:
            raise TimeoutException(current_grid)
    def generate_level_attempt(h, w):
            start_time = time.time()
            grid = initialize_grid(h, w, all_tiles)
            grid = pad_grid(grid, pad_value="P")
            
            while True:
                check_timeout(start_time, grid)
                

                cell = get_min_entropy_cell(grid, adjacencies)

                if cell == (-1, -1):
                    # print("Contradiction detected (0 entropy). Restarting...")
                    raise RetryException(grid)

                if cell is None:
                    for y in range(len(grid) - 1): 
                        for x in range(len(grid[0])):
                            if isinstance(grid[y][x], set) and len(grid[y][x]) == 0:
                                raise RetryException(grid)
                    break 
                
                y, x = cell
                
                # Skip if already collapsed (string)
                if isinstance(grid[y][x], str): continue
                    
                tile = collapse_cell(grid, y, x, adjacencies, tile_sizes, ratios)
                
                if tile is not None:
                    propagate(grid, y, x, adjacencies, tile_sizes)
                else:
                    # Collapse failed (no valid weighted choice)
                    print(f"Dead end at {y},{x}. Restarting...")
                    raise RetryException(grid)
            
            # Convert to Final Output
            final_grid = []
            for row in grid:
                new_row = []
                for cell in row:
                    if isinstance(cell, str):
                        new_row.append(cell)
                    else:
                        new_row.append("B") 
                final_grid.append(new_row)
                
            return np.array(final_grid)

    # Main Retry Loop
    attempts = 0
    while True:
        attempts += 1
        try:
            # print(f"--- Generation Attempt {attempts} ---")
            return generate_level_attempt(height, width)
            
        except RetryException as e:
            # Catch the exception 'e', access 'e.grid', and display it
            # print("Attempt failed. visualizing state:")
            # debug_display(e.grid)
            continue
            
        except TimeoutException as e:
            print(f"Attempt {attempts} timed out ({timeout}s).")
            print("State at timeout:")
            debug_display(e.grid)
            continue

def render_grid(grid, images, cell_size):
    rows = len(grid)
    cols = len(grid[0])
    cw, ch = cell_size
    img_w = cols * cw
    img_h = rows * ch
    
    canvas = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))

    covered_cells = set()

    for y in range(rows):
        for x in range(cols):
            tile_name = grid[y][x]

            if tile_name in ["B", "P"]: continue
            if tile_name not in images: continue
            if (x, y) in covered_cells: continue

            img = images[tile_name]

            width_in_cells = math.ceil(img.width / cw)
            height_in_cells = math.ceil(img.height / ch)

            is_tall_object = height_in_cells > 1
            has_neighbor_below = (y + 1 < rows) and (grid[y+1][x] == tile_name)

            if is_tall_object and has_neighbor_below:
                continue

            px = x * cw
            py = (y + 1) * ch - img.height

            canvas.alpha_composite(img, dest=(px, py))

            for dy in range(height_in_cells):
                for dx in range(width_in_cells):
                    target_x = x + dx
                    target_y = y - dy 

                    if target_x >= cols or target_y < 0:
                        continue

                    if grid[target_y][target_x] == tile_name:
                        covered_cells.add((target_x, target_y))

    return canvas



if __name__ == "__main__":
    try:
        adj, imgs, sizes, c_size = load_data()
        
        print("Generating grid...")
        final_grid_data = generate_level(GRID_HEIGHT, GRID_WIDTH, adj, sizes,TIMEOUT)
        
        print("Rendering image...")
        final_img = render_grid(final_grid_data, imgs, c_size)
        
        print(f"Saving to {OUTPUT_PATH}...")
        final_img.save(OUTPUT_PATH)
        debug_display(final_grid_data)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")