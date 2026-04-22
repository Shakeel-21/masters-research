import random
import time
import math
import os
import json
import copy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = "Generation\\mario1t ratios\\mario_1t_used"
TILES_DIR = BASE_DIR
RULES_FILE = os.path.join(BASE_DIR, "adjacency_rules.txt")
RATIOS_FILE = os.path.join(BASE_DIR, "ratios.json")        

base_output_path = os.path.join(BASE_DIR, "..", "no clump 50", "pic.png")

folder = os.path.dirname(base_output_path)

# --- Ensure the frames directory exists! ---
os.makedirs(folder, exist_ok=True)

filename = os.path.basename(base_output_path)  
name, extension = os.path.splitext(filename)
final_path = base_output_path
counter = 1



GRID_WIDTH = 230    
GRID_HEIGHT = 14   
TIMEOUT = 5


FRAMES = True
# FRAMES = False

def load_data():
    """Loads images, calculates tile sizes (in grid units), loads rules and ratios."""
    print("Loading tiles, rules, and global ratios...")

    with open(RULES_FILE, 'r') as f:
        adjacencies = json.load(f)
        
    with open(RATIOS_FILE, 'r') as f:
        ratios = json.load(f)
    
    adjacencies = inject_background_rules(adjacencies, ratios)

    images = {}
    tile_sizes = {} 
    
    valid_extensions = {".png"}
    found_files = [f for f in os.listdir(TILES_DIR) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not found_files:
        raise FileNotFoundError(f"No images found in {TILES_DIR}!")

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

    return adjacencies, ratios, images, tile_sizes, CELL_SIZE

# Direction Mapping: 0:Top, 1:Bottom, 2:Left, 3:Right
DIR_MAP = {0: "top", 1: "bottom", 2: "left", 3: "right"}

def get_all_tile_names(adjacencies):
    keys = list(adjacencies.keys())
    if "B" not in keys:
        keys.append("B")
    return [k for k in keys if k != "P"]

def inject_background_rules(adjacencies, ratios):
    if "B" not in adjacencies:
        adjacencies["B"] = { "top": [], "bottom": [], "left": [], "right": [] }

    inverse_dir = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    
    sky_weight = 1.0

    for d in ["top", "bottom", "left", "right"]:
        if ["B", sky_weight] not in adjacencies["B"][d]:
             adjacencies["B"][d].append(["B", sky_weight])
        if ["P", sky_weight] not in adjacencies["B"][d]:
             adjacencies["B"][d].append(["P", sky_weight])

    for tile_name, rules in adjacencies.items():
        if tile_name == "B": continue
        for direction, neighbors in rules.items():
            for n_name, n_weight in neighbors:
                if n_name == "B":
                    inv_d = inverse_dir[direction]
                    if not any(n[0] == tile_name for n in adjacencies["B"][inv_d]):
                        adjacencies["B"][inv_d].append([tile_name, n_weight])
    
    print("Injected rules for 'B' (Background).")
    return adjacencies

def initialize_grid(height, width, all_tiles):
    return [[set(all_tiles) for _ in range(width)] for _ in range(height)]

def pad_grid(grid, pad_value="P"):
    width = len(grid[0])
    new_row = [pad_value] * width
    grid.append(new_row)
    return grid

def get_min_entropy_cell(grid, adjacencies):
    min_entropy = float('inf')
    min_cells = []
    
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
                
                if 0 < entropy < min_entropy:
                    min_entropy = entropy
                    min_cells = [(y, x)]
                elif entropy == min_entropy:
                    min_cells.append((y, x))
    
    if min_cells:
        return random.choice(min_cells)
    
    return None

def collapse_cell(grid, y, x, adjacencies, tile_sizes, ratios, history, steps):
    possible_tiles = list(grid[y][x])    
    if not possible_tiles: return None

    current_counts = {t: 0 for t in ratios.keys()}
    collapsed_count = 0   
    for row in grid:
        for cell in row:
            if isinstance(cell, str) and cell in current_counts:
                current_counts[cell] += 1
                collapsed_count += 1   

    total_grid_cells = len(grid) * len(grid[0])
    total_ratio_sum = sum(ratios.values())
    if total_ratio_sum == 0: total_ratio_sum = 1

    progress_ratio = collapsed_count / total_grid_cells
    for t in ratios.keys():
        target_ratio = ratios.get(t, 1.0) / total_ratio_sum
        desired_total = target_ratio * total_grid_cells
        ideal_current = desired_total * progress_ratio
        
        history[t]['actual'].append(current_counts.get(t, 0))
        history[t]['ideal'].append(ideal_current)
    
    steps.append(collapsed_count)

    weights = []

    for tile in possible_tiles:
        directional_probs = []
        is_illegal = False

        for dy, dx, d_key in [(-1,0,"top"), (1,0,"bottom"), (0,-1,"left"), (0,1,"right")]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
                if isinstance(grid[ny][nx], str):
                    allowed = adjacencies.get(tile, {}).get(d_key, [])
                    n_weight = next((w for n, w in allowed if n == grid[ny][nx]), 0.0)

                    if n_weight <= 0.0:
                        is_illegal = True
                        break

                    total_allowed_weight = sum(w for n, w in allowed)
                    prob = n_weight / total_allowed_weight if total_allowed_weight > 0 else 0
                    directional_probs.append(prob)

        if is_illegal:
            weights.append(0.0)
            continue

        
        base_ratio_weight = ratios.get(tile, 1.0) / total_ratio_sum
        local_w = base_ratio_weight
        for p in directional_probs:
            local_w *= p
        

        valid_bottoms = [n[0] for n in adjacencies.get(tile, {}).get("bottom", [])]
        is_structural = (tile == "B") or ("P" in valid_bottoms)

        pacing_multiplier = 1.0
        base_weight = ratios.get(tile, 1.0) 

        if not is_structural:
            target_ratio = base_weight / total_ratio_sum
            desired_total_count = target_ratio * total_grid_cells
            actual_count = current_counts.get(tile, 0)
            
            if actual_count >= desired_total_count:
                pacing_multiplier = 0.0 
            else:
                progress_ratio = collapsed_count / total_grid_cells
                ideal_current_count = desired_total_count * progress_ratio
                deficit = ideal_current_count - actual_count
                
                if deficit < 0:
                    pacing_multiplier = math.exp(deficit * 5) 
                else:
                    pacing_multiplier = min(1.5, 1.0 + (deficit * 0.1))
                    # pacing_multiplier = 1.0 + deficit 

        final_weight = local_w  * pacing_multiplier
        weights.append(final_weight)

    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        if "B" in possible_tiles:
            return "B"
        weights = [1.0 / len(possible_tiles)] * len(possible_tiles)


    chosen_tile = random.choices(possible_tiles, weights=weights, k=1)[0]
    height, width = tile_sizes.get(chosen_tile, (1, 1))

    if (y - height + 1 < 0) or (x + width > len(grid[0])):      
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
        cell_contents = grid[cy][cx]
        
        if isinstance(cell_contents, str):
            height, width = tile_sizes.get(cell_contents, (1, 1))
        else:
            height, width = 1, 1
            
        y_min = cy - height + 1
        y_max = cy     
        y_bottom = cy
        y_top = cy - height + 1
        x_left = cx
        x_right = cx + width - 1
        
        for dy in range(-1, height + 1):
            for dx in range(-1, width + 1):
                ny, nx = cy - dy, cx + dx 
                
                if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
                    if isinstance(grid[ny][nx], set):
                        direction = get_direction(ny, nx, y_top, y_bottom, x_left, x_right)
                        if direction is not None:
                            updated = update_cell(grid, ny, nx, cy, cx, direction, adjacencies)
                            if updated:
                                stack.append((ny, nx))

def update_cell(grid, target_y, target_x, source_y, source_x, direction, adjacencies):
    original_len = len(grid[target_y][target_x])
    source_contents = grid[source_y][source_x]

    if isinstance(source_contents, str):
        possible_sources = [source_contents]
    else:
        possible_sources = list(source_contents)
        
    to_remove = set()

    for potential_tile in grid[target_y][target_x]:
        has_support = False
        
        for src_tile in possible_sources:
            if is_valid_neighbor(src_tile, potential_tile, direction, adjacencies):
                has_support = True
                break

        if not has_support:
            to_remove.add(potential_tile)

    if to_remove:
        grid[target_y][target_x] -= to_remove
        return len(grid[target_y][target_x]) < original_len
        
    return False

def get_direction(y, x, y_top, y_bottom, x_left, x_right):
    is_vertical_out = y < y_top or y > y_bottom
    is_horizontal_out = x < x_left or x > x_right
    
    if is_vertical_out and is_horizontal_out:
        return None 
        
    if y < y_top: return 0    
    if y > y_bottom: return 1 
    if x < x_left: return 2   
    if x > x_right: return 3  
    
    return None

def is_valid_neighbor(source_tile, target_tile, direction_idx, adjacencies):
    dir_key = DIR_MAP.get(direction_idx)
    if not dir_key: return False
    
    if source_tile not in adjacencies:
        return False 

    allowed_neighbors = adjacencies[source_tile][dir_key] 
    allowed_names = {n[0] for n in allowed_neighbors}
    
    return target_tile in allowed_names

class TimeoutException(Exception):
    def __init__(self, grid):
        self.grid = grid

class RetryException(Exception):
    def __init__(self, grid):
        self.grid = grid

# Pass FRAMES in so the function knows whether to record states
def generate_level(height, width, adjacencies, ratios, tile_sizes, timeout): 
    all_tiles = get_all_tile_names(adjacencies)
    
    def check_timeout(start_t, current_grid):
        if time.time() - start_t > timeout:
            raise TimeoutException(current_grid)
            
    def generate_level_attempt(h, w):
            start_time = time.time()
            grid = initialize_grid(h, w, all_tiles)
            grid = pad_grid(grid, pad_value="P")
            generation_history = {t: {'actual': [], 'ideal': []} for t in ratios.keys()}
            collapse_steps = []
            
            placements = 0 
            frame_states = [] # <--- NEW: List to hold our successful grid snapshots
            
            while True:
                check_timeout(start_time, grid)
                
                cell = get_min_entropy_cell(grid, adjacencies)

                if cell == (-1, -1):
                    raise RetryException(grid)

                if cell is None:
                    # Level finished successfully!
                    break 
                
                y, x = cell
                
                if isinstance(grid[y][x], str): continue
                    
                tile = collapse_cell(grid, y, x, adjacencies, tile_sizes, ratios, generation_history, collapse_steps)
                
                if tile is not None:
                    propagate(grid, y, x, adjacencies, tile_sizes)
                    
                    # --- NEW LOGIC: Save a text snapshot, not an image! ---
                    placements += 1
                    if FRAMES and placements % 50 == 0:
                        # deepcopy ensures we don't accidentally link to the original sets
                        frame_states.append(copy.deepcopy(grid)) 
                        
                else:
                    print(f"Dead end at {y},{x}. Restarting...")
                    raise RetryException(grid) # This throws away frame_states!
            
            final_grid = []
            for row in grid:
                new_row = []
                for cell in row:
                    if isinstance(cell, str):
                        new_row.append(cell)
                    else:
                        new_row.append("B") 
                final_grid.append(new_row)
                
            # --- Return frame_states at the end! ---
            return np.array(final_grid), generation_history, collapse_steps, frame_states

    attempts = 0
    while True:
        attempts += 1
        try:
            return generate_level_attempt(height, width)
        except RetryException as e:
            continue
        except TimeoutException as e:
            continue

def plot_pacing_graphs(history, steps):
    num_tiles = len(history)
    cols = 4 
    rows = math.ceil(num_tiles / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows), sharex=True)
    
    if num_tiles > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for i, (tile, data) in enumerate(history.items()):
        ax = axes[i]
        ax.plot(steps, data['actual'], label='Actual', color='blue', linewidth=2)
        ax.plot(steps, data['ideal'], label='Ideal', color='orange', linestyle='--', linewidth=2)
        ax.set_title(f"Tile: '{tile}'")
        ax.grid(True, alpha=0.3)
        if i == 0: 
            ax.legend()
        
        ax.fill_between(steps, data['actual'], data['ideal'], 
                        where=[a < i for a, i in zip(data['actual'], data['ideal'])], 
                        color='red', alpha=0.1)
        ax.fill_between(steps, data['actual'], data['ideal'], 
                        where=[a >= i for a, i in zip(data['actual'], data['ideal'])], 
                        color='green', alpha=0.1)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def render_grid(grid, images, tile_sizes, cell_size, max_entropy=0): 
    rows = len(grid)
    cols = len(grid[0])
    cw, ch = cell_size
    img_w = cols * cw
    img_h = rows * ch
    
    canvas = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    
    # --- NEW: Create a transparent overlay for the heatmap ---
    heatmap_overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = None
    if max_entropy > 0:
        draw = ImageDraw.Draw(heatmap_overlay)
        
    covered_cells = set()

    for y in range(rows - 1, -1, -1):
        for x in range(cols):
            tile_name = grid[y][x]

            # --- NEW LOGIC: Draw Heatmap for uncollapsed sets ---
            if isinstance(tile_name, set): 
                if draw and len(tile_name) > 0:
                    entropy = len(tile_name)
                    
                    # Normalize entropy from 0.0 (hottest/fewest options) to 1.0 (coldest/most options)
                    normalized = min(1.0, max(0.0, (entropy - 1) / (max_entropy - 1))) if max_entropy > 1 else 0
                    
                    # Colors: Hot = Red (255, 0, 0), Cold = Blue (0, 0, 255)
                    r = int(255 * (1 - normalized))
                    b = int(255 * normalized)
                    
                    # Opacity: Hot/Red is dark (200), Cold/Blue is faint (40)
                    alpha = int(40 + 160 * (1 - normalized)) 
                    
                    px = x * cw
                    py = y * ch
                    draw.rectangle([px, py, px + cw, py + ch], fill=(r, 0, b, alpha))
                continue 

            # Draw normal collapsed tiles
            if tile_name in ["B", "P"]: continue
            if tile_name not in images: continue
            if (x, y) in covered_cells: continue

            img = images[tile_name]
            height_in_cells, width_in_cells = tile_sizes.get(tile_name, (1, 1))

            px = x * cw
            py = (y + 1) * ch - img.height

            canvas.alpha_composite(img, dest=(px, py))

            for dy in range(height_in_cells):
                for dx in range(width_in_cells):
                    target_x = x + dx
                    target_y = y - dy 

                    if target_x >= cols or target_y < 0:
                        continue

                    if not isinstance(grid[target_y][target_x], set) and grid[target_y][target_x] == tile_name:
                        covered_cells.add((target_x, target_y))

    # --- Apply the heatmap overlay on top of the tiles ---
    if max_entropy > 0:
        canvas.alpha_composite(heatmap_overlay)

    return canvas

if __name__ == "__main__":
    try:
        for i in range(1):
            while os.path.exists(final_path):
                new_filename = f"{name}{counter}{extension}"
                final_path = os.path.join(folder, new_filename)
                counter += 1
            adj, ratios, imgs, sizes, c_size = load_data() 
            
            # Calculate max possible entropy to calibrate the heatmap colors
            max_possible_entropy = len(get_all_tile_names(adj))
            
            print("Generating grid...")
            current_run_name = f"{name}{counter if counter > 1 else ''}" 
            

            final_grid_data, gen_history, col_steps, frame_states = generate_level(
                GRID_HEIGHT, GRID_WIDTH, adj, ratios, sizes, TIMEOUT
            ) 
            
            # --- Render the frames with the Heatmap turned ON ---
            if FRAMES and frame_states:
                print(f"Level generated successfully! Rendering {len(frame_states)} frames with heatmap...")
                for i, state_grid in enumerate(frame_states):
                    step_num = (i + 1) * 50
                    # Pass max_possible_entropy to trigger the heatmap
                    temp_img = render_grid(state_grid, imgs, sizes, c_size, max_entropy=max_possible_entropy)
                    frame_filename = f"{current_run_name}_step_{step_num:04d}.png"
                    frame_path = os.path.join(folder, frame_filename)
                    temp_img.save(frame_path)
            
            print("Rendering final image...")
            # Pass 0 to max_entropy so the final image has no heatmap overlay
            final_img = render_grid(final_grid_data, imgs, sizes, c_size, max_entropy=0)
            
            print(f"Saving to {final_path}...")
            final_img.save(final_path)
            plot_pacing_graphs(gen_history, col_steps)
            print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")