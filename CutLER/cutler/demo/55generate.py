import random
import time
import math
import os
import json
import copy
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter, defaultdict
import re

filename = "level.png"
ROOT_DIR = os.path.join("Generation", "mixedSizes")
GRID_WIDTH = 50    
GRID_HEIGHT = 14   
TIMEOUT = 1000

DEBUG = False
MERGE = False
FRAMES = False


def merge_wfc_levels(level_paths, output_dir):
    print(f"Merging {len(level_paths)} level datasets into {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    global_tiles_dir = os.path.join(output_dir, "tiles")
    os.makedirs(global_tiles_dir, exist_ok=True)

    global_ratios = defaultdict(float)
    global_adj = defaultdict(lambda: {
        "top": defaultdict(int), 
        "bottom": defaultdict(int), 
        "left": defaultdict(int), 
        "right": defaultdict(int)
    })

    for level_path in level_paths:
        ratio_file = os.path.join(level_path, "ratios.json")
        if os.path.exists(ratio_file):
            with open(ratio_file, 'r') as f:
                ratios = json.load(f)
                for tile, weight in ratios.items():
                    global_ratios[tile] += float(weight)

        adj_file = os.path.join(level_path, "adjacency_rules.txt")
        if os.path.exists(adj_file):
            with open(adj_file, 'r') as f:
                adjacencies = json.load(f)
                for tile, directions in adjacencies.items():
                    for direction, neighbors in directions.items():
                        for n_name, n_weight in neighbors:
                            global_adj[tile][direction][n_name] += int(n_weight)

        tiles_dir = os.path.join(level_path, "tiles")
        if os.path.exists(tiles_dir):
            for tile_img in os.listdir(tiles_dir):
                src_img = os.path.join(tiles_dir, tile_img)
                dst_img = os.path.join(global_tiles_dir, tile_img)
                if not os.path.exists(dst_img):
                    shutil.copy2(src_img, dst_img)

    for tile in global_ratios:
        global_ratios[tile] = round(global_ratios[tile] / len(level_paths), 2)

    final_adj = {}
    for tile, directions in global_adj.items():
        final_adj[tile] = {}
        for d in ["top", "bottom", "left", "right"]:
            final_adj[tile][d] = [[n, w] for n, w in directions[d].items()]

    with open(os.path.join(output_dir, "ratios.json"), 'w') as f:
        json.dump(global_ratios, f, indent=4)
    with open(os.path.join(output_dir, "adjacency_rules.txt"), 'w') as f:
        json.dump(final_adj, f, indent=4)
        
    print("Merge complete!\n")


def load_data(rules_file, ratios_file, tiles_dir):
    print("Loading tiles, rules, and global ratios...")

    with open(rules_file, 'r') as f:
        adjacencies = json.load(f)
        
    with open(ratios_file, 'r') as f:
        ratios = json.load(f)
    
    adjacencies = inject_background_rules(adjacencies, ratios)

    images = {}
    tile_sizes = {} 
    
    valid_extensions = {".png"}
    found_files = [f for f in os.listdir(tiles_dir) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not found_files:
        raise FileNotFoundError(f"No images found in {tiles_dir}!")

    temp_widths = []
    temp_heights = []
    for fname in found_files:
        img_path = os.path.join(tiles_dir, fname)
        img = Image.open(img_path).convert("RGBA")
        images[fname] = img
        temp_widths.append(img.size[0]) 
        temp_heights.append(img.size[1]) 
    
    base_w = Counter(temp_widths).most_common(1)[0][0]
    base_h = Counter(temp_heights).most_common(1)[0][0]
    
    CELL_SIZE = (base_w, base_h)
    print(f"Detected Base Cell Size: {CELL_SIZE}")

    for fname, img in images.items():
        w_units = img.width // CELL_SIZE[0]
        h_units = img.height // CELL_SIZE[1]
        tile_sizes[fname] = (h_units, w_units) 

    for special in ["B", "P"]:
        tile_sizes[special] = (1, 1)

    # --- INTEGER ID BIJECTION MAPS ---
    all_vocab = list(adjacencies.keys())
    for special in ["B", "P"]:
        if special not in all_vocab:
            all_vocab.append(special)
            
    tile_to_id = {name: idx for idx, name in enumerate(all_vocab)}
    id_to_tile = {idx: name for idx, name in enumerate(all_vocab)}

    # Translate Adjacency Rules to Ints
    int_adjacencies = {}
    for t_name, directions in adjacencies.items():
        t_id = tile_to_id[t_name]
        int_adjacencies[t_id] = {}
        for d, neighbors in directions.items():
            int_adjacencies[t_id][d] = [[tile_to_id[n], w] for n, w in neighbors if n in tile_to_id]

    # Translate Ratios & Sizes to Ints
    int_ratios = {tile_to_id[k]: v for k, v in ratios.items() if k in tile_to_id}
    int_tile_sizes = {tile_to_id[k]: v for k, v in tile_sizes.items() if k in tile_to_id}

    return int_adjacencies, int_ratios, images, int_tile_sizes, CELL_SIZE, tile_to_id, id_to_tile


DIR_MAP = {0: "top", 1: "bottom", 2: "left", 3: "right"}


def get_all_tile_names(adjacencies, id_to_tile):
    # Returns Integer IDs of all tiles except padding
    pad_id = next(k for k, v in id_to_tile.items() if v == "P")
    return [k for k in adjacencies.keys() if k != pad_id]


def inject_background_rules(adjacencies, ratios):
    if "B" not in adjacencies:
        adjacencies["B"] = { "top": [], "bottom": [], "left": [], "right": [] }
    if "P" not in adjacencies:
        adjacencies["P"] = { "top": [], "bottom": [], "left": [], "right": [] }

    inverse_dir = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    sky_weight = 1.0

    for d in ["top", "bottom", "left", "right"]:
        if ["B", sky_weight] not in adjacencies["B"][d]:
             adjacencies["B"][d].append(["B", sky_weight])
        if ["P", sky_weight] not in adjacencies["B"][d]:
             adjacencies["B"][d].append(["P", sky_weight])
             
        if ["P", sky_weight] not in adjacencies["P"][d]:
             adjacencies["P"][d].append(["P", sky_weight])
        if ["B", sky_weight] not in adjacencies["P"][d]:
             adjacencies["P"][d].append(["B", sky_weight])

    for tile_name, rules in adjacencies.items():
        if tile_name in ["B", "P"]: continue
        for direction, neighbors in rules.items():
            for n_name, n_weight in neighbors:
                if n_name == "B":
                    inv_d = inverse_dir[direction]
                    if not any(n[0] == tile_name for n in adjacencies["B"][inv_d]):
                        adjacencies["B"][inv_d].append([tile_name, n_weight])
                if n_name == "P":
                    inv_d = inverse_dir[direction]
                    if not any(n[0] == tile_name for n in adjacencies["P"][inv_d]):
                        adjacencies["P"][inv_d].append([tile_name, n_weight])
    
    print("Injected rules for 'B' (Background) and 'P' (Padding).")
    return adjacencies


def initialize_grid(height, width, playable_tiles):
    return [[set(playable_tiles) for _ in range(width)] for _ in range(height)]


def pad_grid(grid, pad_value):
    width = len(grid[0])
    for row in grid:
        row.insert(0, pad_value)
        row.append(pad_value)
        
    new_width = width + 2
    top_row = [pad_value] * new_width
    bottom_row = [pad_value] * new_width
    
    grid.insert(0, top_row)
    grid.append(bottom_row)
    return grid

def analyze_complex_tiles(playable_tiles, id_to_tile, adjacencies):
    raw_groups = defaultdict(set)
    pattern = re.compile(r"(.+?)_tile_.+_y\d+_x\d+")
    
    for tile_id in playable_tiles:
        tile_name = id_to_tile[tile_id]
        match = pattern.match(tile_name)
        if match:
            base_name = match.group(1)
            raw_groups[base_name].add(tile_id)
            
    blueprints = {}
    tile_meta = {}
    
    for base_name, pieces in raw_groups.items():
        visited = {}
        
        # 1. Start BFS graph walk from an arbitrary piece
        start_piece = next(iter(pieces))
        queue = [(start_piece, 0, 0)]
        visited[start_piece] = (0, 0)
        
        while queue:
            curr_id, cy, cx = queue.pop(0)
            
            # 2. Walk the exact topological links defined by your adjacency rules
            for d_key, dy, dx in [("top", -1, 0), ("bottom", 1, 0), ("left", 0, -1), ("right", 0, 1)]:
                allowed = adjacencies.get(curr_id, {}).get(d_key, [])
                for n_id, _ in allowed:
                    # If the neighbor is part of the same complex structure, map its coordinate!
                    if n_id in pieces and n_id not in visited:
                        visited[n_id] = (cy + dy, cx + dx)
                        queue.append((n_id, cy + dy, cx + dx))
                        
        # Failsafe: Just in case OpenCV extracted a completely disconnected floating pixel
        for p_id in pieces:
            if p_id not in visited:
                match = re.search(r"_y(\d+)_x(\d+)", id_to_tile[p_id])
                visited[p_id] = (int(match.group(1)), int(match.group(2)))
                
        # 3. Normalize the graph so the top-left corner of the structure is (0,0)
        min_y = min(y for y, x in visited.values())
        min_x = min(x for y, x in visited.values())
        
        blueprints[base_name] = {'pieces': [], 'max_y': 0, 'max_x': 0}
        
        for p_id, (ry, rx) in visited.items():
            norm_y = ry - min_y
            norm_x = rx - min_x
            
            blueprints[base_name]['pieces'].append({
                'id': p_id,
                'local_y': norm_y,
                'local_x': norm_x
            })
            tile_meta[p_id] = {
                'base_name': base_name,
                'local_y': norm_y,
                'local_x': norm_x
            }
            blueprints[base_name]['max_y'] = max(blueprints[base_name]['max_y'], norm_y)
            blueprints[base_name]['max_x'] = max(blueprints[base_name]['max_x'], norm_x)
            
    return blueprints, tile_meta

def initial_grid_prune(grid, tile_meta, blueprints, adjacencies, pad_id):
    playable_height = len(grid) - 2
    playable_width = len(grid[0]) - 2
    
    for y in range(1, playable_height + 1):
        for x in range(1, playable_width + 1):
            to_remove = set()
            for tile_id in grid[y][x]:
                
                # 1. Structural Bounds Check (Complex tiles)
                if tile_id in tile_meta:
                    meta = tile_meta[tile_id]
                    base_name = meta['base_name']
                    max_y = blueprints[base_name]['max_y']
                    max_x = blueprints[base_name]['max_x']
                    
                    # Calculate implied top-left and bottom-right using normalized integers
                    top_y = y - meta['local_y']
                    left_x = x - meta['local_x']
                    bottom_y = top_y + max_y
                    right_x = left_x + max_x
                    
                    # STRICT BOUNDS: Keep entirely within playable area (1 to playable_width/height)
                    if top_y < 1 or left_x < 1 or bottom_y > playable_height or right_x > playable_width:
                        to_remove.add(tile_id)
                        continue # Already removed, skip other checks
                        
                # 2. Mid-Air Boundary Trap (ALL tiles, Core and Complex)
                # If we are NOT at the absolute bottom, we cannot place tiles that ONLY allow "P" below them
                if y < playable_height:
                    allowed_bottoms = adjacencies.get(tile_id, {}).get("bottom", [])
                    # Check if "P" is the ONLY allowed neighbor
                    if len(allowed_bottoms) == 1 and allowed_bottoms[0][0] == pad_id:
                        to_remove.add(tile_id)
                        continue
                        
                # If we are NOT at the absolute top, we cannot place tiles that ONLY allow "P" above them
                if y > 1:
                    allowed_tops = adjacencies.get(tile_id, {}).get("top", [])
                    # Check if "P" is the ONLY allowed neighbor
                    if len(allowed_tops) == 1 and allowed_tops[0][0] == pad_id:
                        to_remove.add(tile_id)
                        continue
                        
            if to_remove:
                grid[y][x] -= to_remove
                
    return grid

def is_footprint_clear(grid, target_y, target_x, target_tile_id, blueprints, tile_meta):
    if target_tile_id not in tile_meta:
        return True 
        
    meta = tile_meta[target_tile_id]
    blueprint_pieces = blueprints[meta['base_name']]['pieces']
    
    for piece in blueprint_pieces:
        global_y = target_y - meta['local_y'] + piece['local_y']
        global_x = target_x - meta['local_x'] + piece['local_x']
        
        # STRICT BOUNDS: Do not allow complex pieces to overwrite the padding (P)
        if global_y < 1 or global_y >= len(grid) - 1 or global_x < 1 or global_x >= len(grid[0]) - 1:
            return False
            
        cell_contents = grid[global_y][global_x]
        
        if isinstance(cell_contents, set):
            if piece['id'] not in cell_contents:
                return False
        else:
            if cell_contents != piece['id']:
                return False
                
    return True

def get_min_entropy_cell(grid):
    min_entropy = float('inf')
    min_cells = []
    
    for y in range(len(grid) - 2, -1, -1):
        for x in range(len(grid[0])):
            if isinstance(grid[y][x], set):
                entropy = len(grid[y][x])
                
                if entropy == 0:
                    return (-1, -1) 
                    
                if 0 < entropy < min_entropy:
                    min_entropy = entropy
                    min_cells = [(y, x)]
                elif entropy == min_entropy:
                    min_cells.append((y, x))
    
    if min_cells:
        return random.choice(min_cells)
    
    return None


def collapse_cell(grid, y, x, adjacencies, tile_sizes, ratios, history, steps, state, total_roof_weight, total_floor_weight, pad_id, blueprints, tile_meta):
    possible_tiles = list(grid[y][x])    
    if not possible_tiles: return None

    total_grid_cells = len(grid) * len(grid[0])
    total_ratio_sum = sum(ratios.values()) or 1

    progress_ratio = state['collapsed_count'] / total_grid_cells
    for t in ratios.keys():
        target_ratio = ratios.get(t, 1.0) / total_ratio_sum
        desired_total = target_ratio * total_grid_cells
        ideal_current = desired_total * progress_ratio
        
        history[t]['actual'].append(state['current'].get(t, 0))
        history[t]['ideal'].append(ideal_current)
    
    steps.append(state['collapsed_count'])

    weights = []
    legal_indices = [] 
    playable_width = len(grid[0]) - 2 

    for i, tile in enumerate(possible_tiles):
        if not is_footprint_clear(grid, y, x, tile, blueprints, tile_meta):
            weights.append(0.0)
            grid[y][x].discard(tile)
            continue

        directional_probs = []
        is_illegal = False

        for dy, dx, d_key in [(-1,0,"top"), (1,0,"bottom"), (0,-1,"left"), (0,1,"right")]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
                if not isinstance(grid[ny][nx], set):
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
            
        legal_indices.append(i)
        
        base_ratio_weight = ratios.get(tile, 1.0) / total_ratio_sum
        local_w = base_ratio_weight
        for p in directional_probs:
            local_w *= p
        
        allowed_bottoms = adjacencies.get(tile, {}).get("bottom", [])
        allowed_tops = adjacencies.get(tile, {}).get("top", [])
        
        valid_bottoms = [n[0] for n in allowed_bottoms]
        valid_tops = [n[0] for n in allowed_tops]

        height, _ = tile_sizes.get(tile, (1, 1))
        is_against_floor = (y + 1 < len(grid)) and (grid[y + 1][x] == pad_id)
        is_against_roof = (y - height >= 0) and (grid[y - height][x] == pad_id)
        
        acts_as_floor = is_against_floor and (pad_id in valid_bottoms)
        acts_as_roof = is_against_roof and (pad_id in valid_tops) 
        
        is_structural = False
            
        if acts_as_roof:
            my_roof_weight = next((w for n, w in allowed_tops if n == pad_id), 0)
            target_roof_count = (my_roof_weight / total_roof_weight) * playable_width
            
            if state['roof'].get(tile, 0) < target_roof_count:
                is_structural = True
                
        elif acts_as_floor:
            my_floor_weight = next((w for n, w in allowed_bottoms if n == pad_id), 0)
            target_floor_count = (my_floor_weight / total_floor_weight) * playable_width
            
            if state['floor'].get(tile, 0) < target_floor_count:
                is_structural = True

        pacing_multiplier = 1.0
        base_weight = ratios.get(tile, 1.0) 
        
        # FIX 1: Instant O(1) dictionary check instead of string slicing
        is_complex_clone = tile in tile_meta

        if not is_structural and not is_complex_clone:
            target_ratio = base_weight / total_ratio_sum
            desired_total_count = target_ratio * total_grid_cells
            actual_count = state['current'].get(tile, 0)
            
            if actual_count >= desired_total_count:
                pacing_multiplier = 0.0
            else:
                progress_ratio = state['collapsed_count'] / total_grid_cells
                ideal_current_count = desired_total_count * progress_ratio
                deficit = ideal_current_count - actual_count
                
                if deficit < 0:
                    pacing_multiplier = math.exp(deficit * 5) 
                else:
                    percent_behind = deficit / desired_total_count if desired_total_count > 0 else 0
                    pacing_multiplier = 1.0 + min(0.25, percent_behind * 0.4)

        final_weight = local_w * pacing_multiplier
        weights.append(final_weight)

    total_weight = sum(weights)
    
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
        chosen_tile = random.choices(possible_tiles, weights=weights, k=1)[0]
    else:
        if not legal_indices:
            return None 
            
        legal_ids = [possible_tiles[i] for i in legal_indices]
        bg_id = next((k for k, v in ratios.items() if k in legal_ids), None)
        if bg_id is not None:
            chosen_tile = bg_id
        else:
            fallback_weights = [0.0] * len(possible_tiles)
            for i in legal_indices:
                fallback_weights[i] = 1.0 / len(legal_indices)
            chosen_tile = random.choices(possible_tiles, weights=fallback_weights, k=1)[0]

    collapsed_coords = []
    
    if chosen_tile in tile_meta:
        # ATOMIC STAMP: It's a complex structure, stamp every piece instantly!
        meta = tile_meta[chosen_tile]
        base_name = meta['base_name']
        blueprint = blueprints[base_name]['pieces']
        
        for piece in blueprint:
            global_y = y - meta['local_y'] + piece['local_y']
            global_x = x - meta['local_x'] + piece['local_x']
            
            # Force write the exact piece to the grid
            grid[global_y][global_x] = piece['id']
            collapsed_coords.append((global_y, global_x))
            
            # FIX 2: Track the specific piece ID, not the chosen anchor tile
            if piece['id'] in state['current']:
                state['current'][piece['id']] += 1
                state['collapsed_count'] += 1
                
    else:
        # GENERIC STAMP: It's a 1x1 core tile
        grid[y][x] = chosen_tile
        collapsed_coords.append((y, x))
        
        if chosen_tile in state['current']:
            state['current'][chosen_tile] += 1
            state['collapsed_count'] += 1
            
            # Structural tracking for core tiles
            is_against_floor = (y + 1 < len(grid)) and (grid[y + 1][x] == pad_id)
            is_against_roof = (y - 1 >= 0) and (grid[y - 1][x] == pad_id)
            if is_against_roof:
                state['roof'][chosen_tile] += 1
            if is_against_floor:
                state['floor'][chosen_tile] += 1
                
    return collapsed_coords




def propagate(grid, y, x, adjacencies, tile_sizes, timeout_check_callback=None):
    stack = [(y, x)]
    in_stack = {(y, x)}
    
    while stack:
        if timeout_check_callback:
            timeout_check_callback(grid)
            
        cy, cx = stack.pop()
        in_stack.remove((cy, cx))
        cell_contents = grid[cy][cx]
        
        if not isinstance(cell_contents, set):
            height, width = tile_sizes.get(cell_contents, (1, 1))
        else:
            height, width = 1, 1
            
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
                            
                            if updated and (ny, nx) not in in_stack:
                                stack.append((ny, nx))
                                in_stack.add((ny, nx))


def update_cell(grid, target_y, target_x, source_y, source_x, direction, adjacencies):
    original_len = len(grid[target_y][target_x])
    source_contents = grid[source_y][source_x]

    if not isinstance(source_contents, set):
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
    # Linear scan through small integer arrays is blistering fast in PyPy
    for item in allowed_neighbors:
        if item[0] == target_tile:
            return True
    return False


class TimeoutException(Exception):
    def __init__(self, grid):
        self.grid = grid


class RetryException(Exception):
    def __init__(self, grid):
        self.grid = grid


class MaxRetriesException(Exception):
    def __init__(self, grid):
        self.grid = grid


def generate_level(height, width, adjacencies, ratios, tile_sizes, timeout, pad_id, playable_tiles): 
    global_start_time = time.time()
    
    def check_timeout(current_grid):
        if time.time() - global_start_time > timeout:
            raise TimeoutException(current_grid)
            
    def generate_level_attempt(h, w):
            grid = initialize_grid(h, w, playable_tiles)
            grid = pad_grid(grid, pad_value=pad_id)

            blueprints, tile_meta = analyze_complex_tiles(playable_tiles, id_to_tile, adjacencies)
            print("Pruning grid...")
            grid = initial_grid_prune(grid, tile_meta, blueprints, adjacencies, pad_id)
            print("Starting...")

            generation_history = {t: {'actual': [], 'ideal': []} for t in ratios.keys()}
            collapse_steps = []
            
            placements = 0 
            frame_states = [] 

            state = {
                'current': {t: 0 for t in ratios.keys()},
                'roof': {t: 0 for t in ratios.keys()},
                'floor': {t: 0 for t in ratios.keys()},
                'collapsed_count': 0
            }
            playable_height = len(grid) - 2 # Ignore the top/bottom padding rows
            playable_width = len(grid[0]) - 2 # Ignore left/right padding cols

            for y in range(1, playable_height + 1):
                for x in range(1, playable_width + 1):
                    to_remove = set()
                    for tile_id in grid[y][x]:
                        # If we are NOT at the absolute bottom row, we cannot place tiles that ONLY allow "P" below them
                        if y < playable_height:
                            allowed_bottoms = [n[0] for n in adjacencies.get(tile_id, {}).get("bottom", [])]
                            if len(allowed_bottoms) == 1 and allowed_bottoms[0] == pad_id:
                                to_remove.add(tile_id)
                        
                        # If we are NOT at the absolute top row, we cannot place tiles that ONLY allow "P" above them
                        if y > 1:
                            allowed_tops = [n[0] for n in adjacencies.get(tile_id, {}).get("top", [])]
                            if len(allowed_tops) == 1 and allowed_tops[0] == pad_id:
                                to_remove.add(tile_id)
                                
                    grid[y][x] -= to_remove
            
            total_roof_weight = sum(next((w for n, w in rules.get("top", []) if n == pad_id), 0) for rules in adjacencies.values()) or 1
            total_floor_weight = sum(next((w for n, w in rules.get("bottom", []) if n == pad_id), 0) for rules in adjacencies.values()) or 1
            
            h_len = len(grid)
            w_len = len(grid[0])
            for x in range(w_len):
                if grid[0][x] == pad_id: 
                    propagate(grid, 0, x, adjacencies, tile_sizes, check_timeout)       
                if grid[h_len-1][x] == pad_id: 
                    propagate(grid, h_len-1, x, adjacencies, tile_sizes, check_timeout)

            # for y in range(h_len):
            #     if grid[y][0] == pad_id:
            #         propagate(grid, y, 0, adjacencies, tile_sizes, check_timeout)
            #     if grid[y][w_len-1] == pad_id:
            #         propagate(grid, y, w_len-1, adjacencies, tile_sizes, check_timeout)

            while True:
                check_timeout(grid)
                cell = get_min_entropy_cell(grid)

                if cell == (-1, -1):
                    raise RetryException(grid)

                if cell is None:
                    break 
                
                y, x = cell
                if not isinstance(grid[y][x], set): continue
                    
                collapsed_coords = collapse_cell(grid, y, x, adjacencies, tile_sizes, ratios, generation_history, collapse_steps, state, total_roof_weight, total_floor_weight, pad_id, blueprints, tile_meta)
                
                if collapsed_coords:
                    # Propagate outwards from EVERY piece we just locked in
                    for cy, cx in collapsed_coords:
                        propagate(grid, cy, cx, adjacencies, tile_sizes, check_timeout)
                    placements += 1
                else:
                    print(f"Dead end at {y},{x}. Restarting...")
                    raise RetryException(grid)
            
            return grid, generation_history, collapse_steps, frame_states

    attempts = 0
    while True:
        attempts += 1
        try:
            return generate_level_attempt(height, width)
        except RetryException as e:
            if attempts >= 3:
                print(f"Encountered 5 dead ends. Aborting this generation...")
                raise MaxRetriesException(e.grid)
            continue
        except TimeoutException as e:
            raise e 


def render_grid(grid, images, tile_sizes, cell_size, id_to_tile, max_entropy=0): 
    rows = len(grid)
    cols = len(grid[0])
    cw, ch = cell_size
    img_w = cols * cw
    img_h = rows * ch
    
    canvas = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    heatmap_overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw_canvas = ImageDraw.Draw(canvas)
    
    draw = None
    if max_entropy > 0:
        draw = ImageDraw.Draw(heatmap_overlay)
    covered_cells = set()

    for y in range(rows - 1, -1, -1):
        for x in range(cols):
            cell_val = grid[y][x]
            if isinstance(cell_val, set): 
                continue 

            tile_name = id_to_tile.get(cell_val, "")
            if not tile_name or tile_name in ["B", "P"]: continue
            if (x, y) in covered_cells: continue

            # --- Calculate exact pixel placement ---
            px = x * cw
            
            # --- Try to find the image ---
            img_key = tile_name
            if img_key not in images and img_key + ".png" in images:
                img_key += ".png"
            
            # --- Image Found! ---
            img = images[img_key]
            py = (y + 1) * ch - img.height
            
            
            # Draw the actual image over the yellow square
            canvas.alpha_composite(img, dest=(px, py))

            # Mark covered cells
            height_in_cells, width_in_cells = tile_sizes.get(cell_val, (1, 1))
            for dy in range(height_in_cells):
                for dx in range(width_in_cells):
                    covered_cells.add((x + dx, y - dy))

    if max_entropy > 0:
        canvas.alpha_composite(heatmap_overlay)
        
    return canvas

def save_readable_debug_grid(grid, id_to_tile, filepath):
    lines = []
    for row in grid:
        row_str = []
        for cell in row:
            if isinstance(cell, set):
                if len(cell) == 0:
                    row_str.append("[DEAD END]")
                else:
                    row_str.append(f"[Set: {len(cell)}]")
            else:
                name = id_to_tile.get(cell, str(cell))
                if name == "P":
                    row_str.append("PAD")
                elif name == "B":
                    row_str.append("BKG")
                elif "_tile_" in name:
                    # Compress: mario_1t_seg_62_tile_00004_y6_x5 -> m62(y6_x5)
                    try:
                        parts = name.split("_tile_")
                        base = parts[0].split("_seg_")[1] # gets "62"
                        coords = parts[1].split("_y")[1] # gets "6_x5"
                        row_str.append(f"m{base}(y{coords})")
                    except:
                        row_str.append(name[:12])
                else:
                    # Core tile: tile_00004.png -> t_00004
                    row_str.append(name.replace("tile_", "t_").replace(".png", ""))
        
        # Format columns nicely with a fixed width of 14 characters
        formatted_row = " | ".join(f"{s:^14}" for s in row_str)
        lines.append(formatted_row)
        
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    level_folders = []
    if os.path.exists(ROOT_DIR):
        if os.path.exists(os.path.join(ROOT_DIR, "adjacency_rules.txt")):
            level_folders.append(ROOT_DIR)
        else:
            for item in os.listdir(ROOT_DIR):
                item_path = os.path.join(ROOT_DIR, item)
                if item in ("merged_data", "Generations"):
                    continue
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "adjacency_rules.txt")):
                    level_folders.append(item_path)

    if MERGE and len(level_folders) > 1:
        merged_output_dir = os.path.join(ROOT_DIR, "merged_data")
        merge_wfc_levels(level_folders, merged_output_dir)
        target_folders = [merged_output_dir]
    else:
        target_folders = level_folders

    for base_dir in target_folders:
        print(f"\n--- Processing Directory: {base_dir} ---")
        
        TILES_DIR = os.path.join(base_dir, "tiles")
        RULES_FILE = os.path.join(base_dir, "adjacency_rules.txt")
        RATIOS_FILE = os.path.join(base_dir, "ratios.json")        

        base_output_path = os.path.join(base_dir, "Generations", filename)
        folder = os.path.dirname(base_output_path)
        os.makedirs(folder, exist_ok=True)

        out_filename = os.path.basename(base_output_path)  
        name, extension = os.path.splitext(out_filename)
        final_path = base_output_path
        counter = 1
        
        try:
            for i in range(1):
                while os.path.exists(final_path):
                    new_filename = f"{name}{counter}{extension}"
                    final_path = os.path.join(folder, new_filename)
                    counter += 1
                
                adj, ratios, imgs, sizes, c_size, tile_to_id, id_to_tile = load_data(RULES_FILE, RATIOS_FILE, TILES_DIR) 
                
                pad_id = tile_to_id["P"]
                playable_tiles = get_all_tile_names(adj, id_to_tile)
                
                #print("Generating grid...")
                final_grid_data, gen_history, col_steps, frame_states = generate_level(
                    GRID_HEIGHT, GRID_WIDTH, adj, ratios, sizes, TIMEOUT, pad_id, playable_tiles
                )
                
                print("Rendering final image...")
                final_img = render_grid(final_grid_data, imgs, sizes, c_size, id_to_tile)
                
                print(f"Saving to {final_path}...")
                final_img.save(final_path)
                print("Done processing folder!")

                
            
        except TimeoutException as e:
            print(f"Generation timed out for {base_dir}. Saving debug image and skipping to the next folder...")
            debug_img = render_grid(e.grid, imgs, sizes, c_size, id_to_tile)
            debug_path = os.path.join(folder, f"failed_timeout_{name}{counter}{extension}")
            debug_img.save(debug_path)
            continue
        except MaxRetriesException as e:
            print(f"Max retries (5) reached for {base_dir}. Saving debug image and skipping to the next folder...")
            debug_img = render_grid(e.grid, imgs, sizes, c_size, id_to_tile,max_entropy = len(playable_tiles))
            debug_path = os.path.join(folder, f"failed_deadend_{name}{counter}{extension}")
            debug_img.save(debug_path)

            txt_path = os.path.join(folder, f"failed_deadend_{name}{counter}.txt")
            save_readable_debug_grid(e.grid, id_to_tile, txt_path)
            continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in {base_dir}: {e}")
            continue