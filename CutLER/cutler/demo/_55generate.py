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
import heapq

DEBUG = False

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
            # Store as {neighbor_id: weight} for O(1) lookups
            int_adjacencies[t_id][d] = {tile_to_id[n]: w for n, w in neighbors if n in tile_to_id}
            
    # Translate Ratios & Sizes to Ints
    int_ratios = {tile_to_id[k]: v for k, v in ratios.items() if k in tile_to_id}
    int_tile_sizes = {tile_to_id[k]: v for k, v in tile_sizes.items() if k in tile_to_id}

    return int_adjacencies, int_ratios, images, int_tile_sizes, CELL_SIZE, tile_to_id, id_to_tile


DIR_MAP = {0: "top", 1: "bottom", 2: "left", 3: "right"}


def get_all_tile_names(adjacencies, id_to_tile):
    pad_id = next(k for k, v in id_to_tile.items() if v == "P")
    return [k for k in adjacencies.keys() if k != pad_id]

def inject_background_rules(adjacencies, ratios):
    if "B" not in adjacencies:
        adjacencies["B"] = { "top": [], "bottom": [], "left": [], "right": [] }
    if "P" not in adjacencies:
        adjacencies["P"] = { "top": [], "bottom": [], "left": [], "right": [] }

    inverse_dir = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    
    sky_to_sky_weight = float(ratios.get("B", 100.0))
    pad_weight = 1.0

    for d in ["top", "bottom", "left", "right"]:
        if ["B", sky_to_sky_weight] not in adjacencies["B"][d]:
             adjacencies["B"][d].append(["B", sky_to_sky_weight])
             
        if ["P", pad_weight] not in adjacencies["B"][d]:
             adjacencies["B"][d].append(["P", pad_weight])
             
        if ["P", pad_weight] not in adjacencies["P"][d]:
             adjacencies["P"][d].append(["P", pad_weight])
             
        if ["B", pad_weight] not in adjacencies["P"][d]:
             adjacencies["P"][d].append(["B", pad_weight])

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
    clone_pattern = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y\d+_x\d+")
    
    for tile_id in playable_tiles:
        tile_name = id_to_tile[tile_id]
        match = clone_pattern.match(tile_name)
        if match:
            base_name = match.group(1)
            raw_groups[base_name].add(tile_id)
            
    blueprints = {}
    tile_meta = {}
    
    for base_name, pieces in raw_groups.items():
        visited = {}
        start_piece = next(iter(pieces))
        queue = [(start_piece, 0, 0)]
        visited[start_piece] = (0, 0)
        
        while queue:
            curr_id, cy, cx = queue.pop(0)
            
            for d_key, dy, dx in [("top", -1, 0), ("bottom", 1, 0), ("left", 0, -1), ("right", 0, 1)]:
                allowed = adjacencies.get(curr_id, {}).get(d_key, {})
                for n_id in allowed: # Iterates over dictionary keys
                    if n_id in pieces and n_id not in visited:
                        visited[n_id] = (cy + dy, cx + dx)
                        queue.append((n_id, cy + dy, cx + dx))
                        
        name_off = {}
        for p in pieces:
            m = re.search(r"_y(\d+)_x(\d+)", id_to_tile[p])
            name_off[p] = (int(m.group(1)), int(m.group(2)))

        visited = {}
        remaining = set(pieces)
        while remaining:
            # deterministic: start each component at its top-left-most piece
            start = min(remaining, key=lambda p: name_off[p])
            local = {start: (0, 0)}
            queue = [start]
            while queue:
                curr = queue.pop(0)
                cy, cx = local[curr]
                for d_key, dy, dx in [("top",-1,0),("bottom",1,0),("left",0,-1),("right",0,1)]:
                    for n_id in adjacencies.get(curr, {}).get(d_key, {}):
                        if n_id in remaining and n_id not in local:
                            local[n_id] = (cy + dy, cx + dx)
                            queue.append(n_id)
            sy, sx = name_off[start]                 # anchor this component
            for p, (cy, cx) in local.items():
                visited[p] = (cy + sy, cx + sx)
            remaining -= set(local)
                
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

    floor_only, roof_only = set(), set()
    for t, dirs in adjacencies.items():
        b = dirs.get("bottom", {})
        if len(b) == 1 and pad_id in b:
            floor_only.add(t)
        tp = dirs.get("top", {})
        if len(tp) == 1 and pad_id in tp:
            roof_only.add(t)

    # clone pieces grouped by their legal placement window
    y_groups, x_groups = defaultdict(set), defaultdict(set)
    for t, meta in tile_meta.items():
        bp = blueprints[meta['base_name']]
        y_groups[(meta['local_y'], bp['max_y'])].add(t)
        x_groups[(meta['local_x'], bp['max_x'])].add(t)

    x_rm = {}
    for x in range(1, playable_width + 1):
        rm = set()
        for (lx, mx), tiles in x_groups.items():
            left_x = x - lx
            if left_x < 1 or left_x + mx > playable_width:
                rm |= tiles
        x_rm[x] = rm

    n_full = len(grid[1][1]) if playable_height >= 1 and playable_width >= 1 else 0

    for y in range(1, playable_height + 1):
        row_rm = set()
        if y < playable_height:
            row_rm |= floor_only
        if y > 1:
            row_rm |= roof_only
        for (ly, my), tiles in y_groups.items():
            top_y = y - ly
            if top_y < 1 or top_y + my > playable_height:
                row_rm |= tiles

        base = None
        for x in range(1, playable_width + 1):
            cell = grid[y][x]
            if x_rm[x] or len(cell) != n_full:
                grid[y][x] = cell - row_rm - x_rm[x]
            else:
                if base is None:
                    base = frozenset(cell - row_rm)   # interior columns share a domain
                grid[y][x] = set(base)
    return grid


def get_min_entropy_cell(grid, heap):
    while heap:
        entropy, _, y, x = heapq.heappop(heap)
        
        cell = grid[y][x]
        if not isinstance(cell, set):
            continue 
            
        actual_entropy = len(cell)
        
        if actual_entropy == 0:
            return (-1, -1) 
            
        if actual_entropy == entropy:
            return (y, x)
            
    return None 

def collapse_cell(grid, y, x, adjacencies, cache, tile_sizes, ratios, history, steps, state, pad_id, blueprints, tile_meta, b_id):
    possible_tiles = list(grid[y][x])    
    if not possible_tiles: 
        return []

    playable_height = len(grid) - 2
    playable_width = len(grid[0]) - 2
    total_playable_cells = playable_height * playable_width
    total_ratio_sum = state['ratio_sum']

    progress_ratio = state['cells_done'] / total_playable_cells if total_playable_cells > 0 else 0.0

    if history is not None:
        for t in ratios.keys():
            desired_total = (ratios.get(t, 1.0) / total_ratio_sum) * total_playable_cells
            history[t]['actual'].append(state['current'].get(t, 0))
            history[t]['ideal'].append(desired_total * progress_ratio)
    
    steps.append(state['collapsed_count'])

    weights = []
    legal_indices = []

    for i, tile in enumerate(possible_tiles):
        if tile == pad_id:
            continue

        is_complex = tile in tile_meta
        is_valid_blueprint = True
        
        if is_complex:
            meta = tile_meta[tile]
            base_name = meta['base_name']
            top_y = y - meta['local_y']
            left_x = x - meta['local_x']
            bottom_y = top_y + blueprints[base_name]['max_y']
            right_x = left_x + blueprints[base_name]['max_x']
            
            if top_y < 1 or left_x < 1 or bottom_y >= len(grid) - 1 or right_x >= len(grid[0]) - 1:
                continue 

            blueprint = blueprints[base_name]['pieces']
            
            for piece in blueprint:
                gy = y - meta['local_y'] + piece['local_y']
                gx = x - meta['local_x'] + piece['local_x']
                
                if not (1 <= gy < len(grid) - 1 and 1 <= gx < len(grid[0]) - 1):
                    is_valid_blueprint = False
                    break
                    
                cell_state = grid[gy][gx]
                if isinstance(cell_state, set):
                    if piece['id'] not in cell_state:
                        is_valid_blueprint = False
                        break
                else:
                    if cell_state != piece['id']:
                        is_valid_blueprint = False
                        break
                        
        if not is_valid_blueprint:
            continue

        directional_probs = []
        is_illegal = False

        for dy, dx, d_key in [(-1, 0, "top"), (1, 0, "bottom"), (0, -1, "left"), (0, 1, "right")]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
                neighbor_val = grid[ny][nx]
                if not isinstance(neighbor_val, set):
                    allowed = adjacencies.get(tile, {}).get(d_key, {})
                    
                    n_weight = allowed.get(neighbor_val, 0.0)

                    if n_weight <= 0.0:
                        is_illegal = True
                        break

                    total_allowed_weight = cache.dir_total[d_key].get(tile, 0.0)
                    prob = n_weight / total_allowed_weight if total_allowed_weight > 0 else 0
                    directional_probs.append(prob)

        if is_illegal:
            continue
            
        legal_indices.append(i)

        base_weight = ratios.get(tile, 1.0)
        base_ratio_weight = base_weight / total_ratio_sum
        local_w = base_ratio_weight
        for p in directional_probs:
            local_w *= p

        allowed_bottoms = adjacencies.get(tile, {}).get("bottom", {})
        allowed_tops = adjacencies.get(tile, {}).get("top", {})
        
        valid_bottoms = allowed_bottoms.keys()
        valid_tops = allowed_tops.keys()

        is_against_floor = (y == len(grid) - 2)
        is_against_roof = (y == 1)
        
        acts_as_floor = is_against_floor and (pad_id in valid_bottoms) and tile != b_id
        acts_as_roof = is_against_roof and (pad_id in valid_tops) and tile != b_id

        is_structural = acts_as_floor or acts_as_roof
        
        pacing_multiplier = 1.0
        
        if not is_structural and tile in ratios and tile != b_id:
            desired_total_count = (base_weight / total_ratio_sum) * total_playable_cells
            actual_count = state['current'].get(tile, 0)
            
            if actual_count >= desired_total_count:
                pacing_multiplier = 0.0
            else:
                ideal_current_count = desired_total_count * progress_ratio
                deficit = ideal_current_count - actual_count
                
                if deficit < 0:
                    pacing_multiplier = math.exp(deficit * 5) 
                else:
                    percent_behind = deficit / desired_total_count if desired_total_count > 0 else 0
                    pacing_multiplier = 1.0 + min(0.25, percent_behind * 0.4)

        if is_complex and progress_ratio < 0.15:
            throttle = max(0.01, progress_ratio / 0.15)
            pacing_multiplier *= throttle
            
        final_weight = local_w * pacing_multiplier
        weights.append(final_weight)
        
    total_weight = sum(weights)
    
    if total_weight > 0:
        norm_weights = [w / total_weight for w in weights]
        legal_tiles = [possible_tiles[idx] for idx in legal_indices]
        chosen_tile = random.choices(legal_tiles, weights=norm_weights, k=1)[0]
    else:
        if not legal_indices:
            grid[y][x] = set()
            return []
            
        legal_tiles = [possible_tiles[idx] for idx in legal_indices]
        if b_id is not None and b_id in legal_tiles:
            chosen_tile = b_id
        else:
            fallback_weights = [1.0 / len(legal_tiles)] * len(legal_tiles)
            chosen_tile = random.choices(legal_tiles, weights=fallback_weights, k=1)[0]
    
    collapsed_coords = []
    
    if chosen_tile in tile_meta:
        meta = tile_meta[chosen_tile]
        blueprint = blueprints[meta['base_name']]['pieces']
        
        for piece in blueprint:
            global_y = y - meta['local_y'] + piece['local_y']
            global_x = x - meta['local_x'] + piece['local_x']
            
            if isinstance(grid[global_y][global_x], set):
                grid[global_y][global_x] = piece['id']
                collapsed_coords.append((global_y, global_x))
                state['cells_done'] += 1
                
                if piece['id'] in state['current']:
                    state['current'][piece['id']] += 1
                    state['collapsed_count'] += 1
                    
                if global_y == 1 and piece['id'] in state['roof']:
                    state['roof'][piece['id']] += 1
                if global_y == len(grid) - 2 and piece['id'] in state['floor']:
                    state['floor'][piece['id']] += 1
    else:
        if isinstance(grid[y][x], set):
            grid[y][x] = chosen_tile
            collapsed_coords.append((y, x))
            state['cells_done'] += 1 
            
            if chosen_tile in state['current']:
                state['current'][chosen_tile] += 1
                state['collapsed_count'] += 1
                
            if y == 1 and chosen_tile in state['roof']:
                state['roof'][chosen_tile] += 1
            if y == len(grid) - 2 and chosen_tile in state['floor']:
                state['floor'][chosen_tile] += 1
                
    return collapsed_coords

class AdjCache:
    __slots__ = ("allowed", "dir_total", "_c", "hits", "misses", "cap")

    def __init__(self, adjacencies, cap=200000):
        self.allowed, self.dir_total = {}, {}
        for d in ("top", "bottom", "left", "right"):
            self.allowed[d] = {t: frozenset(dirs.get(d, {})) for t, dirs in adjacencies.items()}
            self.dir_total[d] = {t: (sum(dirs.get(d, {}).values()) or 0.0)
                                 for t, dirs in adjacencies.items()}
        self._c = {d: {} for d in ("top", "bottom", "left", "right")}
        self.hits = self.misses = 0
        self.cap = cap

    def union(self, dir_key, domain):
        c = self._c[dir_key]
        key = frozenset(domain)
        u = c.get(key)
        if u is not None:
            self.hits += 1
            return u
        self.misses += 1
        a = self.allowed[dir_key]
        u = frozenset().union(*[a[t] for t in key if t in a]) if key else frozenset()
        if len(c) >= self.cap:
            c.clear()          # bound memory; hit rate barely moves
        c[key] = u
        return u

    
def propagate(grid, y, x, adjacencies, tile_sizes, heap, cache, tiebreak, timeout_check_callback=None):
    stack = [(y, x)]
    in_stack = {(y, x)}
    H = len(grid)           
    W = len(grid[0])
    ticks = 0
    while stack:
        ticks += 1
        if timeout_check_callback and (ticks & 511) == 0:
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
        
        for nx in range(max(0, x_left), min(W, x_right + 1)):
            for ny, direction in ((y_top - 1, 0), (y_bottom + 1, 1)):
                if 0 <= ny < H and isinstance(grid[ny][nx], set):
                    if update_cell(grid, ny, nx, cy, cx, direction, adjacencies, heap, cache, tiebreak):
                        if (ny, nx) not in in_stack:
                            stack.append((ny, nx))
                            in_stack.add((ny, nx))
        for ny in range(max(0, y_top), min(H, y_bottom + 1)):
            for nx, direction in ((x_left - 1, 2), (x_right + 1, 3)):
                if 0 <= nx < W and isinstance(grid[ny][nx], set):
                    if update_cell(grid, ny, nx, cy, cx, direction, adjacencies, heap, cache, tiebreak):
                        if (ny, nx) not in in_stack:
                            stack.append((ny, nx))
                            in_stack.add((ny, nx))

def update_cell(grid, target_y, target_x, source_y, source_x, direction, adjacencies, heap, cache, tiebreak):
    target = grid[target_y][target_x]
    original_len = len(target)

    dir_key = DIR_MAP.get(direction)
    if not dir_key:
        return False

    source_contents = grid[source_y][source_x]
    if isinstance(source_contents, set):
        valid_targets = cache.union(dir_key, source_contents)
    else:
        valid_targets = cache.allowed[dir_key].get(source_contents) or frozenset()

    target &= valid_targets                 # in-place, no new set allocated

    new_len = len(target)
    if new_len < original_len:
        heapq.heappush(heap, (new_len, tiebreak[target_y][target_x], target_y, target_x))
        return True
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


class TimeoutException(Exception):
    def __init__(self, grid):
        self.grid = grid

class RetryException(Exception):
    def __init__(self, grid):
        self.grid = grid

class MaxRetriesException(Exception):
    def __init__(self, grid):
        self.grid = grid


def generate_level(height, width, adjacencies, ratios, tile_sizes, timeout, pad_id, playable_tiles, b_id, frames=False): 
    global_start_time = time.time()
    blueprints, tile_meta = analyze_complex_tiles(playable_tiles, id_to_tile, adjacencies)
    cache = AdjCache(adjacencies)
    ratio_sum = sum(ratios.values()) or 1
    
    def check_timeout(current_grid):
        if time.time() - global_start_time > timeout:
            raise TimeoutException(current_grid)
            
    def generate_level_attempt(h, w):
            grid = initialize_grid(h, w, playable_tiles)
            grid = pad_grid(grid, pad_value=pad_id)

            
            print("Pruning grid...")
            grid = initial_grid_prune(grid, tile_meta, blueprints, adjacencies, pad_id)
            print("Starting...")

            generation_history = {t: {'actual': [], 'ideal': []} for t in ratios.keys()} if frames else None
            collapse_steps = []
            
            placements = 0 
            frame_states = [] 

            state = {
                'current': {t: 0 for t in ratios.keys()},
                'roof': {t: 0 for t in ratios.keys()},
                'floor': {t: 0 for t in ratios.keys()},
                'collapsed_count': 0,
                'cells_done': 0,
                'ratio_sum': ratio_sum
            }
            playable_height = len(grid) - 2 
            playable_width = len(grid[0]) - 2 

            tiebreak = [[random.random() for _ in range(len(grid[0]))]
                        for _ in range(len(grid))]
            # --- Initialize Priority Queue ---
            heap = []
            for y in range(1, playable_height + 1):
                for x in range(1, playable_width + 1):
                    if isinstance(grid[y][x], set):
                        heapq.heappush(heap, (len(grid[y][x]), random.random(), y, x))
            
            h_len = len(grid)
            w_len = len(grid[0])
            for x in range(w_len):
                if grid[0][x] == pad_id: 
                    propagate(grid, 0,       x, adjacencies, tile_sizes, heap, cache, tiebreak, check_timeout)       
                if grid[h_len-1][x] == pad_id: 
                    propagate(grid, h_len-1, x, adjacencies, tile_sizes, heap, cache, tiebreak, check_timeout)

            while True:
                check_timeout(grid)
                cell = get_min_entropy_cell(grid, heap)

                if cell == (-1, -1):
                    raise RetryException(grid)

                if cell is None:
                    break 
                
                y, x = cell
                if not isinstance(grid[y][x], set): continue
                    
                collapsed_coords = collapse_cell(grid, y, x, adjacencies, cache, tile_sizes, ratios, generation_history, collapse_steps, state, pad_id, blueprints, tile_meta, b_id)
                
                if collapsed_coords:
                    for cy, cx in collapsed_coords:
                        propagate(grid, cy,     cx, adjacencies, tile_sizes, heap, cache, tiebreak, check_timeout)
                    placements += 1

                    if frames and placements % 50 == 0:
                        frame_states.append(copy.deepcopy(grid))
                else:
                    print(f"Dead end at {y},{x}. Restarting...")
                    raise RetryException(grid)
            
            return grid, generation_history, collapse_steps, frame_states

    attempts = 0
    while True:
        attempts += 1
        try:
            grid, history, steps, frames_list = generate_level_attempt(height, width)
            return grid, history, steps, frames_list, attempts
        except RetryException as e:
            if attempts >= 10:
                print(f"Encountered 10 dead ends. Aborting this generation...")
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
    draw = None
    if max_entropy > 0:
        draw = ImageDraw.Draw(heatmap_overlay)
        
    covered_cells = set()
    clone_pattern = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+)_y\d+_x\d+")

    for y in range(rows - 1, -1, -1):
        for x in range(cols):
            cell_val = grid[y][x]

            if isinstance(cell_val, set): 
                if draw and len(cell_val) > 0:
                    entropy = len(cell_val)
                    normalized = min(1.0, max(0.0, (entropy - 1) / (max_entropy - 1))) if max_entropy > 1 else 0
                    r = int(255 * (1 - normalized))
                    b = int(255 * normalized)
                    alpha = int(40 + 160 * (1 - normalized)) 
                    
                    px = x * cw
                    py = y * ch
                    draw.rectangle([px, py, px + cw, py + ch], fill=(r, 0, b, alpha))
                continue 

            tile_name = id_to_tile.get(cell_val, str(cell_val))

            if tile_name in ["B", "P"]: continue
            if (x, y) in covered_cells: continue

            image_key = tile_name
            match = clone_pattern.match(tile_name)
            if match:
                image_key = f"{match.group(2)}.png"

            if image_key not in images: continue

            img = images[image_key]
            height_in_cells, width_in_cells = tile_sizes.get(image_key, (1, 1))

            px = x * cw
            py = (y + 1) * ch - img.height

            canvas.alpha_composite(img, dest=(px, py))

            for dy in range(height_in_cells):
                for dx in range(width_in_cells):
                    target_x = x + dx
                    target_y = y - dy 

                    if target_x >= cols or target_y < 0:
                        continue

                    if not isinstance(grid[target_y][target_x], set) and grid[target_y][target_x] == cell_val:
                        covered_cells.add((target_x, target_y))

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
                    try:
                        parts = name.split("_tile_")
                        base = parts[0].split("_seg_")[1]
                        coords = parts[1].split("_y")[1] 
                        row_str.append(f"m{base}(y{coords})")
                    except:
                        row_str.append(name[:12])
                else:
                    row_str.append(name.replace("tile_", "t_").replace(".png", ""))
        
        formatted_row = " | ".join(f"{s:^14}" for s in row_str)
        lines.append(formatted_row)
        
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


# --- REFACTORED GENERATION LOOP ---
def run_generation(root_dir, grid_width, grid_height, num_levels, timeout=2000, filename="eval.png", merge_data=False, frames=False):
    """
    Executes the WFC generation process. 
    Catches failures per attempt so partial batches can be analyzed instead of abandoning the folder.
    """
    level_folders = []
    generated_file_paths = []
    
    if os.path.exists(root_dir):
        if os.path.exists(os.path.join(root_dir, "adjacency_rules.txt")):
            level_folders.append(root_dir)
        else:
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if item in ("merged_data", "Generations"):
                    continue
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "adjacency_rules.txt")):
                    level_folders.append(item_path)

    if merge_data and len(level_folders) > 1:
        merged_output_dir = os.path.join(root_dir, "merged_data")
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
        
        # Load data ONLY ONCE per folder to save IO time
        global id_to_tile
        adj, ratios, imgs, sizes, c_size, tile_to_id, id_to_tile = load_data(RULES_FILE, RATIOS_FILE, TILES_DIR) 
        
        b_id = tile_to_id.get("B")
        pad_id = tile_to_id["P"]
        playable_tiles = get_all_tile_names(adj, id_to_tile)
        
        failed_count = 0
        success_count = 0

        for level_idx in range(num_levels):
            final_path = base_output_path
            counter = level_idx + 1 # Better explicit counter mapping
            
            while os.path.exists(final_path):
                new_filename = f"{name}{counter}{extension}"
                final_path = os.path.join(folder, new_filename)
                counter += 1

            current_run_name = f"{name}{counter if counter > 1 else ''}"
            
            try:
                start_t = time.time()
                final_grid_data, gen_history, col_steps, frame_states, attempts = generate_level(
                    grid_height, grid_width, adj, ratios, sizes, timeout, pad_id, playable_tiles, b_id, frames
                )
                
                if frames and frame_states:
                    print(f"Level generated successfully! Rendering {len(frame_states)} frames with heatmap...")
                    for idx, state_grid in enumerate(frame_states):
                        step_num = (idx + 1) * 50
                        temp_img = render_grid(state_grid, imgs, sizes, c_size, id_to_tile, max_entropy=len(playable_tiles))
                        frame_filename = f"{current_run_name}_step_{step_num:04d}.png"
                        frame_path = os.path.join(folder, frame_filename)
                        temp_img.save(frame_path)

                print(f"Rendering final image for attempt {level_idx + 1}...")
                final_img = render_grid(final_grid_data, imgs, sizes, c_size, id_to_tile, max_entropy = len(playable_tiles))
                
                print(f"Saving to {final_path}...")
                final_img.save(final_path)
                t_taken = time.time() - start_t
                generated_file_paths.append({
                    'path': final_path,
                    'grid': final_grid_data,
                    'time': t_taken,
                    'retries': attempts - 1, 
                    'id_to_tile': id_to_tile 
                })
                print(f"Done processing attempt {level_idx + 1}!")
                success_count += 1

            except TimeoutException as e:
                print(f"Generation timed out on attempt {level_idx + 1}. Saving debug image and recording failure...")
                debug_img = render_grid(e.grid, imgs, sizes, c_size, id_to_tile, max_entropy=len(playable_tiles))
                debug_path = os.path.join(folder, f"failed_timeout_{name}{counter}{extension}")
                debug_img.save(debug_path)
                failed_count += 1
                
            except MaxRetriesException as e:
                print(f"Max retries (10) reached on attempt {level_idx + 1}. Saving debug image and recording failure...")
                debug_img = render_grid(e.grid, imgs, sizes, c_size, id_to_tile, max_entropy=len(playable_tiles))
                debug_path = os.path.join(folder, f"failed_deadend_{name}{counter}{extension}")
                debug_img.save(debug_path)
                txt_path = os.path.join(folder, f"failed_deadend_{name}{counter}.txt")
                save_readable_debug_grid(e.grid, id_to_tile, txt_path)
                failed_count += 1
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Unexpected error in {base_dir} on attempt {level_idx + 1}: {e}")
                failed_count += 1

        print(f"\n--- Finished Processing {base_dir} ---")
        print(f"Successful Levels: {success_count}/{num_levels}")
        print(f"Failed Levels: {failed_count}/{num_levels}")

    return generated_file_paths

if __name__ == "__main__":
    test_dir = os.path.join("Generation", "1MixedSizedTest")
    run_generation(test_dir, grid_width=100, grid_height=14, num_levels=2)