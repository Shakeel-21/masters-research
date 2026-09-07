"""
complex_structure_emergence.py
==============================
Do the complex (multi-cell) macros produce structures that the core-tile
generator never builds on its own?

WHERE THE STRUCTURE LIST COMES FROM
-----------------------------------
From the complex dataset's DECLARED VOCABULARY - adjacency_rules.txt (and
cross-checked against ratios.json), not from what happened to appear in the
generated output. So the analysis covers every macro the solver was allowed to
place, including ones it never placed. A macro that appears in neither batch is
still counted as "did not emerge in Core", which is the correct treatment.

HOW A MACRO BECOMES A PATTERN
-----------------------------
A complex clone id has the form

    <macro>_<core_tile>_y<row>_x<col>

so each cell of a macro already names the CORE tile occupying it. A macro is
therefore a 2D arrangement of core-tile names - an n-gram with the macro's own
shape rather than a fixed window. Both batches fold into the same alphabet
(core tile names plus "B"), so the macros can be searched for in Core output.

No tile classes, no semantic labels and no training corpus are required. The
only assumptions are the clone naming convention and that "B" is background.

TWO LAYOUT FRAMES
-----------------
layout="name" (default)
    Offsets are read from the clone name, i.e. the tile's position in the
    source level segment. This is the structure as extracted from the original
    Mario level - the thing you actually want to ask about. Cells of the
    bounding box with no mapped tile are wildcards, so partially-mapped and
    non-rectangular macros match correctly.

layout="blueprint"
    Replicates the solver's own reconstruction (breadth-first walk over the
    adjacency graph, falling back to name offsets for pieces the walk cannot
    reach). This is the geometry that actually gets stamped during generation.

They are not always the same, and `layout_diagnostics` reports where they
disagree - see the note in that function.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

CLONE_RE = re.compile(r"(.+?)_((?:split_[LR]_)?tile_.+?)_y(\d+)_x(\d+)$")
TAIL_RE = re.compile(r"_y(\d+)_x(\d+)$")
SPLIT_RE = re.compile(r"^split_[LR]_")

BACKGROUND = "B"
PADDING = "P"
UNCOLLAPSED = "UNCOLLAPSED"
WILDCARD = None

# Same convention as the solver: "top" is the row above.
DIRS = (("top", -1, 0), ("bottom", 1, 0), ("left", 0, -1), ("right", 0, 1))


# --------------------------------------------------------------------------
# Folding grids into a shared alphabet
# --------------------------------------------------------------------------

def fold_name(raw_name, strip_split=True):
    """
    Reduce any tile id to the core-tile alphabet.

      "tile_00012.png"                       -> "tile_00012.png"
      "mario-2-1_seg_12_tile_00012_y1_x0"    -> "tile_00012.png"
      "B" / "P"                              -> unchanged
    """
    m = CLONE_RE.match(raw_name)
    core = m.group(2) if m else raw_name
    if strip_split:
        core = SPLIT_RE.sub("", core)
    if core in (BACKGROUND, PADDING, UNCOLLAPSED):
        return core
    return core if core.endswith(".png") else f"{core}.png"


def fold_grid(grid, id_to_tile, strip_split=True):
    """WFC grid -> 2D object array of core-tile names, padding ring removed."""
    h, w = len(grid), len(grid[0])
    out = np.empty((h, w), dtype=object)
    for y in range(h):
        row = grid[y]
        for x in range(w):
            cell = row[x]
            if isinstance(cell, (set, frozenset)):
                out[y, x] = UNCOLLAPSED
            else:
                out[y, x] = fold_name(id_to_tile.get(cell, str(cell)), strip_split)
    pad = out == PADDING
    keep_r, keep_c = ~pad.all(axis=1), ~pad.all(axis=0)
    if keep_r.any() and keep_c.any():
        out = out[np.ix_(keep_r, keep_c)]
    return out


# --------------------------------------------------------------------------
# Macro templates
# --------------------------------------------------------------------------

def _read_vocab(dataset_dir, rules_file="adjacency_rules.txt"):
    with open(os.path.join(dataset_dir, rules_file)) as f:
        return json.load(f)


def _macro_groups(vocab, strip_split=True):
    """macro name -> {clone id: core tile name}"""
    groups = defaultdict(dict)
    for key in vocab:
        m = CLONE_RE.match(key)
        if not m:
            continue
        core = m.group(2)
        if strip_split:
            core = SPLIT_RE.sub("", core)
        groups[m.group(1)][key] = core if core.endswith(".png") else f"{core}.png"
    return groups


def _name_offsets(pieces):
    out = {}
    for p in pieces:
        m = TAIL_RE.search(p)
        out[p] = (int(m.group(1)), int(m.group(2)))
    return out


def _blueprint_offsets(pieces, vocab, fallback="components"):
    """
    Reconstruct a macro's cell layout, mirroring the solver.

    The walk over the adjacency graph is what makes a macro robust to an
    unsupervised segmentor: the layout is derived FROM the learned contact
    relations, so the blueprint cannot contradict the rules and become
    unplaceable. Gaps left by a missed or mis-cut tile get closed.

    The three fallbacks differ only in how pieces the walk cannot reach are
    positioned:

      "components" - each disconnected group is walked separately and then
                     anchored at its top-left-most piece's source offset.
                     Rule-consistent within a group, source-faithful between
                     groups. Deterministic. This matches the fixed solver.
      "anchored"   - single walk; unreachable pieces take their source offset
                     expressed relative to the start piece's own source offset.
      "as_solver"  - single walk; unreachable pieces take their RAW source
                     offset, mixing an absolute frame into a start-relative
                     one. This is the original behaviour and is retained so the
                     audit can report which macros it distorted.

    Returns (offsets, unreached_pieces, start_piece).
    """
    order = sorted(pieces)
    names = _name_offsets(pieces)

    def walk(seed, pool):
        local = {seed: (0, 0)}
        queue = [seed]
        while queue:
            cur = queue.pop(0)
            cy, cx = local[cur]
            rules = vocab.get(cur, {})
            for d, dy, dx in DIRS:
                for entry in rules.get(d, []):
                    n = entry[0] if isinstance(entry, (list, tuple)) else entry
                    if n in pool and n not in local:
                        local[n] = (cy + dy, cx + dx)
                        queue.append(n)
        return local

    if fallback == "components":
        visited = {}
        remaining = set(pieces)
        first_start = None
        while remaining:
            seed = min(remaining, key=lambda p: (names[p], p))
            if first_start is None:
                first_start = seed
            group = walk(seed, remaining)
            sy, sx = names[seed]
            for p, (cy, cx) in group.items():
                visited[p] = (cy + sy, cx + sx)
            remaining -= set(group)
        # "unreached" here means pieces outside the first component
        first_group = walk(first_start, set(pieces))
        unreached = [p for p in order if p not in first_group]
        return visited, unreached, first_start

    start = order[0]
    visited = walk(start, set(pieces))
    unreached = [p for p in order if p not in visited]
    if fallback == "anchored":
        sy, sx = names[start]
        for p in unreached:
            visited[p] = (names[p][0] - sy, names[p][1] - sx)
    else:
        for p in unreached:
            visited[p] = names[p]
    return visited, unreached, start


def _classify_holes(cells, h, w):
    """
    Split the unspecified cells of a macro's bounding box into interior and
    peripheral holes.

    A hole is interior if specified cells sit on both sides of it, either
    horizontally or vertically. That matters for placement: the pieces flanking
    an interior hole each learned a rule about what sits next to them, so the
    hole has to be filled by something those rules allow. If the tile that used
    to be there was missed by the segmentor, no such tile exists in the
    vocabulary and the macro can never be stamped. A peripheral hole has no
    flanking pieces, so nothing constrains it.
    """
    interior = peripheral = 0
    for y in range(h):
        for x in range(w):
            if (y, x) in cells:
                continue
            row = [c for (cy, cx), _ in cells.items() if cy == y for c in (cx,)]
            col = [c for (cy, cx), _ in cells.items() if cx == x for c in (cy,)]
            h_flank = any(c < x for c in row) and any(c > x for c in row)
            v_flank = any(c < y for c in col) and any(c > y for c in col)
            if h_flank or v_flank:
                interior += 1
            else:
                peripheral += 1
    return interior, peripheral


def _to_template(offsets, core_of):
    """
    Normalise offsets to origin and build a sparse {(dy, dx): core_tile} map.
    Offsets claimed by more than one piece become wildcards, because which
    piece wins cannot be determined from the ruleset alone.
    """
    min_y = min(y for y, _ in offsets.values())
    min_x = min(x for _, x in offsets.values())
    claims = defaultdict(set)
    for piece, (y, x) in offsets.items():
        claims[(y - min_y, x - min_x)].add(core_of[piece])

    cells, collisions = {}, 0
    for off, names in claims.items():
        if len(names) == 1:
            cells[off] = next(iter(names))
        else:
            collisions += 1          # left as a wildcard
    h = max(y for y, _ in claims) + 1
    w = max(x for _, x in claims) + 1
    return cells, h, w, collisions, len(claims)


def load_macro_templates(dataset_dir, layout="blueprint", strip_split=True,
                         rules_file="adjacency_rules.txt", fallback="components",
                         verbose=True):
    """
    Build one template per macro from the declared vocabulary.

    Returns {macro: {cells, h, w, size, footprint, collisions, signature}}
    where `cells` is sparse - missing offsets are wildcards.
    """
    if layout not in ("name", "blueprint"):
        raise ValueError("layout must be 'name' or 'blueprint'")

    vocab = _read_vocab(dataset_dir, rules_file)
    groups = _macro_groups(vocab, strip_split)

    templates = {}
    total_collisions = 0
    for macro, core_of in groups.items():
        pieces = set(core_of)
        if layout == "name":
            offsets = _name_offsets(pieces)
        else:
            offsets, _, _ = _blueprint_offsets(pieces, vocab, fallback)
        cells, h, w, collisions, footprint = _to_template(offsets, core_of)
        total_collisions += collisions
        interior, peripheral = _classify_holes(cells, h, w)
        templates[macro] = {
            "cells": cells,
            "h": h,
            "w": w,
            "size": len(cells),
            "footprint": footprint,
            "collisions": collisions,
            "distinct_tiles": len(set(cells.values())),
            "interior_holes": interior,
            "peripheral_holes": peripheral,
            "start_invariant": start_invariant(set(core_of), vocab),
            "signature": canonical_signature(cells),
        }

    if verbose:
        print(f"[emergence] {len(templates)} macros from "
              f"{os.path.basename(dataset_dir)} ({layout} layout)"
              + (f"; {total_collisions} colliding cells treated as wildcards"
                 if total_collisions else ""))
    return templates


def start_invariant(pieces, vocab, cap=40):
    """
    Does the walk produce the same layout from every possible start piece?

    It should. If it does not, the learned rules for that macro are
    geometrically contradictory - two pieces are related in more than one
    direction, or a chain of rules closes into a loop that does not add up - so
    the reconstructed shape depends on traversal order. Returns True/False, or
    None if the macro is too large to check exhaustively.
    """
    if len(pieces) > cap:
        return None
    ref = None
    for seed in sorted(pieces):
        local = {seed: (0, 0)}
        queue = [seed]
        while queue:
            cur = queue.pop(0)
            cy, cx = local[cur]
            for d, dy, dx in DIRS:
                for entry in vocab.get(cur, {}).get(d, []):
                    n = entry[0] if isinstance(entry, (list, tuple)) else entry
                    if n in pieces and n not in local:
                        local[n] = (cy + dy, cx + dx)
                        queue.append(n)
        if len(local) < len(pieces):
            continue                       # disconnected; handled elsewhere
        my = min(y for y, _ in local.values())
        mx = min(x for _, x in local.values())
        norm = tuple(sorted((p, y - my, x - mx) for p, (y, x) in local.items()))
        if ref is None:
            ref = norm
        elif norm != ref:
            return False
    return True


def canonical_signature(cells):
    """Order-independent identity of a template, so duplicate macros collapse."""
    return tuple(sorted((y, x, v) for (y, x), v in cells.items()))


def dedupe_templates(templates):
    by_sig = defaultdict(list)
    for name, t in templates.items():
        by_sig[t["signature"]].append(name)
    return by_sig


def cross_check_ratios(dataset_dir, templates, ratios_file="ratios.json"):
    """
    Confirm the macro list derived from the rules matches the one implied by
    ratios.json. A mismatch means the solver's weighting and its constraints
    disagree about the vocabulary, which is worth knowing about.
    """
    path = os.path.join(dataset_dir, ratios_file)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        ratios = json.load(f)
    in_ratios = set()
    for k in ratios:
        m = CLONE_RE.match(k)
        if m:
            in_ratios.add(m.group(1))
    return {
        "macros_in_rules": len(templates),
        "macros_in_ratios": len(in_ratios),
        "rules_only": sorted(set(templates) - in_ratios),
        "ratios_only": sorted(in_ratios - set(templates)),
    }


def layout_diagnostics(dataset_dir, strip_split=True,
                       rules_file="adjacency_rules.txt"):
    """
    Classify each macro by how its two layout frames relate.

      "agree"       - the walk reproduces the source-segment offsets exactly.
      "compacted"   - the walk reached every piece and closed gaps left by the
                      segmentor. This is the walk doing its job: the gaps are
                      missing or mis-cut tiles, not empty space in the level.
      "frame-mixed" - some pieces were unreachable, so their raw source offsets
                      were combined with walk-relative offsets for the rest.
                      Those are different coordinate frames, so the resulting
                      shape can be distorted by however far the start piece sits
                      from the segment origin. Anchoring the fallback to the
                      start piece's own source offset removes this.
    """
    vocab = _read_vocab(dataset_dir, rules_file)
    groups = _macro_groups(vocab, strip_split)
    rows = []
    for macro, core_of in groups.items():
        pieces = set(core_of)
        name_off = _name_offsets(pieces)
        bp_off, unreached, start = _blueprint_offsets(pieces, vocab, "as_solver")
        anc_off, _, _ = _blueprint_offsets(pieces, vocab, "anchored")
        cmp_off, _, _ = _blueprint_offsets(pieces, vocab, "components")

        def norm(off):
            my = min(y for y, _ in off.values())
            mx = min(x for _, x in off.values())
            return {p: (y - my, x - mx) for p, (y, x) in off.items()}

        n_n, n_b, n_a, n_c = (norm(name_off), norm(bp_off),
                              norm(anc_off), norm(cmp_off))
        # relation describes the CURRENT (component) reconstruction
        kind = "agree" if n_n == n_c else "compacted"
        # legacy_relation describes what the original single-walk code did
        if n_n == n_b:
            legacy = "agree"
        elif not unreached:
            legacy = "compacted"
        else:
            legacy = "frame-mixed"

        def box(o):
            return f"{max(y for y,_ in o.values())+1}x{max(x for _,x in o.values())+1}"

        rows.append({
            "macro": macro,
            "pieces": len(pieces),
            "unreached_by_walk": len(unreached),
            "relation": kind,
            "legacy_relation": legacy,
            "source_box": box(n_n),
            "component_box": box(n_c),
            "legacy_walk_box": box(n_b),
            "fix_changed_shape": int(n_c != n_b),
        })
    order = {"frame-mixed": 0, "compacted": 1, "agree": 2}
    rows.sort(key=lambda r: (order[r["legacy_relation"]], -r["pieces"]))
    return rows


# --------------------------------------------------------------------------
# Exact emergence
# --------------------------------------------------------------------------

def build_position_index(grids):
    """tile name -> [(grid_index, y, x)]. Used to anchor the pattern search."""
    idx = defaultdict(list)
    for gi, g in enumerate(grids):
        H, W = g.shape
        for y in range(H):
            for x in range(W):
                idx[g[y, x]].append((gi, y, x))
    return idx


def count_exact(grids, template, index):
    """
    Occurrences of a template across `grids`, anchored on its rarest required
    tile so cost tracks that tile's frequency rather than grid area.
    Returns (occurrences, levels_containing_at_least_one).
    """
    cells = template["cells"]
    if not cells:
        return 0, 0

    anchor = min(cells, key=lambda off: len(index.get(cells[off], ())))
    ay, ax = anchor
    positions = index.get(cells[anchor], ())
    if not positions:
        return 0, 0

    items = list(cells.items())
    total, hit = 0, set()
    for gi, py, px in positions:
        g = grids[gi]
        H, W = g.shape
        oy, ox = py - ay, px - ax
        if oy < 0 or ox < 0 or oy + template["h"] > H or ox + template["w"] > W:
            continue
        for (dy, dx), name in items:
            if g[oy + dy, ox + dx] != name:
                break
        else:
            total += 1
            hit.add(gi)
    return total, len(hit)


# --------------------------------------------------------------------------
# Partial emergence via sub-windows
# --------------------------------------------------------------------------

def window_counter(grids, k=3, skip_uncollapsed=True):
    c = Counter()
    for g in grids:
        H, W = g.shape
        for y in range(H - k + 1):
            for x in range(W - k + 1):
                win = tuple(g[y:y + k, x:x + k].flatten())
                if skip_uncollapsed and UNCOLLAPSED in win:
                    continue
                c[win] += 1
    return c


def template_subwindows(template, k=3):
    """Every fully-specified kxk sub-block of a macro (wildcards excluded)."""
    cells = template["cells"]
    out = []
    for y in range(template["h"] - k + 1):
        for x in range(template["w"] - k + 1):
            win, complete = [], True
            for dy in range(k):
                for dx in range(k):
                    v = cells.get((y + dy, x + dx))
                    if v is None:
                        complete = False
                        break
                    win.append(v)
                if not complete:
                    break
            if complete:
                out.append(tuple(win))
    return out


def partial_coverage(template, window_counts, k=3):
    """
    Fraction of a macro's kxk sub-blocks that occur in the given window set.

    Full coverage with no exact match means Core builds every local piece of
    the structure but never assembles them in that configuration - a stronger
    and more defensible claim than "this structure is absent".
    """
    subs = template_subwindows(template, k)
    if not subs:
        return float("nan"), 0
    return sum(1 for s in subs if s in window_counts) / len(subs), len(subs)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

SIZE_BUCKETS = ((2, "1-2"), (4, "3-4"), (9, "5-9"), (16, "10-16"), (10 ** 9, "17+"))


def emergence_report(templates, core_grids, complex_grids=None, k_values=(2, 3),
                     out_dir=None, diagnostics=None, ratio_check=None,
                     verbose=True):
    """Full analysis. Returns a dict; optionally writes a CSV and summary."""
    if not templates:
        raise ValueError("no macro templates - check the clone naming "
                         "convention in adjacency_rules.txt")
    if not core_grids:
        raise ValueError("no Core generations to search")

    by_sig = dedupe_templates(templates)
    reps = {names[0]: templates[names[0]] for names in by_sig.values()}

    core_index = build_position_index(core_grids)
    core_wins = {k: window_counter(core_grids, k) for k in k_values}
    core_cells = sum(g.size for g in core_grids)
    comp_index = build_position_index(complex_grids) if complex_grids else None

    rows = []
    for name, t in reps.items():
        n_exact, n_levels = count_exact(core_grids, t, core_index)
        row = {
            "macro": name,
            "duplicate_of": len(by_sig[t["signature"]]) - 1,
            "box_h": t["h"],
            "box_w": t["w"],
            "specified_cells": t["size"],
            "wildcard_cells": t["h"] * t["w"] - t["size"],
            "interior_holes": t.get("interior_holes"),
            "peripheral_holes": t.get("peripheral_holes"),
            "distinct_tiles": t.get("distinct_tiles"),
            "start_invariant": t.get("start_invariant"),
            "colliding_cells": t["collisions"],
            "core_exact_count": n_exact,
            "core_levels_hit": n_levels,
            "core_emerges": int(n_exact > 0),
        }
        for k in k_values:
            cov, n_sub = partial_coverage(t, core_wins[k], k)
            row[f"core_cov_{k}x{k}"] = None if cov != cov else round(cov, 4)
            row[f"n_sub_{k}x{k}"] = n_sub
        if comp_index is not None:
            c_exact, c_levels = count_exact(complex_grids, t, comp_index)
            row["complex_exact_count"] = c_exact
            row["complex_levels_hit"] = c_levels
        rows.append(row)

    rows.sort(key=lambda r: (-r["specified_cells"], r["macro"]))

    n_struct = len(reps)
    emerged = [r for r in rows if r["core_emerges"]]
    placed = [r for r in rows if r.get("complex_exact_count", 0) > 0] if comp_index else []

    S = []
    S.append("=== COMPLEX STRUCTURE EMERGENCE IN CORE GENERATIONS ===")
    S.append(f"Macro definitions in the complex vocabulary: {len(templates)}")
    S.append(f"Distinct structures (after deduplication): {n_struct}")
    S.append(f"Core generations searched: {len(core_grids)} ({core_cells} cells)")
    if ratio_check:
        S.append(f"Vocabulary cross-check: {ratio_check['macros_in_rules']} macros in "
                 f"adjacency_rules, {ratio_check['macros_in_ratios']} in ratios.json"
                 + ("" if not (ratio_check['rules_only'] or ratio_check['ratios_only'])
                    else f"  MISMATCH: rules-only={ratio_check['rules_only']}, "
                         f"ratios-only={ratio_check['ratios_only']}"))

    S.append("")
    S.append("1. EXACT EMERGENCE")
    S.append(f"Structures the Core generator produces unaided: "
             f"{len(emerged)}/{n_struct} ({100*len(emerged)/n_struct:.1f}%)")
    S.append(f"Structures unique to the complex vocabulary: "
             f"{n_struct-len(emerged)}/{n_struct} "
             f"({100*(n_struct-len(emerged))/n_struct:.1f}%)")
    tot = sum(r["core_exact_count"] for r in rows)
    S.append(f"Total emergent occurrences in Core: {tot} "
             f"({1000.0*tot/core_cells:.3f} per 1000 cells)")
    if comp_index:
        S.append(f"Structures actually present in Complex output: "
                 f"{len(placed)}/{n_struct} "
                 f"({100*len(placed)/n_struct:.1f}%) - macros the solver never "
                 f"placed cannot be credited with adding structure")

    uni = [r for r in rows if r.get("distinct_tiles") == 1]
    if uni:
        uni_em = sum(1 for r in uni if r["core_emerges"])
        S.append(f"Of these, {len(uni)} structures are built from a single "
                 f"repeated tile ({uni_em} emerge in Core). A uniform slab is "
                 f"easy for the core generator to reproduce at any size, so "
                 f"read the size breakdown below alongside distinct_tiles.")

    S.append("")
    S.append("Emergence rate by structure size (specified cells):")
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        for lim, lab in SIZE_BUCKETS:
            if r["specified_cells"] <= lim:
                buckets[lab][0] += 1
                buckets[lab][1] += r["core_emerges"]
                break
    for _, lab in SIZE_BUCKETS:
        if lab in buckets:
            n, e = buckets[lab]
            S.append(f"  {lab:>6} cells: {e}/{n} emerge ({100*e/n:.1f}%)")

    inv_bad = [r["macro"] for r in rows if r.get("start_invariant") is False]
    if inv_bad:
        S.append("")
        S.append("Macros whose learned rules give a different shape depending on "
                 "where the walk starts:")
        S.append(f"  {len(inv_bad)} - {', '.join(inv_bad[:8])}")
        S.append("  Their rules are geometrically contradictory, so the stamped "
                 "shape is arbitrary.")

    if comp_index is not None:
        holed = [r for r in rows if (r.get("interior_holes") or 0) > 0]
        solid = [r for r in rows if (r.get("interior_holes") or 0) == 0]
        def placed_rate(g):
            return (sum(1 for r in g if r.get("complex_exact_count", 0) > 0),
                    len(g))
        ph, nh = placed_rate(holed), placed_rate(solid)
        S.append("")
        S.append("Placement vs interior holes (a hole flanked by pieces must be "
                 "fillable by something the rules allow):")
        S.append(f"  with interior holes:    {ph[0]}/{ph[1]} placed at least once")
        S.append(f"  without interior holes: {nh[0]}/{nh[1]} placed at least once")

    S.append("")
    S.append("2. PARTIAL EMERGENCE (is the local texture present in Core?)")
    for k in k_values:
        vals = [r[f"core_cov_{k}x{k}"] for r in rows
                if r[f"core_cov_{k}x{k}"] is not None]
        if not vals:
            S.append(f"  {k}x{k}: no structure is large enough to contain a "
                     f"complete {k}x{k} sub-block")
            continue
        full = sum(1 for v in vals if v >= 0.999)
        zero = sum(1 for v in vals if v <= 0.001)
        assembled = sum(1 for r in rows
                        if r[f"core_cov_{k}x{k}"] is not None
                        and r[f"core_cov_{k}x{k}"] >= 0.999
                        and not r["core_emerges"])
        S.append(f"  {k}x{k}: mean sub-block coverage {float(np.mean(vals)):.3f} "
                 f"over {len(vals)} structures; {full} fully covered, "
                 f"{zero} entirely absent")
        S.append(f"        every {k}x{k} piece present but never assembled: {assembled}")

    novel = {}
    if complex_grids:
        S.append("")
        S.append("3. NOVEL LOCAL STRUCTURE IN COMPLEX OUTPUT")
        for k in k_values:
            comp_w = window_counter(complex_grids, k)
            core_w = core_wins[k]
            comp_total = sum(comp_w.values()) or 1
            new_inst = sum(v for w, v in comp_w.items() if w not in core_w)
            new_dist = sum(1 for w in comp_w if w not in core_w)
            core_only = sum(1 for w in core_w if w not in comp_w)
            novel[k] = {"novel_instance_rate": new_inst / comp_total,
                        "novel_distinct": new_dist,
                        "distinct_complex": len(comp_w),
                        "distinct_core": len(core_w),
                        "core_only_distinct": core_only}
            S.append(f"  {k}x{k}: {new_inst}/{comp_total} window instances "
                     f"({100*new_inst/comp_total:.2f}%) never occur in Core")
            S.append(f"        distinct windows - Complex {len(comp_w)}, "
                     f"Core {len(core_w)}, Complex-only {new_dist}, "
                     f"Core-only {core_only}")

    if diagnostics:
        counts = Counter(d["relation"] for d in diagnostics)
        legacy = Counter(d["legacy_relation"] for d in diagnostics)
        changed = [d for d in diagnostics if d["fix_changed_shape"]]
        S.append("")
        S.append("4. MACRO GEOMETRY AUDIT")
        S.append(f"Current (per-component) reconstruction vs the source offsets: "
                 f"agree {counts.get('agree', 0)}, "
                 f"compacted {counts.get('compacted', 0)} of {len(diagnostics)}.")
        S.append(f"  'compacted' is the walk closing gaps left by the segmentor - "
                 f"expected and intended.")
        S.append(f"Original single-walk reconstruction, for reference: "
                 f"agree {legacy.get('agree', 0)}, "
                 f"compacted {legacy.get('compacted', 0)}, "
                 f"frame-mixed {legacy.get('frame-mixed', 0)}.")
        if changed:
            S.append(f"Macros whose blueprint shape the fix changed: {len(changed)}")
            for d in changed[:12]:
                S.append(f"    {d['macro']:<26} pieces={d['pieces']:<3} "
                         f"outside first component={d['unreached_by_walk']:<3} "
                         f"source {d['source_box']} / now {d['component_box']} / "
                         f"was {d['legacy_walk_box']}")
        else:
            S.append("The fix changes no blueprint shape in this vocabulary.")

    S.append("")
    S.append("Note: unmapped cells inside a macro's bounding box are treated as")
    S.append("wildcards, so matching is permissive. This biases emergence counts")
    S.append("upward, which is the conservative direction for the claim that a")
    S.append("structure does not arise in Core.")

    text = "\n".join(S)
    if verbose:
        print("\n" + text)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        import csv
        with open(os.path.join(out_dir, "complex_structure_emergence.csv"),
                  "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        if diagnostics:
            with open(os.path.join(out_dir, "macro_geometry_audit.csv"),
                      "w", newline="") as f:
                wtr = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
                wtr.writeheader()
                wtr.writerows(diagnostics)
        with open(os.path.join(out_dir, "complex_structure_emergence.txt"), "w") as f:
            f.write(text)

    return {"rows": rows, "summary": text, "novel": novel,
            "n_structures": n_struct, "n_emerged": len(emerged)}