import math

# =================================================================
# Data from src/config.py
# =================================================================

COLUMN_LOCATIONS_4F = [(0, 0), (5, 0), (10, 0), (15, 0),
                       (0, 6), (5, 6), (10, 6), (15, 6),
                       (0, 10), (5, 10), (10, 10), (15, 10),
                       (5, 15), (10, 15), (15, 15)]

BEAM_CONNECTIONS_4F = [(0, 1), (1, 2), (2, 3),
                       (4, 5), (5, 6), (6, 7),
                       (8, 9), (9, 10), (10, 11),
                       (12, 13), (13, 14),
                       (0, 4), (4, 8),
                       (1, 5), (5, 9), (9, 12),
                       (2, 6), (6, 10), (10, 13),
                       (3, 7), (7, 11), (11, 14)]

# Updated 6F Data (Closed Wings)
COLUMN_LOCATIONS_6F = [
    (0, 0), (6, 0), (10, 0), (17, 0),
    (0, 6), (6, 6), (10, 6), (17, 6),
    (0, 11), (6, 11), (10, 11), (17, 11)
]

BEAM_CONNECTIONS_6F = [
    (0, 1), (1, 2), (2, 3),    
    (4, 5), (5, 6), (6, 7),    
    (8, 9), (10, 11),          
    (0, 4), (4, 8),            
    (1, 5), (5, 9),            
    (2, 6), (6, 10),           
    (3, 7), (7, 11)            
]

COLUMN_LOCATIONS_8F = [
            (5, 0), (11, 0),
    (0, 5), (5, 5), (11, 5), (18, 5),
    (0, 11), (5, 11), (11, 11), (18, 11),
            (5, 18), (11, 18)
]

BEAM_CONNECTIONS_8F = [
    (0, 1),                 
    (2, 3), (3, 4), (4, 5), 
    (6, 7), (7, 8), (8, 9), 
    (10, 11),               
    (2, 6),                 
    (0, 3), (3, 7), (7, 10), 
    (1, 4), (4, 8), (8, 11), 
    (5, 9)                  
]

# =================================================================
# Logic
# =================================================================

def get_bays(col_locs, beam_conns):
    # Map beam index to sorted tuple of points
    beam_lookup = {}
    for idx, (p1, p2) in enumerate(beam_conns):
        beam_lookup[tuple(sorted((p1, p2)))] = idx

    col_indices = {loc: i for i, loc in enumerate(col_locs)}
    xs = sorted(list(set(loc[0] for loc in col_locs)))
    ys = sorted(list(set(loc[1] for loc in col_locs)))

    bays = []
    
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x1, x2 = xs[i], xs[i+1]
            y1, y2 = ys[j], ys[j+1]
            
            p_bl = (x1, y1); p_br = (x2, y1)
            p_tl = (x1, y2); p_tr = (x2, y2)
            
            if not all(p in col_indices for p in [p_bl, p_br, p_tl, p_tr]):
                continue
                
            idx_bl, idx_br = col_indices[p_bl], col_indices[p_br]
            idx_tl, idx_tr = col_indices[p_tl], col_indices[p_tr]
            
            b_bott = tuple(sorted((idx_bl, idx_br)))
            b_top  = tuple(sorted((idx_tl, idx_tr)))
            b_left = tuple(sorted((idx_bl, idx_tl)))
            b_right= tuple(sorted((idx_br, idx_tr)))
            
            if all(b in beam_lookup for b in [b_bott, b_top, b_left, b_right]):
                bay_beams = [beam_lookup[b] for b in [b_bott, b_top, b_left, b_right]]
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                bays.append({'beams': bay_beams, 'centroid': centroid})
    return bays

def get_pattern_beams(bays, pattern_type, max_x, max_y):
    loaded_beams = set()
    
    for bay in bays:
        cx, cy = bay['centroid']
        load_bay = False
        
        # P1: Base Heavy (Y < 2/3 H)
        if pattern_type == 1:
            if cy < (2/3 * max_y):
                load_bay = True
                
        # P2: Right Heavy (X > 1/3 W)
        elif pattern_type == 2:
            if cx > (1/3 * max_x):
                load_bay = True
                
        # P3: T-Shape
        elif pattern_type == 3:
            # Top Third
            if cy > (2/3 * max_y):
                if (max_x/3) < cx < (2/3 * max_x):
                    load_bay = True
            # Mid Third
            elif (max_y/3) < cy < (2/3 * max_y):
                if cx < (2/3 * max_x):
                    load_bay = True
            # Bot Third -> Empty
            
        # P4: Roof Corner (X > 1/3, Y < 2/3)
        elif pattern_type == 4:
            if cx > (max_x/3) and cy < (2 * max_y / 3):
                load_bay = True
                
        if load_bay:
            for b in bay['beams']:
                loaded_beams.add(b)
                
    return sorted(list(loaded_beams))

def process_example(name, col_locs, beam_conns, floors):
    xs = [p[0] for p in col_locs]
    ys = [p[1] for p in col_locs]
    max_x, max_y = max(xs), max(ys)
    
    bays = get_bays(col_locs, beam_conns)
    
    patterns = {}
    
    # Sequence definition
    if floors == 4:
        seq = [1, 2, 3, 4]
    elif floors == 6:
        seq = [1, 2, 3, 1, 2, 4]
    elif floors == 8:
        seq = [1, 2, 3, 1, 2, 3, 1, 4]
    
    for i, p_type in enumerate(seq):
        floor_num = i + 1
        beams = get_pattern_beams(bays, p_type, max_x, max_y)
        patterns[floor_num] = beams
        
    print(f"\n{name} = {{")
    for f, beams in patterns.items():
        print(f"    {f}: {beams},")
    print("}")

print("# Generated Load Patterns")
process_example("LOAD_PATTERNS_4F", COLUMN_LOCATIONS_4F, BEAM_CONNECTIONS_4F, 4)
process_example("LOAD_PATTERNS_6F", COLUMN_LOCATIONS_6F, BEAM_CONNECTIONS_6F, 6)
process_example("LOAD_PATTERNS_8F", COLUMN_LOCATIONS_8F, BEAM_CONNECTIONS_8F, 8)