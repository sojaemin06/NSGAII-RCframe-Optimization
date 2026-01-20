import pandas as pd
import numpy as np
import h5py
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def reduce_database(col_csv_path, beam_csv_path, mat_path):
    print(">>> Starting Database Reduction Process...")
    
    # 1. Load Original Data
    col_df = pd.read_csv(col_csv_path)
    beam_df = pd.read_csv(beam_csv_path)
    
    print(f"Original Columns: {len(col_df)}")
    print(f"Original Beams: {len(beam_df)}")
    
    # --- Column Reduction Logic ---
    # Group by Geometry and Material
    # Binning criteria: rho (Reinforcement Ratio) with 0.5% step? Or simply select pareto-optimal within group.
    # Strategy: For same (b, h, fck, fy), sort by Strength (PM_Volume or Pn_max) and Cost.
    # Keep only those that are "Non-dominated" in Cost vs Strength space within the group.
    # Or simpler: Discrete Rho bins (e.g. 1%, 1.5%, 2%...) and pick cheapest.
    
    reduced_col_indices = []
    
    # Grouping keys
    group_keys = ['b', 'h', 'fck', 'fy']
    
    for name, group in col_df.groupby(group_keys):
        # Sort by Cost ascending
        group = group.sort_values('Cost')
        
        # Simple Binning by Rho (step 0.005 = 0.5%)
        # This ensures we have a range of reinforcement ratios for each size
        bins = [0.01 + i*0.005 for i in range(20)] # 1% to 10%
        
        selected_indices = []
        for i in range(len(bins)-1):
            lower, upper = bins[i], bins[i+1]
            # Find candidates in this rho range
            candidates = group[(group['rho'] >= lower) & (group['rho'] < upper)]
            
            if not candidates.empty:
                # Pick the one with lowest Cost (already sorted)
                best_idx = candidates.index[0]
                selected_indices.append(best_idx)
        
        reduced_col_indices.extend(selected_indices)
        
    reduced_col_indices = sorted(list(set(reduced_col_indices)))
    reduced_col_df = col_df.loc[reduced_col_indices].copy()
    
    # --- [NEW] Force Exact Number (Target: 800) ---
    TARGET_COLS = 800
    if len(reduced_col_df) > TARGET_COLS:
        # Sort by Performance (PM_Volume) to ensure range coverage
        if 'PM_Volume' in reduced_col_df.columns:
            reduced_col_df = reduced_col_df.sort_values('PM_Volume')
        else:
            reduced_col_df = reduced_col_df.sort_values(['b', 'h', 'rho'])
            
        # Uniformly sample indices to keep distribution
        selected_indices = np.linspace(0, len(reduced_col_df) - 1, TARGET_COLS, dtype=int)
        reduced_col_df = reduced_col_df.iloc[selected_indices]

    # Re-index 'name' column to 1..N
    old_col_ids = reduced_col_df['name'].values # Keep track for MAT mapping
    reduced_col_df['name'] = range(1, len(reduced_col_df) + 1)
    
    print(f"Reduced Columns: {len(reduced_col_df)} ({(len(reduced_col_df)/len(col_df))*100:.1f}%)")
    
    # --- Beam Reduction Logic ---
    # Beams usually defined by (b, h) and Rho.
    # Similar logic.
    reduced_beam_indices = []
    for name, group in beam_df.groupby(group_keys): # Beam also has b, h, fck, fy
        group = group.sort_values('Cost')
        
        # Beam Rho is usually lower. Step 0.2%?
        # Column 'rho' in Beam CSV? Let's check column names.
        # Beam_generator output: rho_s_T (tension), rho_s_C (compression).
        # CSV likely has 'rho' or similar. Assuming 'rho' exists based on previous checks.
        # If not, use 'N_r' (number of bars) as proxy.
        
        # Check available columns
        rho_col = 'rho' if 'rho' in group.columns else 'N_r' 
        
        if rho_col == 'rho':
             bins = [0.005 + i*0.002 for i in range(20)] # 0.5% to 4.5%, step 0.2%
        else:
             bins = range(2, 20, 1) # Number of bars
             
        if rho_col == 'rho':
            for i in range(len(bins)-1):
                lower, upper = bins[i], bins[i+1]
                candidates = group[(group[rho_col] >= lower) & (group[rho_col] < upper)]
                if not candidates.empty:
                    reduced_beam_indices.append(candidates.index[0])
        else:
             for n_bar in bins:
                candidates = group[group[rho_col] == n_bar]
                if not candidates.empty:
                    reduced_beam_indices.append(candidates.index[0])

    reduced_beam_indices = sorted(list(set(reduced_beam_indices)))
    reduced_beam_df = beam_df.loc[reduced_beam_indices].copy()
    
    # --- [NEW] Force Exact Number (Target: 500) ---
    TARGET_BEAMS = 500
    if len(reduced_beam_df) > TARGET_BEAMS:
        # Sort by Performance (PiM)
        if 'PiM' in reduced_beam_df.columns:
            reduced_beam_df = reduced_beam_df.sort_values('PiM')
        else:
            reduced_beam_df = reduced_beam_df.sort_values(['b', 'h'])
            
        selected_indices = np.linspace(0, len(reduced_beam_df) - 1, TARGET_BEAMS, dtype=int)
        reduced_beam_df = reduced_beam_df.iloc[selected_indices]

    reduced_beam_df['name'] = range(1, len(reduced_beam_df) + 1)
    
    print(f"Reduced Beams: {len(reduced_beam_df)} ({(len(reduced_beam_df)/len(beam_df))*100:.1f}%)")
    
    # --- Save Reduced CSVs ---
    reduced_col_df.to_csv("column_sections_reduced.csv", index=False)
    reduced_beam_df.to_csv("beam_sections_reduced.csv", index=False)
    
    # --- MAT File Reduction (Columns only) ---
    print("Reducing MAT file...")
    if os.path.exists(mat_path):
        with h5py.File(mat_path, 'r') as f_src:
            # Create new MAT file
            with h5py.File("pm_dataset_reduced.mat", 'w') as f_dst:
                # MATLAB cells are usually stored as object references
                # This is tricky with h5py if structure is complex.
                # 'Column_Mdata' is 1xN cell array.
                
                # We need to copy references. 
                # But creating a new cell array in HDF5/MATLAB format via h5py is hard.
                # Easier: Copy ALL data refs, but update the main cell array pointer?
                # Actually, simpler way: Just rely on the index mapping.
                # But we want a smaller file.
                
                # Alternative: Just keep using the OLD MAT file, but use a mapping index.
                # The 'load_section_data' in utils.py can handle the mapping.
                pass
                
    # Since re-writing MATLAB cell arrays in HDF5 via Python is complex and error-prone,
    # we will use an Index Mapping strategy.
    # We will save a mapping file: Reduced_ID -> Original_ID
    
    mapping_df = pd.DataFrame({
        'Reduced_ID': range(1, len(reduced_col_df) + 1),
        'Original_ID': old_col_ids # Assuming 'name' in original CSV matches MAT index (1-based)
    })
    mapping_df.to_csv("column_id_mapping.csv", index=False)
    print("MAT file reduction skipped (complex format). Use 'column_id_mapping.csv' for lookup.")
    print(">>> Reduction Complete.")

if __name__ == "__main__":
    reduce_database(
        "column_sections_simple02.csv", 
        "beam_sections_simple02.csv", 
        "pm_dataset_simple02.mat"
    )
