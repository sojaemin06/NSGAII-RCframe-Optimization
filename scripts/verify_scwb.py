import sys
import os
import h5py
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_section_data, get_pm_capacity_from_df, load_pm_data_for_column

def verify_scwb_logic():
    print("### SCWB Logic Verification ###")
    
    # 1. Load Data
    print("Loading section data...")
    beam_sections_df, column_sections_df, _, _ = load_section_data()
    h5_path = 'pm_dataset_simple02.mat'
    
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found.")
        return

    h5_file = h5py.File(h5_path, 'r')
    
    try:
        # 2. Test Column Capacity at P=0 (Pure Bending)
        # Pick a random column section (e.g., index 0)
        col_idx = 0
        col_name = column_sections_df.iloc[col_idx]['ID'] if 'ID' in column_sections_df.columns else f"Idx {col_idx}"
        print(f"\nTesting Column Section: {col_name} (Index {col_idx})")
        
        pm_df = load_pm_data_for_column(h5_file, col_idx)
        if pm_df.empty:
            print("Error: PM DataFrame is empty.")
            return

        # Test Strong Axis (z)
        pn0_z, mn0_z = get_pm_capacity_from_df(0, pm_df, axis='z')
        print(f"  > Strong Axis (z) at P=0: Pn={pn0_z:.2f}, Mn={mn0_z:.2f}")
        
        # Test Weak Axis (y)
        pn0_y, mn0_y = get_pm_capacity_from_df(0, pm_df, axis='y')
        print(f"  > Weak Axis (y) at P=0:   Pn={pn0_y:.2f}, Mn={mn0_y:.2f}")
        
        if mn0_z == 0 or mn0_y == 0:
            print("  [WARNING] Column moment capacity at P=0 is ZERO. Check utils.get_pm_capacity_from_df logic.")
        else:
            print("  [OK] Column moment capacities are non-zero.")

        # 3. Simulate SCWB Check for a Joint
        print("\nSimulating Joint SCWB Check:")
        # Assume Beam: Index 0
        beam_idx = 0
        beam_mn = beam_sections_df.iloc[beam_idx]['PiM']
        print(f"  > Beam (Index {beam_idx}) Mn: {beam_mn:.2f} kNm")
        
        # Scenario: 1 Column (Continuous) + 2 Beams framing into it in X-direction
        # Sum Mb = 2 * Beam Mn
        sum_mb = 2 * beam_mn
        
        # Sum Mc = 2 * Column Mn (Above + Below) - Unrotated (Strong axis resists Y-moment? No.)
        # In evaluate():
        # X-Beam -> Moment about Global Y
        # Column Unrotated (Rot=0): Local y // Global Y. So resisting moment is Mn_y (Weak Axis).
        # Wait, usually Columns are oriented such that Strong Axis resists larger moments.
        # Let's check logic in evaluate():
        # if rot == 0: sum_mc_for_x_beams += mn0_y
        
        # Case A: Unrotated Column
        sum_mc_unrotated = 2 * mn0_y 
        ratio_unrotated = (1.2 * sum_mb) / (sum_mc_unrotated + 1e-9)
        print(f"  > Case A (Unrotated, resists with Weak Axis):")
        print(f"    Sum(1.2*Mb) = {1.2*sum_mb:.2f}")
        print(f"    Sum(Mc)     = {sum_mc_unrotated:.2f} (2 * Mn_y)")
        print(f"    DCR Ratio   = {ratio_unrotated:.4f}")
        
        # Case B: Rotated Column
        sum_mc_rotated = 2 * mn0_z
        ratio_rotated = (1.2 * sum_mb) / (sum_mc_rotated + 1e-9)
        print(f"  > Case B (Rotated, resists with Strong Axis):")
        print(f"    Sum(1.2*Mb) = {1.2*sum_mb:.2f}")
        print(f"    Sum(Mc)     = {sum_mc_rotated:.2f} (2 * Mn_z)")
        print(f"    DCR Ratio   = {ratio_rotated:.4f}")
        
        if ratio_unrotated > 1.2 and ratio_rotated > 1.2:
             print("\n  [Insight] Even with rotation, SCWB might be critical if beams are strong relative to columns.")
        elif ratio_unrotated > 1.2:
             print("\n  [Insight] Rotation is required to satisfy SCWB.")
        else:
             print("\n  [Insight] SCWB is satisfied easily.")

    finally:
        h5_file.close()

if __name__ == "__main__":
    verify_scwb_logic()
