import pandas as pd
import sys
import os
import h5py
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_pm_capacity_from_df, load_pm_data_for_column

def check_ranges():
    print("### Checking Section Database Strength Ranges ###")
    
    # 1. Load CSVs
    try:
        beam_df = pd.read_csv("beam_sections_simple02.csv")
        col_df = pd.read_csv("column_sections_simple02.csv")
        h5_file = h5py.File("pm_dataset_simple02.mat", 'r')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Beam Strengths (PiM)
    if 'PiM' in beam_df.columns:
        print(f"\n[Beam Sections] (Total {len(beam_df)} types)")
        print(f"  > Min PiM: {beam_df['PiM'].min():.2f} kNm")
        print(f"  > Max PiM: {beam_df['PiM'].max():.2f} kNm")
        print(f"  > Avg PiM: {beam_df['PiM'].mean():.2f} kNm")
    else:
        print("\n[Error] 'PiM' column not found in beam CSV.")

    # 3. Column Strengths (Pure Bending Mn at P=0)
    print(f"\n[Column Sections] (Total {len(col_df)} types)")
    
    mn_values = []
    
    # Check first, middle, last, and random samples
    indices_to_check = [0, len(col_df)//2, len(col_df)-1]
    
    print("  > Sampling Column Capacities at P=0 (Pure Bending):")
    for idx in indices_to_check:
        pm_df = load_pm_data_for_column(h5_file, idx)
        if pm_df.empty:
            print(f"    idx {idx}: P-M Data Missing!")
            continue
            
        _, mn_z = get_pm_capacity_from_df(0, pm_df, axis='z')
        _, mn_y = get_pm_capacity_from_df(0, pm_df, axis='y')
        mn = max(mn_z, mn_y) # Assume strong axis usage
        mn_values.append(mn)
        print(f"    idx {idx} ({col_df.iloc[idx]['b']}x{col_df.iloc[idx]['h']}): Mn_zero = {mn:.2f} kNm")

    # 4. Scenario Check
    # Worst Case: Max Beam vs Min Column (Interior Joint: 4 beams vs 2 columns)
    # Demand = 1.2 * (4 * MaxBeam)
    # Capacity = 2 * MinCol
    
    max_beam_mn = beam_df['PiM'].max()
    min_col_mn = min(mn_values) if mn_values else 0
    
    worst_demand = 1.2 * (4 * max_beam_mn)
    worst_capacity = 2 * min_col_mn
    
    print(f"\n[Worst-Case Scenario Assessment]")
    print(f"  > Worst Demand (4 Max Beams): {worst_demand:.2f} kNm")
    print(f"  > Worst Capacity (2 Min Cols): {worst_capacity:.2f} kNm")
    if worst_capacity > 0:
        print(f"  > Potential Max Ratio (H): {worst_demand / worst_capacity:.2f}")
    else:
        print(f"  > Potential Max Ratio (H): Infinite (Capacity=0)")

    h5_file.close()

if __name__ == "__main__":
    check_ranges()
