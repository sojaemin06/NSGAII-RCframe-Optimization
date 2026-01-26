import pandas as pd
import os

input_path = "column_sections_reduced.csv"
output_path = "column_sections_expanded_rotated.csv"

# Load original data
df = pd.read_csv(input_path)

# Create rotated copy
df_rot = df.copy()

# 1. Swap Geometry
df_rot['temp_b'] = df_rot['b']
df_rot['b'] = df_rot['h']
df_rot['h'] = df_rot['temp_b']
df_rot.drop(columns=['temp_b'], inplace=True)

# 2. Swap Shear Strength (Vn_y <-> Vn_z)
# Assuming Z is strong axis usually, but we just swap indices
df_rot['temp_vn_z'] = df_rot['PI_Vn_z']
df_rot['PI_Vn_z'] = df_rot['PI_Vn_y']
df_rot['PI_Vn_y'] = df_rot['temp_vn_z']
df_rot.drop(columns=['temp_vn_z'], inplace=True)

# 3. Swap Moment Strength (Mn_y <-> Mn_z)
df_rot['temp_mn_z'] = df_rot['PI_Mn_z']
df_rot['PI_Mn_z'] = df_rot['PI_Mn_y']
df_rot['PI_Mn_y'] = df_rot['temp_mn_z']
df_rot.drop(columns=['temp_mn_z'], inplace=True)

# 4. Swap Rebar Spacing
df_rot['temp_space_h'] = df_rot['spacing_h']
df_rot['spacing_h'] = df_rot['spacing_b']
df_rot['spacing_b'] = df_rot['temp_space_h']
df_rot.drop(columns=['temp_space_h'], inplace=True)

# 5. Swap Side Rebars
# Top/Bottom <-> Left/Right
df_rot['temp_top'] = df_rot['side_rebars_top']
df_rot['temp_bot'] = df_rot['side_rebars_bottom']

df_rot['side_rebars_top'] = df_rot['side_rebars_left']
df_rot['side_rebars_bottom'] = df_rot['side_rebars_right']

df_rot['side_rebars_left'] = df_rot['temp_top']
df_rot['side_rebars_right'] = df_rot['temp_bot']

df_rot.drop(columns=['temp_top', 'temp_bot'], inplace=True)

# 6. Prepare for Interleaving
# Add 'original_name' to track source (before concat)
df['original_name'] = df['name']
df_rot['original_name'] = df['name']

# Add a temporary sort key to maintain order
# Original: 0, 2, 4...
# Rotated: 1, 3, 5...
df['sort_key'] = df.index * 2
df_rot['sort_key'] = df.index * 2 + 1

# Combine and Sort
df_expanded = pd.concat([df, df_rot]).sort_values('sort_key').reset_index(drop=True)
df_expanded.drop(columns=['sort_key'], inplace=True)

# 7. Re-assign IDs (1 ~ 1600)
df_expanded['name'] = range(1, len(df_expanded) + 1)

# Save
df_expanded.to_csv(output_path, index=False)
print(f"Expanded DB saved to {output_path}")
print(f"Total items: {len(df_expanded)} (Interleaved: Original -> Rotated -> Original -> ...)")
print("Sample check (first 4 rows):")
# Use a safer print method or list existing columns if key error persists
try:
    print(df_expanded[['name', 'h', 'b', 'original_name']].head(4))
except KeyError as e:
    print(f"Print Error: {e}")
    print("Columns available:", df_expanded.columns.tolist())