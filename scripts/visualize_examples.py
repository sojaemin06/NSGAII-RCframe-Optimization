import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# 프로젝트 루트 디렉토리를 path에 추가하여 src 모듈 import 가능하게 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as cfg
from src.utils import visualize_load_patterns, generate_load_patterns

def main():
    output_dir = "Visualization_Examples"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating plan visualizations in '{output_dir}'...")

    # Define Examples (Synced with main.py)
    examples = {
        'Example_1_4Story': {
            'Floors': 4,
            'Col_Locs': cfg.COLUMN_LOCATIONS_4F,
            'Beam_Conns': cfg.BEAM_CONNECTIONS_4F
        },
        'Example_2_6Story': {
            'Floors': 6,
            'Col_Locs': cfg.COLUMN_LOCATIONS_6F,
            'Beam_Conns': cfg.BEAM_CONNECTIONS_6F
        },
        'Example_3_8Story': {
            'Floors': 8,
            'Col_Locs': cfg.COLUMN_LOCATIONS_8F,
            'Beam_Conns': cfg.BEAM_CONNECTIONS_8F
        }
    }

    for name, config in examples.items():
        print(f" - Visualizing {name}...")
        
        # 1. Select Load Patterns (Hardcoded)
        if 'Example_1' in name:
            patterns = cfg.LOAD_PATTERNS_4F
        elif 'Example_2' in name:
            patterns = cfg.LOAD_PATTERNS_6F
        elif 'Example_3' in name:
            patterns = cfg.LOAD_PATTERNS_8F
        else:
            patterns = {}
        
        # 2. Visualize
        # Create a sub-folder for each example to keep clean
        ex_dir = os.path.join(output_dir, name)
        os.makedirs(ex_dir, exist_ok=True)
        
        visualize_load_patterns(
            config['Col_Locs'], 
            config['Beam_Conns'], 
            patterns, 
            output_folder=ex_dir
        )
        
    print("\nVisualization Complete.")
    print("Check the 'Visualization_Examples' folder for .png files.")

if __name__ == "__main__":
    main()
