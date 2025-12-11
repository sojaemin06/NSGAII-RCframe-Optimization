
import os
import src.config as cfg
from src.utils import visualize_load_patterns

def verify():
    output_dir = "Verification_Load_Patterns"
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify 4F
    print("Visualizing 4F Patterns...")
    visualize_load_patterns(cfg.COLUMN_LOCATIONS_4F, cfg.BEAM_CONNECTIONS_4F, cfg.LOAD_PATTERNS_4F, output_dir)
    if os.path.exists(os.path.join(output_dir, "load_pattern_visualization.png")):
        if os.path.exists(os.path.join(output_dir, "load_pattern_4F.png")):
            os.remove(os.path.join(output_dir, "load_pattern_4F.png"))
        os.rename(os.path.join(output_dir, "load_pattern_visualization.png"), os.path.join(output_dir, "load_pattern_4F.png"))
    
    # Verify 6F
    print("Visualizing 6F Patterns...")
    visualize_load_patterns(cfg.COLUMN_LOCATIONS_6F, cfg.BEAM_CONNECTIONS_6F, cfg.LOAD_PATTERNS_6F, output_dir)
    if os.path.exists(os.path.join(output_dir, "load_pattern_visualization.png")):
        if os.path.exists(os.path.join(output_dir, "load_pattern_6F.png")):
            os.remove(os.path.join(output_dir, "load_pattern_6F.png"))
        os.rename(os.path.join(output_dir, "load_pattern_visualization.png"), os.path.join(output_dir, "load_pattern_6F.png"))
    
    # Verify 8F
    print("Visualizing 8F Patterns...")
    visualize_load_patterns(cfg.COLUMN_LOCATIONS_8F, cfg.BEAM_CONNECTIONS_8F, cfg.LOAD_PATTERNS_8F, output_dir)
    if os.path.exists(os.path.join(output_dir, "load_pattern_visualization.png")):
        if os.path.exists(os.path.join(output_dir, "load_pattern_8F.png")):
            os.remove(os.path.join(output_dir, "load_pattern_8F.png"))
        os.rename(os.path.join(output_dir, "load_pattern_visualization.png"), os.path.join(output_dir, "load_pattern_8F.png"))
    
    print(f"Verification complete. Images saved in {output_dir}")

if __name__ == "__main__":
    verify()
