import h5py
import pandas as pd
import numpy as np
import openseespy.opensees as ops
import src.config as cfg
from src.structural_analysis import evaluate, build_model_for_section
from src.utils import load_section_data, get_beam_lengths, get_grouping_maps

def debug_final_fix():
    print("### Verifying Structural Analysis Fix ###")
    
    cfg.FLOORS = 6
    cfg.COLUMN_LOCATIONS = cfg.COLUMN_LOCATIONS_6F
    cfg.BEAM_CONNECTIONS = cfg.BEAM_CONNECTIONS_6F
    cfg.BEAM_TRIBUTARY_WIDTHS = cfg.BEAM_TRIBUTARY_WIDTHS_6F
    dynamic_patterns = cfg.LOAD_PATTERNS_6F
    
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    num_locations = len(cfg.COLUMN_LOCATIONS)
    num_columns = num_locations * cfg.FLOORS
    num_beams = len(cfg.BEAM_CONNECTIONS) * cfg.FLOORS
    beam_lengths = get_beam_lengths(cfg.COLUMN_LOCATIONS, cfg.BEAM_CONNECTIONS)
    
    num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
        "Hybrid", num_locations, num_columns, num_beams, cfg.FLOORS, cfg.BEAM_CONNECTIONS, cfg.COLUMN_LOCATIONS
    )
    
    chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
    individual = [len(column_sections)//2] * num_col_groups + [0] * num_col_groups + [len(beam_sections)//2] * num_beam_groups
    
    print("Running evaluate()...")
    results = evaluate(
        individual, cfg.DL_AREA_LOAD, cfg.LL_AREA_LOAD, h5_file, dynamic_patterns,
        col_map, beam_map, beam_sections, column_sections,
        beam_sections_df, column_sections_df, beam_lengths,
        chromosome_structure, num_columns, num_beams
    )
    
    print("\n[Final Results]")
    print("Max Drift Ratio:", results['max_drift_ratio'])
    print("Story Drifts X:", results['story_drifts_x'])
    print("Story Drifts Y:", results['story_drifts_y'])
    print("Violation:", results['violation'])

    h5_file.close()

if __name__ == "__main__":
    debug_final_fix()
