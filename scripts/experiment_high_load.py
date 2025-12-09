import sys
import os
import random
import pandas as pd
import numpy as np
import h5py
import math

# 프로젝트 루트 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.structural_analysis import evaluate
from src.config import *
import src.config as config

def run_high_load_experiment():
    print("="*60)
    print("🚀 High Load Experiment (Seismic Stability Check)")
    print("="*60)

    # 1. 데이터 로드
    print("[1] Loading Section Database...")
    try:
        beam_sections_df = pd.read_csv("beam_sections_simple02.csv")
        column_sections_df = pd.read_csv("column_sections_simple02.csv") # 기본 DB 사용
        h5_file = h5py.File("pm_dataset_simple02.mat", 'r')
        
        beam_sections = [(row["b"]/1000, row["h"]/1000) for _, row in beam_sections_df.iterrows()]
        column_sections = [(row["b"]/1000, row["h"]/1000) for _, row in column_sections_df.iterrows()]
        print(f" - Loaded {len(beam_sections)} beam sections, {len(column_sections)} column sections.")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return

    # 2. 매핑 및 유전자 구조 생성
    print("[2] Generating Mapping & Chromosome...")
    num_locations = len(COLUMN_LOCATIONS)
    num_columns = num_locations * FLOORS
    num_beams = len(BEAM_CONNECTIONS) * FLOORS
    
    # Hybrid Grouping (간소화) - 여기서는 테스트를 위해 단순 매핑 사용 가능하지만
    # 실제 로직과 동일하게 맞추기 위해 가상의 그룹핑 할당
    # (실제 그룹핑 로직은 main.py 등에 있으나 여기선 단순화하여 층별 그룹핑 가정)
    
    col_map = {}
    beam_map = {}
    
    # 간단히 층별로 그룹핑한다고 가정 (테스트 목적)
    num_col_groups = FLOORS
    num_beam_groups = FLOORS
    
    for i in range(num_columns):
        floor = i // num_locations
        col_map[i + 1] = floor
        
    for i in range(num_beams):
        floor = i // len(BEAM_CONNECTIONS)
        beam_map[num_columns + i + 1] = floor

    chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
    
    # 3. 임의의 개체 생성 (중간 정도 크기의 단면 선택)
    print("[3] Creating a Test Individual (Mid-size sections)...")
    mid_col_idx = len(column_sections) // 2
    mid_beam_idx = len(beam_sections) // 2
    
    # 기둥: 중간 크기, 회전: 0 (미회전), 보: 중간 크기
    individual = [mid_col_idx] * num_col_groups + [0] * num_col_groups + [mid_beam_idx] * num_beam_groups
    
    # 4. 고하중 조건 확인 및 설정
    # 현재 config.py의 파라미터들이 고하중(지진)을 유발하는지 확인
    # SDS=0.60, SD1=0.36 (Site Class D, High Seismicity) -> OK
    
    print(f"[4] Running Evaluation under High Seismic Load...")
    print(f" - Parameters: SDS={config.SDS}, SD1={config.SD1}, R={config.R_COEFF}")
    
    # 5. Evaluate 실행
    # PATTERNS_BY_FLOOR는 config.py에 정의된 것 사용 (키가 int인지 확인)
    patterns = {k: set(v) for k, v in PATTERNS_BY_FLOOR.items()} 
    
    results = evaluate(
        individual, 
        config.DL_AREA_LOAD, 
        config.LL_AREA_LOAD, 
        config.WX_RAND, config.WY_RAND, 
        config.EX_RAND, config.EY_RAND, # 이 값들은 내부 ELF 로직에 의해 덮어씌워지거나 보조적으로 사용됨
        h5_file, 
        patterns,
        col_map, beam_map, 
        beam_sections, column_sections,
        beam_sections_df, column_sections_df,
        # beam_lengths 계산 필요
        [math.sqrt((COLUMN_LOCATIONS[p2][0] - COLUMN_LOCATIONS[p1][0])**2 + 
                   (COLUMN_LOCATIONS[p2][1] - COLUMN_LOCATIONS[p1][1])**2) 
         for p1, p2 in BEAM_CONNECTIONS],
        chromosome_structure,
        num_columns, num_beams
    )
    
    # 6. 결과 분석
    print("\n" + "="*60)
    print("📊 Experiment Results")
    print("="*60)
    
    if results['cost'] == float('inf'):
        print("❌ Analysis FAILED (Non-convergence or Error)")
        print(f"   Violation: {results['violation']}")
    else:
        print("✅ Analysis COMPLETED Successfully")
        print(f" - Cost: {results['cost']:,.2f}")
        print(f" - CO2: {results['co2']:,.2f}")
        print(f" - Mean DCR: {results['mean_strength_ratio']:.4f}")
        print(f" - Max DCR: {results['max_strength_ratio']:.4f}")
        print(f" - Max Drift Ratio: {results['violation_drift']:.4f} (Allowable: 1.0 = 1.5%)")
        
        # 지진 하중 수준 확인 (간접적)
        # forces_df에서 기둥의 최대 전단력을 통해 유추하거나, 
        # evaluate 함수 내에서 출력된 로그(있다면) 확인
        
        # DCR 분포 확인
        print(f" - DCR > 1.0 count: {sum(1 for r in results['strength_ratios'] if r > 1.0)}")
        
    print("="*60)

if __name__ == "__main__":
    run_high_load_experiment()