
# =================================================================
# ===              1. 최적화 및 모델링 주요 설정                  ===
# =================================================================

# --- 1.1. 그룹핑 및 교배 전략 선택 ---
GROUPING_STRATEGY = "Hybrid" # "Hybrid", "Individual", "ByFloor", "Uniform"
CROSSOVER_STRATEGY = "OnePoint"  # "OnePoint", "TwoPoint", "Uniform"

# --- 1.2. 건물 기본 정보 ---
FLOORS = 4
H = 4.0

# --- 1.3. 건물 형상 정보 (기본값 - 4층) ---
# [4-Story Irregular Plan (Existing)]
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

BEAM_TRIBUTARY_WIDTHS_4F = [
    3.0, 3.0, 3.0,      # Y=0
    5.0, 5.0, 5.0,      # Y=6
    2.0, 4.5, 4.5,      # Y=10
    2.5, 2.5,           # Y=15
    2.5, 2.5,           # X=0
    5.0, 5.0, 2.5,      # X=5
    5.0, 5.0, 5.0,      # X=10
    2.5, 2.5, 2.5       # X=15
]

# [6-Story: 'U' Shape Plan with Closed Wings]
# Grid X: 0, 6, 10, 17 (Spans: 6m, 4m, 7m)
# Grid Y: 0, 6, 11     (Spans: 6m, 5m)
# Open Bay: Top-Middle (between X=6~10, Y=11 line is open)
# Columns:
# Y=0:  (0,0), (6,0), (10,0), (17,0)  [Indices 0,1,2,3]
# Y=6:  (0,6), (6,6), (10,6), (17,6)  [Indices 4,5,6,7]
# Y=11: (0,11),(6,11),(10,11),(17,11) [Indices 8,9,10,11]
COLUMN_LOCATIONS_6F = [
    (0, 0), (6, 0), (10, 0), (17, 0),
    (0, 6), (6, 6), (10, 6), (17, 6),
    (0, 11), (6, 11), (10, 11), (17, 11)
]

BEAM_CONNECTIONS_6F = [
    # X-Beams (Horizontal)
    (0, 1), (1, 2), (2, 3),    # Y=0: 0-6, 6-10, 10-17
    (4, 5), (5, 6), (6, 7),    # Y=6: 0-6, 6-10, 10-17
    (8, 9), (10, 11),          # Y=11: 0-6, 10-17 (Middle 6-10 is OPEN)
    
    # Y-Beams (Vertical)
    (0, 4), (4, 8),            # X=0: 0-6, 6-11
    (1, 5), (5, 9),            # X=6: 0-6, 6-11
    (2, 6), (6, 10),           # X=10: 0-6, 6-11
    (3, 7), (7, 11)            # X=17: 0-6, 6-11
]

# Total Beams = 8 (X) + 8 (Y) = 16 beams
BEAM_TRIBUTARY_WIDTHS_6F = [
    # X-Beams (Y=0)
    3.0, 3.0, 3.0,
    # X-Beams (Y=6)
    5.5, 2.0, 6.0,  # Middle (5,6) only takes load from bottom (2.0) as top is open
    # X-Beams (Y=11)
    2.5, 2.5,       # Left(0-6) and Right(10-17) wings
    
    # Y-Beams (X=0)
    3.0, 3.0,
    # Y-Beams (X=6)
    5.0, 3.0,       # Top part (6-11) takes 3.0 (from Left 6m/2), Right is open
    # Y-Beams (X=10)
    5.5, 3.5,       # Top part (6-11) takes 3.5 (from Right 7m/2), Left is open
    # Y-Beams (X=17)
    3.5, 3.5
]

# [8-Story: Cruciform (+) Shape with Irregular Spans]
# Center Core: X(5~11), Y(5~11) -> 6m x 6m
# Left Wing: X(0~5) -> 5m span
# Right Wing: X(11~18) -> 7m span
# Bottom Wing: Y(0~5) -> 5m span
# Top Wing: Y(11~18) -> 7m span
# Coordinates:
# Row Y=0:        (5,0),  (11,0)
# Row Y=5: (0,5), (5,5),  (11,5), (18,5)
# Row Y=11: (0,11),(5,11), (11,11),(18,11)
# Row Y=18:       (5,18), (11,18)
COLUMN_LOCATIONS_8F = [
            (5, 0), (11, 0),
    (0, 5), (5, 5), (11, 5), (18, 5),
    (0, 11), (5, 11), (11, 11), (18, 11),
            (5, 18), (11, 18)
]
# Indices:
# 0,1
# 2,3,4,5
# 6,7,8,9
# 10,11
BEAM_CONNECTIONS_8F = [
    # X-Beams
    (0, 1),                 # Y=0 (6m)
    (2, 3), (3, 4), (4, 5), # Y=5 (5m, 6m, 7m)
    (6, 7), (7, 8), (8, 9), # Y=11 (5m, 6m, 7m)
    (10, 11),               # Y=18 (6m)
    # Y-Beams
    (2, 6),                 # X=0 (6m)
    (0, 3), (3, 7), (7, 10), # X=5 (5m, 6m, 7m)
    (1, 4), (4, 8), (8, 11), # X=11 (5m, 6m, 7m)
    (5, 9)                  # X=18 (6m)
]
BEAM_TRIBUTARY_WIDTHS_8F = [
    2.5,                    # Y=0 (from 5m span/2)
    2.5, 5.5, 3.5,          # Y=5
    2.5, 6.5, 3.5,          # Y=11 (6m/2 + 7m/2 = 6.5)
    3.5,                    # Y=18
    2.5,                    # X=0
    2.5, 6.0, 3.5,          # X=5
    2.5, 6.0, 3.5,          # X=11
    3.0                     # X=18
]

# Default to 4F
COLUMN_LOCATIONS = COLUMN_LOCATIONS_4F
BEAM_CONNECTIONS = BEAM_CONNECTIONS_4F
BEAM_TRIBUTARY_WIDTHS = BEAM_TRIBUTARY_WIDTHS_4F

# [Load Patterns - Hardcoded Checkerboard]
# Indices based on BEAM_CONNECTIONS order in this file.

# 4F (L-shape): 
# Pattern A (Floors 1, 3): Load Bays (0,0), (10,0), (5,6), (10,10)
# Pattern B (Floors 2, 4): Load Bays (5,0), (0,6), (10,6), (5,10)
# Beams:
# Row 0(Y=0): 0,1,2. Row 1(Y=6): 3,4,5. Row 2(Y=10): 6,7,8. Row 3(Y=15): 9,10
# Col 0(X=0): 11,12. Col 1(X=5): 13,14,15. Col 2(X=10): 16,17,18. Col 3(X=15): 19,20,21
LOAD_PATTERNS_4F = {
    1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 19, 20],
    2: [1, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    3: [3, 4, 6, 7, 9, 12, 14, 15, 17, 18],
    4: [1, 2, 4, 5, 7, 8, 13, 14, 16, 17, 19, 20],
}

# 6F (U-shape):
# X-Beams: Y=0(0,1,2), Y=6(3,4,5). Y=11(None? No, Y=11 has no X-beams in previous def? Wait.)
# Let's check BEAM_CONNECTIONS_6F indices:
# 0,1,2 (Y=0). 3,4,5 (Y=6).
# 6,7 (X=0). 8 (X=6). 9 (X=10). 10,11 (X=17).
# Total 12 beams per floor.
LOAD_PATTERNS_6F = {
    1: [0, 1, 2, 3, 4, 5, 8, 10, 12, 14],
    2: [1, 2, 4, 5, 7, 10, 12, 13, 14, 15],
    3: [],
    4: [0, 1, 2, 3, 4, 5, 8, 10, 12, 14],
    5: [1, 2, 4, 5, 7, 10, 12, 13, 14, 15],
    6: [1, 2, 4, 5, 10, 12, 14],
}

# 8F (Cruciform):
# Indices: 
# X-Beams: 0(Y=0), 1,2,3(Y=5), 4,5,6(Y=11), 7(Y=18)
# Y-Beams: 8(X=0), 9,10,11(X=5), 12,13,14(X=11), 15(X=18)
LOAD_PATTERNS_8F = {
    1: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15],
    2: [0, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
    3: [1, 2, 4, 5, 7, 8, 10, 11, 13, 14],
    4: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15],
    5: [0, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
    6: [1, 2, 4, 5, 7, 8, 10, 11, 13, 14],
    7: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15],
    8: [0, 2, 3, 5, 6, 9, 10, 12, 13, 15],
}

# Default
PATTERNS_BY_FLOOR = LOAD_PATTERNS_4F
# 기존 선하중(kN/m) 대신 면적당 하중(kN/m2)을 정의하여 일관성 확보
# 슬래브 자중(3.6) + 마감(1.4) = 5.0 kN/m2 가정
DL_AREA_LOAD = 5.0 # kN/m2 (Dead Load including Slab Self-weight)

# [수정] 층별 활하중 차등 적용 (용도 변화 모사)
# 1-2층: 로비/상업 (5.0), 3-5층: 사무실 (3.0), 6층 이상: 주거/회의 (2.0)
LL_AREA_LOAD = {
    1: 5.0, 2: 5.0, 
    3: 3.0, 4: 3.0, 5: 3.0,
    6: 2.0, 7: 2.0, 8: 2.0,
    'default': 2.0
}

# [REMOVED] Fixed Load Constants (WX_RAND, etc.) - Now calculated dynamically based on ASCE 7-16

# 각 보별 분담 폭 (Tributary Width) [m]
# BEAM_CONNECTIONS 리스트 순서와 일치해야 함
BEAM_TRIBUTARY_WIDTHS = [
    3.0, 3.0, 3.0,      # Y=0  (X-beams): 6m/2 = 3.0
    5.0, 5.0, 5.0,      # Y=6  (X-beams): 6m/2 + 4m/2 = 5.0
    2.0, 4.5, 4.5,      # Y=10 (X-beams): (0-5구간: 4m/2=2.0), (5-15구간: 4m/2 + 5m/2 = 4.5)
    2.5, 2.5,           # Y=15 (X-beams): 5m/2 = 2.5
    2.5, 2.5,           # X=0  (Y-beams): 5m/2 = 2.5
    5.0, 5.0, 2.5,      # X=5  (Y-beams): (0-10구간: 5.0), (10-15구간: 오른쪽만 있음=2.5)
    5.0, 5.0, 5.0,      # X=10 (Y-beams): 5.0
    2.5, 2.5, 2.5       # X=15 (Y-beams): 2.5
]

# --- 1.4.1 등가정적해석법(ELF) 파라미터 (ASCE 7-16 기준) ---
# 가정: Site Class D (Stiff Soil), Risk Category II
# Ss = 0.75g, S1 = 0.30g 가정 -> Fa=1.2, Fv=1.8
# SDS = 0.60, SD1 = 0.36
SDS = 0.60      # 단주기 설계 스펙트럼 가속도
SD1 = 0.36      # 1초 주기 설계 스펙트럼 가속도

R_COEFF = 5.0   # 반응 수정 계수 (RC Intermediate Moment Frame)
I_FACTOR = 1.0  # 중요도 계수

# 고유 주기 약산식 파라미터 (RC 모멘트 골조)
# Ta = Ct * h_n^x
PERIOD_CT = 0.0466
PERIOD_X = 0.9

# --- 1.4.2 슬래브 하중 정보 (중량 산정용) ---
# 기존 DL_RAND는 보에 가해지는 선하중으로 유지하고,
# 지진력 산정 시 필요한 '추가 슬래브 자중'을 정의합니다.
# (사용자: "고정하중에 슬래브의 무게는 현재 없어")
SLAB_THICKNESS = 0.15 # m
CONCRETE_UNIT_WEIGHT = 24.0 # kN/m3
SLAB_DL_KN_M2 = SLAB_THICKNESS * CONCRETE_UNIT_WEIGHT # 3.6 kN/m2

# --- 1.4.3 비용 파라미터 ---
FORMWORK_UNIT_COST = 25.0 # 단위 면적당 거푸집 설치 비용 (예: $/m^2 또는 KRW/m^2)

# 평면 면적 (자동 계산 또는 고정값)
# 좌표 분석 결과: (15x6) + (15x4) + (10x5) = 90 + 60 + 50 = 200 m^2
FLOOR_AREA = 200.0 # m^2

# --- 1.4.3 풍하중(Wind Load) 파라미터 (ASCE 7-16 MWFRS) ---
BASIC_WIND_SPEED = 30.0 # m/s (V)
# Exposure B Constants (ASCE 7 Table 26.11-1)
ALPHA = 7.0
ZG = 365.76 # m
KZT = 1.0 # Topographic Factor (Flat)
KD = 0.85 # Wind Directionality Factor
G_FACTOR = 0.85 # Gust Effect Factor (Rigid)
CP_WINDWARD = 0.8 # External Pressure Coeff (Windward)
CP_LEEWARD = -0.5 # External Pressure Coeff (Leeward) - Total CP = 0.8 - (-0.5) = 1.3

# 건물 평면 치수 (Wind Load 산정용)
BUILDING_WIDTH_X = 15.0 # m (Y방향 풍하중 수압폭)
BUILDING_WIDTH_Y = 15.0 # m (X방향 풍하중 수압폭)

# --- 1.5. 유전 알고리즘 파라미터 ---
POPULATION_SIZE = 100 
NUM_GENERATIONS = 200
CXPB = 0.8
MUTPB = 0.2

# --- 1.6. 하중 조합 ---
LOAD_COMBINATIONS = [
    # 1. 1.4D
    ("ACI-1", {"DL": 1.4, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 0, "Ey": 0}),
    # 2. 1.2D + 1.6L
    ("ACI-2", {"DL": 1.2, "LL": 1.6, "Wx": 0, "Wy": 0, "Ex": 0, "Ey": 0}),
    # 3. 1.2D + 1.0L + 1.0W
    ("ACI-3", {"DL": 1.2, "LL": 1.0, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
    ("ACI-4", {"DL": 1.2, "LL": 1.0, "Wx": -1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
    ("ACI-5", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 1.0, "Ex": 0, "Ey": 0}),
    ("ACI-6", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": -1.0, "Ex": 0, "Ey": 0}),
    # 4. 1.2D + 1.0L + 1.0E (직교 효과 포함)
    ("ACI-7", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": 0.3}),
    ("ACI-8", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": -0.3}),
    ("ACI-9", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -1.0, "Ey": 0.3}),
    ("ACI-10", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -1.0, "Ey": -0.3}),
    ("ACI-11", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.3, "Ey": 1.0}),
    ("ACI-12", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.3, "Ey": -1.0}),
    ("ACI-13", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.3, "Ey": 1.0}),
    ("ACI-14", {"DL": 1.2, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.3, "Ey": -1.0}),
    # 5. 0.9D + 1.0W
    ("ACI-15", {"DL": 0.9, "LL": 0, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
    ("ACI-16", {"DL": 0.9, "LL": 0, "Wx": -1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
    ("ACI-17", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 1.0, "Ex": 0, "Ey": 0}),
    ("ACI-18", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": -1.0, "Ex": 0, "Ey": 0}),
    # 6. 0.9D + 1.0E (직교 효과 포함)
    ("ACI-19", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": 0.3}),
    ("ACI-20", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 1.0, "Ey": -0.3}),
    ("ACI-21", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": -1.0, "Ey": 0.3}),
    ("ACI-22", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": -1.0, "Ey": -0.3}),
    ("ACI-23", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 0.3, "Ey": 1.0}),
    ("ACI-24", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": 0.3, "Ey": -1.0}),
    ("ACI-25", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": -0.3, "Ey": 1.0}),
    ("ACI-26", {"DL": 0.9, "LL": 0, "Wx": 0, "Wy": 0, "Ex": -0.3, "Ey": -1.0}),
    # --- ✅ ASCE 7 사용성 검토용 하중조합 (신규 추가 및 수정) ---
    # 1. 풍하중(W) 변위 검토용: D + 0.5L + W
    ("ASCE-S-W1", {"DL": 1.0, "LL": 0.5, "Wx": 1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
    ("ASCE-S-W2", {"DL": 1.0, "LL": 0.5, "Wx": -1.0, "Wy": 0, "Ex": 0, "Ey": 0}),
    ("ASCE-S-W3", {"DL": 1.0, "LL": 0.5, "Wx": 0, "Wy": 1.0, "Ex": 0, "Ey": 0}),
    ("ASCE-S-W4", {"DL": 1.0, "LL": 0.5, "Wx": 0, "Wy": -1.0, "Ex": 0, "Ey": 0}),
    # 2. 지진하중(E) 층간변위 검토용: D + L + 0.7E (직교 효과 포함)
    ("ASCE-S-E1", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.7, "Ey": 0.21}),
    ("ASCE-S-E2", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.7, "Ey": -0.21}),
    ("ASCE-S-E3", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.7, "Ey": 0.21}),
    ("ASCE-S-E4", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.7, "Ey": -0.21}),
    ("ASCE-S-E5", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.21, "Ey": 0.7}),
    ("ASCE-S-E6", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": 0.21, "Ey": -0.7}),
    ("ASCE-S-E7", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.21, "Ey": 0.7}),
    ("ASCE-S-E8", {"DL": 1.0, "LL": 1.0, "Wx": 0, "Wy": 0, "Ex": -0.21, "Ey": -0.7}),
]

PATTERNS_BY_FLOOR = {
    1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 19, 20],
    2: [1, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    3: [3, 4, 6, 7, 9, 12, 14, 15, 17, 18],
    4: [1, 2, 4, 5, 7, 8, 13, 14, 16, 17, 19, 20]
}
