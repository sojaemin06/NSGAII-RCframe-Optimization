import pandas as pd
import os

def create_expanded_sections_csv(original_csv_path, new_csv_path):
    """
    원본 단면 CSV 파일을 읽어, 각 단면을 90도 회전시킨 데이터를 추가하고,
    성능 지표(PI_Pn_max)를 기준으로 재정렬하여 2배 크기의 새로운 단면 CSV 파일을 생성합니다.
    """
    try:
        # 1. 원본 CSV 파일 로드
        df_original = pd.read_csv(original_csv_path)
        print(f"원본 파일 '{original_csv_path}'에서 {len(df_original)}개의 단면을 로드했습니다.")
        
        # [추가] 재정렬 후 원본 인덱스를 추적하기 위해 'original_name' 컬럼 생성
        df_original['original_name'] = df_original['name']

        # 2. 회전된 단면 데이터 생성 (기하학적 정보만 교환)
        df_rotated = df_original.copy()

        # h와 b 값 교환
        df_rotated[['h', 'b']] = df_rotated[['b', 'h']]

        # [수정] 강도 값(Vn, Pn, Mn 등)은 교환하지 않습니다.
        # 강도 값은 단면의 로컬 좌표계를 기준으로 하므로, 기하학적 회전만으로 충분합니다.

        print(f"회전된 단면 {len(df_rotated)}개를 생성했습니다.")

        # 3. 원본과 회전된 데이터프레임 결합
        df_expanded = pd.concat([df_original, df_rotated], ignore_index=True)
        
        # [추가] 회전된 데이터의 original_name도 원본과 동일하게 설정 (회전 근원을 알 수 있도록)
        # 이 부분은 df_rotated = df_original.copy()에서 이미 처리됨

        # 4. 성능 지표(PI_Pn_max) 기준으로 전체 데이터를 재정렬
        print(f"'PM_Volume'(강축 P-M 상호작용도 면적)을 기준으로 전체 {len(df_expanded)}개 단면을 재정렬합니다.")
        df_expanded = df_expanded.sort_values(by='PM_Volume', ascending=True, ignore_index=True)

        # 5. 'name' 컬럼을 1부터 순차적으로 새로 부여
        df_expanded['name'] = range(1, len(df_expanded) + 1)

        # 6. 새로운 CSV 파일로 저장
        df_expanded.to_csv(new_csv_path, index=False)
        print(f"총 {len(df_expanded)}개의 단면 데이터를 '{new_csv_path}' 파일로 저장했습니다.")

    except FileNotFoundError:
        print(f"오류: 원본 CSV 파일 '{original_csv_path}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"파일 생성 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    original_file = '../data/column_sections_simple02.csv'
    new_file = '../data/column_sections_expanded_rotated.csv'
    create_expanded_sections_csv(original_file, new_file)