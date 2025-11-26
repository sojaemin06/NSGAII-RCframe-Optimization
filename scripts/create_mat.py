import h5py
import numpy as np
import pandas as pd
import os

def create_expanded_rotated_mat(original_mat_path, expanded_csv_path, new_mat_path):
    """
    재정렬된 확장 CSV 파일의 순서에 맞춰 P-M 상호작용도 데이터를 정렬하고
    새로운 .mat 파일을 생성합니다.
    """
    try:
        # 1. 원본 .mat 파일에서 모든 P-M 데이터를 메모리로 로드
        with h5py.File(original_mat_path, 'r') as hf_in:
            num_original_sections = hf_in['Column_Mdata'].shape[1]
            original_pm_data = []
            for i in range(num_original_sections):
                ref = hf_in['Column_Mdata'][0, i]
                original_pm_data.append(hf_in[ref][()])
            print(f"원본 .mat 파일에서 {len(original_pm_data)}개의 P-M 데이터를 로드했습니다.")

        # 2. 확장된 CSV 파일을 읽어 재정렬된 순서 확인
        df_expanded = pd.read_csv(expanded_csv_path)
        print(f"확장 CSV 파일 '{expanded_csv_path}'에서 {len(df_expanded)}개의 단면 정보를 로드했습니다.")

        # 3. 새로운 .mat 파일 생성 및 데이터 재정렬하여 저장
        with h5py.File(new_mat_path, 'w') as hf_out:
            total_sections = len(df_expanded)

            # 재정렬된 순서에 따라 P-M 데이터를 저장
            for new_idx, row in df_expanded.iterrows():
                # [수정] 'original_name' 컬럼을 직접 참조하여 원본 P-M 데이터의 인덱스를 찾음
                # 'original_name'은 1부터 시작하므로, 0-based 인덱스로 변환하기 위해 -1
                original_pm_index = int(row['original_name']) - 1
                data_to_save = original_pm_data[original_pm_index]
                
                hf_out.create_dataset(f'pm_data_{new_idx}', data=data_to_save)

            # 'Column_Mdata' 참조 배열 생성
            ref_dtype = h5py.special_dtype(ref=h5py.Reference)
            ref_dataset = hf_out.create_dataset('Column_Mdata', (1, total_sections), dtype=ref_dtype)
            for i in range(total_sections):
                ref_dataset[0, i] = hf_out[f'pm_data_{i}'].ref

            print(f"총 {total_sections}개의 P-M 데이터를 재정렬하여 새 파일 '{new_mat_path}'에 저장했습니다.")

    except FileNotFoundError as e:
        print(f"오류: 필요한 파일을 찾을 수 없습니다. '{e.filename}'")
    except Exception as e:
        print(f"오류 발생: {e}")

# --- 스크립트 실행 ---
if __name__ == '__main__':
    original_mat = '../data/pm_dataset_simple02.mat'
    expanded_csv = '../data/column_sections_expanded_rotated.csv'
    new_mat = '../data/pm_dataset_expanded_rotated.mat'
    
    print(f"'{original_mat}'과 '{expanded_csv}' 파일을 참조하여 '{new_mat}' 파일을 생성합니다...")
    create_expanded_rotated_mat(original_mat, expanded_csv, new_mat)
    print("파일 생성 완료!")