
import time
import os
import h5py
import pandas as pd
from src.config import *
from src.utils import load_section_data, get_beam_lengths, calculate_fixed_scale, get_grouping_maps, visualize_load_patterns
from src.optimization import run_ga_optimization
from src.post_processing import save_results_to_csv, plot_results

def main():
    start_time = time.time() # 시간 측정 시작
    # 1. 설정 및 폴더 생성
    output_folder = f"Results_main_LoadIncreased_Pop{POPULATION_SIZE}_Gen{NUM_GENERATIONS}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nResults will be saved to: {output_folder}")
    
    # 2. 데이터 로드
    beam_sections_df, column_sections_df, beam_sections, column_sections = load_section_data()
    h5_file = h5py.File('pm_dataset_simple02.mat', 'r')
    
    # 3. 계산 및 초기화
    num_locations = len(COLUMN_LOCATIONS)
    num_columns = num_locations * FLOORS
    num_beams = len(BEAM_CONNECTIONS) * FLOORS
    beam_lengths = get_beam_lengths(COLUMN_LOCATIONS, BEAM_CONNECTIONS)
    
    # 4. 그룹핑
    num_col_groups, num_beam_groups, col_map, beam_map = get_grouping_maps(
        GROUPING_STRATEGY, num_locations, num_columns, num_beams, FLOORS, BEAM_CONNECTIONS, COLUMN_LOCATIONS
    )
    
    chromosome_structure = {'col_sec': num_col_groups, 'col_rot': num_col_groups, 'beam_sec': num_beam_groups}
    
    # 5. 고정 스케일 계산
    total_column_length = (len(COLUMN_LOCATIONS) * FLOORS) * H
    total_beam_length = sum(beam_lengths) * FLOORS
    fixed_min_cost, fixed_range_cost, fixed_min_co2, fixed_range_co2 = calculate_fixed_scale(
        column_sections_df, beam_sections_df, total_column_length, total_beam_length
    )
    
    print(f"Estimated Cost Range: {fixed_min_cost:,.0f} ~ {fixed_min_cost + fixed_range_cost:,.0f}")
    
    # 6. 하중 패턴 시각화
    visualize_load_patterns(COLUMN_LOCATIONS, BEAM_CONNECTIONS, PATTERNS_BY_FLOOR, output_folder)
    
    # 7. 최적화 실행
    pop, logbook, hof, hof_stats = run_ga_optimization(
        DL=DL_AREA_LOAD, LL=LL_AREA_LOAD, Wx=WX_RAND, Wy=WY_RAND, Ex=EX_RAND, Ey=EY_RAND,
        crossover_method=CROSSOVER_STRATEGY, patterns_by_floor=PATTERNS_BY_FLOOR, h5_file=h5_file,
        num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
        col_map=col_map, beam_map=beam_map, beam_sections=beam_sections, column_sections=column_sections,
        beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, beam_lengths=beam_lengths,
        chromosome_structure=chromosome_structure, num_columns=num_columns, num_beams=num_beams,
        fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
        fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2
    )
    
    # 8. 결과 처리 (추가 루프 포함)
    while True:
        # 유효해 필터링
        valid_solutions = []
        for ind in hof:
            if hasattr(ind, 'detailed_results') and ind.detailed_results.get('violation') == 0.0:
                valid_solutions.append(ind)
        
        if not valid_solutions:
            print("No feasible solutions found in Pareto front.")
        
        sorted_pareto = sorted(valid_solutions, key=lambda ind: ind.fitness.values[0])
        all_results = []
        for i, ind in enumerate(sorted_pareto):
            res = ind.detailed_results
            res['ID'] = f"Sol #{i+1}"
            res['individual'] = ind
            res['ind_object'] = ind
            all_results.append(res)
            
        # 결과 저장 및 시각화
        save_results_to_csv(output_folder, all_results, logbook, hof_stats, chromosome_structure)
        plot_results(output_folder, all_results, logbook, hof_stats, chromosome_structure, 
                     col_map, beam_map, beam_sections, column_sections)
        
        # 추가 세대 진행 여부
        break # 테스트를 위해 자동 종료
        try:
            more_gens = input(f"\nCurrent generations: {logbook[-1]['gen']}. Enter additional generations to continue (or Enter to finish): ")
            if not more_gens.strip():
                break
            add_gens = int(more_gens)
            if add_gens <= 0: continue
            
            pop, logbook, hof, hof_stats = run_ga_optimization(
                DL=DL_RAND, LL=LL_RAND, Wx=WX_RAND, Wy=WY_RAND, Ex=EX_RAND, Ey=EY_RAND,
                crossover_method=CROSSOVER_STRATEGY, patterns_by_floor=PATTERNS_BY_FLOOR, h5_file=h5_file,
                num_generations=add_gens, population_size=POPULATION_SIZE,
                col_map=col_map, beam_map=beam_map, beam_sections=beam_sections, column_sections=column_sections,
                beam_sections_df=beam_sections_df, column_sections_df=column_sections_df, beam_lengths=beam_lengths,
                chromosome_structure=chromosome_structure, num_columns=num_columns, num_beams=num_beams,
                fixed_min_cost=fixed_min_cost, fixed_range_cost=fixed_range_cost,
                fixed_min_co2=fixed_min_co2, fixed_range_co2=fixed_range_co2,
                initial_pop=pop, start_gen=logbook[-1]['gen'], logbook=logbook, hof=hof, hof_stats_history=hof_stats
            )
            
        except ValueError:
            print("Invalid input.")
            break
        except KeyboardInterrupt:
            break

    h5_file.close()
    end_time = time.time() # 시간 측정 종료
    elapsed_time = end_time - start_time
    print(f"\nTotal optimization time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
    print("Optimization finished.")

if __name__ == '__main__':
    main()
