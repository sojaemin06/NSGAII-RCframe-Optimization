import subprocess
import os
import sys
import time

def run_high_load_experiment():
    # 설정
    seed = 42
    pop_size = 100 # 테스트를 위해 작게 설정 (실제로는 600 권장)
    generations = 20 # 테스트를 위해 작게 설정
    ex_load = 400.0 # 10배 증대 (40 -> 400)
    ey_load = 400.0 # 10배 증대
    
    # 현재 스크립트 위치 기준 상대 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "simpledata_2obj_NSGA2.py")
    output_dir = os.path.join(current_dir, "..", "results", f"Experiment_HighLoad_Ex{int(ex_load)}")
    
    python_executable = sys.executable

    print(f"--- Starting High Load Experiment (Ex={ex_load}kN, Ey={ey_load}kN) ---")
    print(f"Output Directory: {output_dir}")
    
    cmd = [
        python_executable, script_path,
        "--seed", str(seed),
        "--pop", str(pop_size),
        "--gen", str(generations),
        "--ex", str(ex_load),
        "--ey", str(ey_load),
        "--output", output_dir,
        "--batch"
    ]
    
    try:
        # 환경 변수 설정 (UTF-8 인코딩 등)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        subprocess.run(cmd, cwd=current_dir, env=env, check=True, encoding='utf-8', errors='replace')
        print("Experiment completed successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error code {e.returncode}")

if __name__ == "__main__":
    run_high_load_experiment()
