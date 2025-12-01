import subprocess
import os
import sys
import time
import random

def run_batch_optimization(num_trials=3, pop_size=20, generations=5):
    # 프로젝트 루트 기준 상대 경로
    base_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "Batch_Validation_Test"))
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "simpledata_2obj_NSGA2.py"))
    
    # 현재 실행 중인 파이썬 실행 파일 경로 (가상환경 내 python.exe 권장)
    python_executable = sys.executable 

    print(f"Starting batch optimization: {num_trials} trials")
    print(f"Config: Pop={pop_size}, Gen={generations}")
    print(f"Output Directory: {base_output_dir}")
    print(f"Python Executable: {python_executable}")

    for i in range(1, num_trials + 1):
        trial_seed = random.randint(1, 10000)
        trial_output_dir = os.path.join(base_output_dir, f"Trial_{i:02d}")
        
        print(f"\n--- Running Trial {i}/{num_trials} (Seed: {trial_seed}) ---")
        
        cmd = [
            python_executable, script_path,
            "--seed", str(trial_seed),
            "--pop", str(pop_size),
            "--gen", str(generations),
            "--output", trial_output_dir,
            "--batch"
        ]
        
        try:
            start_time = time.time()
            # 환경 변수 설정 (UTF-8 인코딩 등)
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            # 실행 (출력 캡처)
            scripts_dir = os.path.dirname(script_path)
            result = subprocess.run(cmd, cwd=scripts_dir, env=env, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
            elapsed = time.time() - start_time
            print(f"Trial {i} completed in {elapsed:.2f} seconds.")
            
            # 로그 파일 저장
            os.makedirs(trial_output_dir, exist_ok=True)
            with open(os.path.join(trial_output_dir, "stdout.log"), "w", encoding='utf-8') as f:
                f.write(result.stdout)
            if result.stderr:
                 with open(os.path.join(trial_output_dir, "stderr.log"), "w", encoding='utf-8') as f:
                    f.write(result.stderr)
            
        except subprocess.CalledProcessError as e:
            print(f"Trial {i} failed with error code {e.returncode}")
            print("Stderr:", e.stderr)

    print("\nBatch optimization completed.")

if __name__ == "__main__":
    # 테스트용 기본값: 3회, 인구 10, 세대 2 (매우 빠른 검증용)
    run_batch_optimization(num_trials=3, pop_size=10, generations=2)
