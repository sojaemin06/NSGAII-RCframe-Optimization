from exp_pop_common import run_pop_batch

if __name__ == "__main__":
    # Part 1: Small population sizes
    target_pops = [100, 200, 300]
    print(f"--- Running Population Experiment Part 1: {target_pops} ---")
    run_pop_batch(target_pops)
