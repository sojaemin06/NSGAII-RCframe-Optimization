from exp_pop_common import run_pop_batch

if __name__ == "__main__":
    # Part 2: Medium population sizes
    target_pops = [400, 500, 600]
    print(f"--- Running Population Experiment Part 2: {target_pops} ---")
    run_pop_batch(target_pops)
