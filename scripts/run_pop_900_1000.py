from exp_pop_common import run_pop_batch

if __name__ == "__main__":
    # Part 4: Extra Large population sizes
    target_pops = [900, 1000]
    print(f"--- Running Population Experiment Part 4: {target_pops} ---")
    run_pop_batch(target_pops)
