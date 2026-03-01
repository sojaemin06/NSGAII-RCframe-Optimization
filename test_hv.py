from deap.benchmarks.tools import hypervolume as hv_indicator
import numpy as np

def test_hv():
    print("Testing HV...")
    fitnesses = [(0.8, 0.001), (0.7, 0.005), (0.6, 0.010)]
    ref_point = [2.5, 2.5]
    
    try:
        val = hv_indicator(fitnesses, ref_point)
        print("HV Result:", val)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_hv()
