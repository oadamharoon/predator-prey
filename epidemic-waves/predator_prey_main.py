import argparse
import matplotlib.pyplot as plt
from predator_prey_model import simulate
from predator_prey_analysis import analyze_coexistence, plot_outcome_heatmap, analyze_thresholds
from predator_prey_plots import plot_time_series, compare_python_introduction_rates, examine_parameter_sensitivity

def main():
    parser = argparse.ArgumentParser(description="Alligator-Python Predator-Prey Model Analysis")
    parser.add_argument("--run_all", action="store_true", help="Run all analyses")
    parser.add_argument("--time_series", action="store_true", help="Generate time series plots")
    parser.add_argument("--compare_intro", action="store_true", help="Compare python introduction rates")
    parser.add_argument("--coexistence", action="store_true", help="Analyze coexistence patterns")
    parser.add_argument("--analyze_thresholds", action="store_true", help="Analyze threshold values")
    parser.add_argument("--sensitivity", action="store_true", help="Perform parameter sensitivity analysis")
    parser.add_argument("--years", type=int, default=50, help="Number of years to simulate")
    args = parser.parse_args()
    
    # Set time steps based on argument
    time_steps = args.years
    
    # Default to running everything if no specific analysis is requested
    run_all = args.run_all or not (args.time_series or args.compare_intro or 
                                  args.coexistence or args.analyze_thresholds or 
                                  args.sensitivity)
    
    if run_all or args.time_series:
        print("Generating time series plots...")
        plot_time_series(time_steps=time_steps)
    
    if run_all or args.compare_intro:
        print("Comparing python introduction rates...")
        compare_python_introduction_rates(It_values=[0, 5, 15, 30], time_steps=time_steps)
    
    if run_all or args.coexistence:
        print("Analyzing coexistence patterns...")
        data = analyze_coexistence(time_steps=time_steps)
        plot_outcome_heatmap(data)
    
    if run_all or args.analyze_thresholds:
        print("Analyzing threshold values...")
        analyze_thresholds()
    
    if run_all or args.sensitivity:
        print("Performing parameter sensitivity analysis...")
        # Examine sensitivity to key parameters from the paper
        examine_parameter_sensitivity('bA', [1.5, 2.5, 3.5, 4.5], time_steps=time_steps)
        examine_parameter_sensitivity('bP', [2.0, 3.0, 4.0, 5.0], time_steps=time_steps)
        examine_parameter_sensitivity('cAP', [0.20, 0.22, 0.24, 0.26], time_steps=time_steps)
        examine_parameter_sensitivity('K', [10000, 15000, 20000, 25000], time_steps=time_steps)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()