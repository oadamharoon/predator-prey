import numpy as np
import matplotlib.pyplot as plt
from predator_prey_model import simulate

def plot_time_series(params=None, time_steps=50):
    """Plot time series with improved visualization"""
    # Use default parameters if none provided
    if params is None:
        params = {
            'bA': 2.5,
            'r': 0.075,
            'mA': 0.12,
            'pAM': 0.0005,
            'pAP': 0.003,
            'bP': 3.0,
            'mP': 0.30,
            'cAP': 0.23,
            'pPM': 0.0005,
            'cAM': 5.0,
            'cPM': 20.0,
            'It': 5.0,
            'K': 15000.0,
            'bm': 1.5
        }
    
    # Run simulation
    time_years, results = simulate(time_steps=time_steps, **params)
    
    # Create subplot for better visualization
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot on first subplot - predators
    axes[0].plot(time_years, results[:, 0], 'b-', linewidth=2, label='Juvenile Alligators')
    axes[0].plot(time_years, results[:, 1], 'g-', linewidth=2, label='Adult Alligators')
    axes[0].plot(time_years, results[:, 2], 'r-', linewidth=2, label='Pythons')
    axes[0].set_ylabel('Predator Population')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_title('Alligator-Python Population Dynamics')
    
    # Plot on second subplot - mammals
    axes[1].plot(time_years, results[:, 3], 'k-', linewidth=2, label='Mammals')
    axes[1].set_xlabel('Time (years)')
    axes[1].set_ylabel('Mammal Population')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('predator_prey_dynamics.pdf')
    plt.show()
    
    # Print final population values for verification
    print(f"Final populations after {time_steps} years:")
    print(f"Juvenile Alligators: {results[-1, 0]:.1f}")
    print(f"Adult Alligators: {results[-1, 1]:.1f}")
    print(f"Pythons: {results[-1, 2]:.1f}")
    print(f"Mammals: {results[-1, 3]:.1f}")

def compare_python_introduction_rates(It_values=[0, 5, 15, 30], time_steps=50):
    """Compare different python introduction rates with improved visualization"""
    fig, axs = plt.subplots(len(It_values), 1, figsize=(10, 4*len(It_values)), sharex=True)
    
    # Base parameters - updated with lower birth rates
    base_params = {
        'bA': 2.5,
        'r': 0.075,
        'mA': 0.12,
        'pAM': 0.0005,
        'pAP': 0.003,
        'bP': 3.0,
        'mP': 0.30,
        'cAP': 0.23,
        'pPM': 0.0005,
        'cAM': 5.0,
        'cPM': 20.0,
        'K': 15000.0,
        'bm': 1.5,
        'time_steps': time_steps
    }
    
    # Arrays to store final populations for each introduction rate
    final_populations = []
    
    # Run simulations for each introduction rate
    for i, It in enumerate(It_values):
        params = base_params.copy()
        params['It'] = It
        t, results = simulate(**params)
        final_populations.append(results[-1, :3])  # Store final predator populations
        
        # Calculate appropriate y-axis range for this plot
        y_max = max(np.max(results[:, 0]), np.max(results[:, 1]), np.max(results[:, 2])) * 1.1
        
        # Plot results with consistent scale per subplot
        axs[i].plot(t, results[:, 0], 'b-', label='Juvenile Alligators')
        axs[i].plot(t, results[:, 1], 'g-', label='Adult Alligators') 
        axs[i].plot(t, results[:, 2], 'r-', label='Pythons')
        
        axs[i].set_ylim(0, y_max)
        axs[i].set_title(f'Python Introduction Rate = {It}')
        axs[i].set_ylabel('Population')
        axs[i].legend()
        axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Print final populations for verification
        print(f"Final populations with introduction rate {It}:")
        print(f"Juvenile Alligators: {results[-1, 0]:.1f}")
        print(f"Adult Alligators: {results[-1, 1]:.1f}")
        print(f"Pythons: {results[-1, 2]:.1f}")
        print(f"Mammals: {results[-1, 3]:.1f}")
        print(f"Predator:Prey Ratio: {(results[-1, 0] + results[-1, 1] + results[-1, 2])/results[-1, 3]:.3f}")
    
    axs[-1].set_xlabel('Time (years)')
    plt.tight_layout()
    plt.savefig('python_introduction_comparison.pdf')
    plt.show()

def examine_parameter_sensitivity(param_name, param_values, time_steps=50):
    """Examine sensitivity to a specific parameter"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Base parameters - updated with lower birth rates
    base_params = {
        'bA': 2.5,
        'r': 0.075,
        'mA': 0.12,
        'pAM': 0.0005,
        'pAP': 0.003,
        'bP': 3.0,
        'mP': 0.30,
        'cAP': 0.23,
        'pPM': 0.0005,
        'cAM': 5.0,
        'cPM': 20.0,
        'It': 5.0,
        'K': 15000.0,
        'bm': 1.5,
        'time_steps': time_steps
    }
    
    # Colors for different parameters
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    
    # Simulate for each parameter value
    for i, value in enumerate(param_values):
        params = base_params.copy()
        params[param_name] = value
        t, results = simulate(**params)
        
        # Plot final populations versus parameter value
        ax.plot(t, results[:, 2], color=colors[i], label=f'{param_name}={value}')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Python Population')
    ax.set_title(f'Sensitivity to {param_name}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'sensitivity_{param_name}.pdf')
    plt.show()

if __name__ == "__main__":
    # Base parameters with reduced birth rates for stability
    stable_params = {
        'bA': 2.5,
        'r': 0.075,
        'mA': 0.12,
        'pAM': 0.0005,
        'pAP': 0.003,
        'bP': 3.0,
        'mP': 0.30,
        'cAP': 0.23,
        'pPM': 0.0005,
        'cAM': 5.0,
        'cPM': 20.0,
        'It': 5.0,
        'K': 15000.0,
        'bm': 1.5
    }
    plot_time_series(stable_params)
    
    # Compare different python introduction rates
    compare_python_introduction_rates()
    
    # Examine sensitivity to key parameters
    examine_parameter_sensitivity('bP', [2.0, 3.0, 4.0, 5.0])
    examine_parameter_sensitivity('cAP', [0.18, 0.21, 0.24, 0.27])