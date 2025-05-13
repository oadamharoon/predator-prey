import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from func_file_utils import save_data, load_data
from predator_prey_model import simulate

def analyze_coexistence(P_threshold=1.0, A_threshold=1.0, time_steps=50):
    """Analyze coexistence patterns for different parameter combinations"""
    # Parameter ranges to explore
    It_range = np.linspace(0, 50, 10)      # Python introduction rate
    K_range = np.linspace(8000, 20000, 10)  # Mammal carrying capacity
    
    # Base parameters that produce stable results - reduced birth rates
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
        'time_steps': time_steps
    }
    
    # Results matrices
    outcome_matrix = np.zeros((len(It_range), len(K_range)), dtype=object)
    python_matrix = np.zeros((len(It_range), len(K_range)))
    alligator_matrix = np.zeros((len(It_range), len(K_range)))
    
    for i, It in enumerate(It_range):
        for j, K in enumerate(K_range):
            # Set parameters for this simulation
            params = base_params.copy()
            params['It'] = It
            params['K'] = K
            
            # Run simulation
            _, results = simulate(**params)
            
            # Get final populations
            final_Y = results[-1, 0]
            final_A = results[-1, 1]
            final_P = results[-1, 2]
            
            # Determine outcome
            alligator_persists = final_A > A_threshold
            python_persists = final_P > P_threshold
            
            if alligator_persists and python_persists:
                outcome = "coexistence"
            elif alligator_persists:
                outcome = "alligator_dominance"
            elif python_persists:
                outcome = "python_dominance"
            else:
                outcome = "extinction"
                
            outcome_matrix[i, j] = outcome
            python_matrix[i, j] = final_P
            alligator_matrix[i, j] = final_A
    
    data = {
        'It_range': It_range,
        'K_range': K_range,
        'outcome_matrix': outcome_matrix,
        'python_matrix': python_matrix,
        'alligator_matrix': alligator_matrix
    }
    
    save_data(data, 'predator_prey_analysis.pkl')
    return data

def plot_outcome_heatmap(data=None):
    """Plot outcome heatmap"""
    if data is None:
        try:
            data = load_data('predator_prey_analysis.pkl')
        except FileNotFoundError:
            data = analyze_coexistence()
    
    outcome_matrix = data['outcome_matrix']
    It_range = data['It_range']
    K_range = data['K_range']
    
    # Create color mapping
    color_map = {
        'coexistence': [0, 1, 0],           # Green
        'alligator_dominance': [0, 0, 1],   # Blue 
        'python_dominance': [1, 0, 0],      # Red
        'extinction': [0.5, 0.5, 0.5]       # Gray
    }
    
    # Convert outcome matrix to color matrix
    color_matrix = np.zeros((len(It_range), len(K_range), 3))
    for i in range(len(It_range)):
        for j in range(len(K_range)):
            color_matrix[i, j] = color_map[outcome_matrix[i, j]]
    
    plt.figure(figsize=(10, 8))
    img = plt.imshow(color_matrix, origin='lower', aspect='auto', 
               extent=[min(K_range), max(K_range), min(It_range), max(It_range)])
    
    # Fixed colorbar implementation
    plt.colorbar(img, label='Outcome')
    
    plt.xlabel('Mammal Carrying Capacity (K)')
    plt.ylabel('Python Introduction Rate (It)')
    plt.title('Ecosystem Outcomes for Different Parameter Combinations')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['coexistence'], label='Coexistence'),
        Patch(facecolor=color_map['alligator_dominance'], label='Alligator Dominance'),
        Patch(facecolor=color_map['python_dominance'], label='Python Dominance'),
        Patch(facecolor=color_map['extinction'], label='Extinction')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig('predator_prey_outcomes.pdf')
    plt.show()
    
    return color_matrix

def analyze_thresholds():
    """Identify threshold values for python introduction rates with improved visualization"""
    # Parameter ranges
    It_values = np.linspace(0, 30, 61)  # Reduced range from 0-100 to 0-30
    time_points = 50
    
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
        'time_steps': time_points
    }
    
    # Arrays to store results
    final_pythons = np.zeros(len(It_values))
    final_alligators = np.zeros(len(It_values))
    
    # Run simulations
    for i, It in enumerate(It_values):
        params = base_params.copy()
        params['It'] = It
        _, results = simulate(**params)
        
        final_pythons[i] = results[-1, 2]
        final_alligators[i] = results[-1, 1]
    
    # Plot results with separate subplots and appropriate scales
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot pythons
    axes[0].plot(It_values, final_pythons, 'r-', linewidth=2)
    axes[0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Persistence Threshold')
    axes[0].set_ylabel('Python Population')
    axes[0].set_title('Python Population vs. Introduction Rate')
    axes[0].legend()
    axes[0].grid(True)
    
    # Add annotation to explain persistence threshold
    axes[0].annotate('Populations below the persistence threshold\nare considered functionally extinct',
                    xy=(15, 1.5), xytext=(15, 10),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Plot alligators
    axes[1].plot(It_values, final_alligators, 'g-', linewidth=2)
    axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Persistence Threshold')
    axes[1].set_xlabel('Python Introduction Rate (It)')
    axes[1].set_ylabel('Adult Alligator Population')
    axes[1].set_title('Alligator Population vs. Introduction Rate')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('python_introduction_threshold.pdf')
    plt.show()

if __name__ == "__main__":
    # Analyze coexistence patterns
    data = analyze_coexistence()
    plot_outcome_heatmap(data)
    
    # Analyze thresholds
    analyze_thresholds()