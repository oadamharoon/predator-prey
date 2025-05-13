import numpy as np
from numba import jit

@jit(nopython=True)
def predator_prey_update(y, params):
    """Discrete-time update for alligator-python-mammal system
    This directly implements the equation system from the paper"""
    Y, A, P, M = y  # Unpack current state
    
    # Unpack parameters
    bA, r, mA, pAM, pAP, bP, mP, cAP, pPM, cAM, cPM, It, K, bm = params
    
    # Update each population using exact equations from the paper
    Y_next = Y*(1-r) + bA*A*(2 - np.exp(-pAM*M) - np.exp(-pAP*P))
    A_next = A*(1-mA) + r*Y
    P_next = P*(1-mP) + bP*P*(1-np.exp(-pPM*M)) - cAP*A*(1-np.exp(-pAP*P)) + It
    M_next = bm*M*(1-M/K) - cAM*A*(1-np.exp(-pAM*M)) - cPM*P*(1-np.exp(-pPM*M))
    
    # Ensure non-negative populations
    Y_next = max(0.0, Y_next)
    A_next = max(0.0, A_next)
    P_next = max(0.0, P_next)
    M_next = max(0.0, M_next)
    
    return np.array([Y_next, A_next, P_next, M_next])

def simulate(
    # Alligator parameters - significantly reduced birth rate
    bA=2.5,       # REDUCED from 12.0 - limiting factor to control population
    r=0.075,      # Juvenile alligator maturation rate (0.07-0.083)
    mA=0.12,      # Increased mortality rate from 0.10
    pAM=0.0005,   # Predation efficiency parameter for alligators on mammals
    pAP=0.003,    # Predation efficiency parameter for alligators on pythons
    
    # Python parameters - reduced birth rate
    bP=3.0,       # REDUCED from 18.0 - limiting factor to control population
    mP=0.30,      # Increased mortality rate from 0.25
    cAP=0.23,     # Adult alligator predation rate on pythons (0.20-0.26)
    pPM=0.0005,   # Predation efficiency parameter for pythons on mammals
    
    # Predation rates 
    cAM=5.0,      # Average mammals consumed per alligator per year
    cPM=20.0,     # Average mammals consumed per python per year
    
    # Management and ecosystem parameters
    It=5.0,       # Python introduction rate 
    K=15000.0,    # Mammal carrying capacity
    bm=1.5,       # Mammal population growth rate
    
    # Initial conditions - more realistic starting populations
    Y0=50, A0=25, P0=5, M0=8000, 
    time_steps=50
):
    """
    Run a discrete-time simulation of the alligator-python-mammal system
    using the equations specified in the paper.
    
    The birth rates (bA, bP) have been significantly reduced from the paper's values
    to create a more realistic ecological pyramid where prey populations exceed
    predator populations, as would be expected in nature.
    """
    # Initialize arrays to store results
    results = np.zeros((time_steps+1, 4))
    results[0] = np.array([Y0, A0, P0, M0])
    
    # Parameter tuple for passing to update function
    params = (bA, r, mA, pAM, pAP, bP, mP, cAP, pPM, cAM, cPM, It, K, bm)
    
    # Perform discrete-time simulation
    for t in range(time_steps):
        results[t+1] = predator_prey_update(results[t], params)
    
    # Create time points array (years)
    time_points = np.arange(time_steps+1)
    
    return time_points, results