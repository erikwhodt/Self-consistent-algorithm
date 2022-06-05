import numpy as np
from momentum_space_utilities.momentum_algorithm import momentum_algorithm
from real_space_utilities.real_space_algorithm import real_algorithm
from concurrent.futures import ProcessPoolExecutor
import datetime as datetime

pi = np.pi

################################################################################
## Variable declarations
################################################################################

## Hopping strength
t = 1
## Interaction strength
U = 9* t
## Initial particle density
density_init = 1
## Chemical potential renormalization
xi = -U*density_init/2
## FM component in z. pi/2 means parallel to xy.
psi = pi/2
## Initial magnetization
magnetization_init=0.5
## Iteration convergence criterion
threshold = 10**-4
average_threshold = 10**-4
## Thermodynamic temperature
beta = 1 / 0.01
## Resolution of the full first Brillouin zone NB: ONLY FOR MOMENTUM_ALGORITHM
k_res = 150
## Size of the finite-size system NB: ONLY FOR REAL_ALGORITHM
N =20
## Resolution, max and minimum for Rashba SOC coupling strength
alpha_res = 1 #50
alpha_min = 0.0 #1
alpha_max = 0.0
# # Resolution, max and minimum for chemical potential
mu_res =1

mu_min = 0.53# 3.75
mu_max = 0.53
## Number of processes NB! Number of processes must be an integer fraction of the resolution in alpha or mu
n_process = 1
## Number of array entries handed to each process. Is given by mu_res / n_process or alpha_res / n_process
n_entries = int(mu_res / n_process)

################################################################################
## File handling
################################################################################

## The file path where data files and plots are saved
file_path = f"data\\phase_diagram_v2\\U={U}\\"

## The file path where initial values for time-evolution are saved 
dynamic_path = "data/dynamics_initial_values/"

################################################################################
## Container declaration
################################################################################

## Array containing the three polarizations 
polarization_array = np.array(['x','y','z'])

## SOC coupling strength array
alpha_array = np.linspace(alpha_min,alpha_max,alpha_res)

## Chemical potential array
mu_array = np.linspace(mu_min, mu_max, mu_res)

## Arrray of magnetic modulation vectors as well as string-versions for printing
Q_array = np.array([[0,0], [0,0],[pi,pi], [pi/2,pi/2],[pi/4,pi/4],[pi/8,pi/8],[pi/16,pi/16], [0,pi], [0,pi/2], [0,pi/4], [0,pi/8], [0,pi/16], [pi,0], [pi/2,0], [pi/4,0], [pi/8,0], [pi/16,0], [pi/2,pi], [pi/4,pi], [pi/8,pi],[pi/16,pi], [pi, pi/2], [pi,pi/4], [pi,pi/8], [pi,pi/16]])
Q_array_text = ['PM', 'FM', '[pi,pi]', '[pi_2,pi_2]','[pi_4,pi_4]','[pi_8,pi_8]','[pi_16,pi_16]',  '[0,pi]', '[0,pi_2]', '[0,pi_4]','[0,pi_8]', '[0,pi_16]', '[pi,0]', '[pi_2,0]', '[pi_4,0]', '[pi_8,0]','[pi_16,0]', '[pi_2,pi]','[pi_4,pi]','[pi_8,pi]','[pi_16,pi]', '[pi,pi_2]','[pi,pi_4]','[pi,pi_8]', '[pi,pi_16]']


################################################################################
## Iterative algorithm
################################################################################

if __name__ == ("__main__"):

    ## The ProcessPoolExecutor manages the running of the different processes. It spawns n_process workers which the execute all instructions (functions) submitted to it using executor.submit()

    with ProcessPoolExecutor(max_workers = n_process) as executor:
        for i in range(n_process):
            
            ## The snippet calling the real-space algorithm
            executor.submit(real_algorithm, file_path, Q_array,Q_array_text, mu_array[i*n_entries:i*n_entries+n_entries], alpha_array,polarization_array, N, density_init, magnetization_init, average_threshold, beta, psi, U, t, i, n_entries, dynamic_path )
            
            ## The snippet calling the momentum-space algorithm
            #executor.submit(momentum_algorithm, file_path, alpha_array, mu_array, polarization_array, Q_array, Q_array_text, density_init, magnetization_init, threshold, k_res, U, psi, beta, t, postfix, n_entries)
