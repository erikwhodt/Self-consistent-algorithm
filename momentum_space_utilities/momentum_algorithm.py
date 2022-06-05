import csv 
import datetime as datetime
from momentum_space_utilities.source import Measurement, Temp
from momentum_space_utilities.functions import *

################################################################################
## Iterative algorithm
################################################################################

def momentum_algorithm(file_path, alpha_array, mu_array, polarization_array, Q_array, Q_array_text, density_init, magnetization_init, threshold, k_res, U, psi, beta, t, postfix, n_entries) -> None:
    
    for index, alpha in enumerate(alpha_array):
        # print(f"Process: {postfix} Current alpha progess: {index+1} of total {len(alpha_array)} values")
        for index_mu, mu in enumerate(mu_array):
            
            data_f = open(f'{file_path}_{postfix*n_entries+index_mu}.csv', 'w', newline='')
            data_writer = csv.writer(data_f)
            data_writer.writerow(['mu', 'alpha', 'angle', 'Q', 'n', 'm', 'TP', 'did it converge?'])

            # print(f"Process: {postfix} Current mu progess: {index_mu+1} of total {len(mu_array)} values")
            for polarization in polarization_array:
                # print(f"## Current polarization: {polarization}")
                for idx, Q in enumerate(Q_array):
                    print(f"Process: {postfix} Current mu progess: {index_mu+1} / {len(mu_array)} Current orientation: {polarization} Current Q: {Q_array_text[idx]}")
                    iteration = 0
                    # print(f"## Process: {postfix} Current polarization: {polarization} Current Q-vector: {Q_array_text[idx]}")

                    ## Reset convergence and PM flags

                    PM = False
                    FM = False
                    if Q_array_text[idx] == 'PM': 
                        PM = True
                    elif Q_array_text[idx] == 'FM': 
                        FM = True
                    
                    ## Create Measurement-object which stores information about the current configuration
                    current_measurement = Measurement(mu, alpha, polarization, Q, Q_array_text[idx], density_init, magnetization_init, 0,True, threshold)

                    temp_1, temp_2, temp_3, temp_4 = Temp(), Temp(), Temp(), Temp()
                    
                    k_grid, k_res_temp, N = brillouinMaster(Q, k_res) 
                    
                    while current_measurement.has_not_converged():
                        
                        current_measurement.density = current_measurement.density_new 
                        current_measurement.magnetization = current_measurement.magnetization_new
                    
                        # print(f"Process: Iteration: ", iteration+1)
                        # print(current_measurement)

                        xi = -U*current_measurement.density / 2
                        h = 2*U*current_measurement.magnetization

                        ## Zeroing out the variables to be summed over the Brillouin zone 
                        current_measurement.density_new, current_measurement.magnetization_new = 0,0 
                        energy, m_x, m_y, m_z, m_tot = 0,0,0,0, 0
                        # now = datetime.datetime.now()
                        if PM: 
                            n_temp, energy = matrix_wrapper_PM(Q, N, k_grid, t, xi, alpha, psi, 0, mu, polarization, beta, k_res_temp)
                        elif FM:
                            m_tot, n_temp, energy = matrix_wrapper_FM(Q, N, k_grid, t, xi, alpha, psi, h, mu, polarization, beta, k_res_temp)
                        else:              
                            m_tot, n_temp, energy = matrix_wrapper(Q, N, k_grid, t, xi, alpha, psi, h, mu, polarization, beta, k_res_temp)
                            
                        # print(f"Zone integration took: {datetime.datetime.now()-now}")
                        
                        ## Dividing by the number of points in the 1BZ and the periodicity in real space to get values per lattice site

                        current_measurement.density_new = 1/(N * k_res_temp**2) * n_temp
                        
                        current_measurement.magnetization_new = m_tot

                        current_measurement.free_energy = energy/(N * k_res_temp**2) - U*current_measurement.density_new**2 / 4 + U*current_measurement.magnetization_new**2
                        
                        ## Values from last 4 iterations are saved for a possible average in case of no convergence

                        if iteration % 4 == 0: 
                            temp_1.set(current_measurement.density_new, current_measurement.magnetization_new, current_measurement.free_energy)
                        elif iteration % 4 == 1:
                            temp_2.set(current_measurement.density_new, current_measurement.magnetization_new, current_measurement.free_energy)
                        elif iteration % 4 == 2: 
                            temp_3.set(current_measurement.density_new, current_measurement.magnetization_new, current_measurement.free_energy)
                        elif iteration  % 4 == 3: 
                            temp_4.set(current_measurement.density_new, current_measurement.magnetization_new, current_measurement.free_energy)

                        ## The iterative algorithm may become stuck in a loop, alternating between two values. The following code captures this event.
                        ## did_not_converge flag is raised.

                        if temp_1 == temp_3 and temp_2 == temp_4 and iteration > 10:       
                            current_measurement.did_it_converge = False
                            current_measurement.density_new = (temp_1.density+temp_2.density+temp_3.density+temp_4.density)/4
                            current_measurement.magnetization_new = (temp_1.magnetization+temp_2.magnetization+temp_3.magnetization+temp_4.magnetization)/4
                            energy = (temp_1.energy+temp_2.energy+temp_3.energy+temp_4.energy)/4
                            break

                        ## The algorithm may be caught in a loop, but with small changes < 10**-5 from iteration to iteration. The following code rounds 
                        ## values to 5 decimals and ends the loop. There is a small chance that the code would've converged if let to itself, but this is 
                        ## considered to be too time consuming. did_not_converge flag is raised.

                        elif np.round(temp_1.density, 5) == np.round(temp_3.density, 5) and np.round(temp_2.density, 5) == np.round(temp_4.density, 5) and np.round(temp_1.magnetization, 5) == np.round(temp_3.magnetization, 5) and np.round(temp_2.magnetization, 5) == np.round(temp_4.magnetization, 5) and iteration > 100:        #Fails to converge, stuck in loop
                            current_measurement.did_it_converge = False
                            current_measurement.density_new = (temp_1.density+temp_2.density+temp_3.density+temp_4.density)/4
                            current_measurement.magnetization_new = (temp_1.magnetization+temp_2.magnetization+temp_3.magnetization+temp_4.magnetization)/4
                            energy = (temp_1.energy+temp_2.energy+temp_3.energy+temp_4.energy)/4
                            break
                        
                        elif np.round(temp_1.energy, 5) == np.round(temp_2.energy, 5) == np.round(temp_3.energy, 5) == np.round(temp_4.energy, 5): 
                            print("Energy equal, breaking...")
                            break

                        ## If code has not converged by iteration 200, the average from the last four iterations are taken as the obtained value. did_not_converge flag
                        ## is raised.

                        elif iteration > 200:        #Fails to converge, stuck in loop
                            current_measurement.did_it_converge = False
                            current_measurement.density_new = (temp_1.density+temp_2.density+temp_3.density+temp_4.density)/4
                            current_measurement.magnetization_new = (temp_1.magnetization+temp_2.magnetization+temp_3.magnetization+temp_4.magnetization)/4
                            energy = (temp_1.energy+temp_2.energy+temp_3.energy+temp_4.energy)/4
                            break

                        iteration += 1           

                    # print("Iteration complete")
                    # print(current_measurement)

                    current_measurement.free_energy = energy/(N * k_res_temp**2) - U*current_measurement.density_new**2 / 4 + U*current_measurement.magnetization_new**2 
                    
                    data_writer.writerow(current_measurement.output())
            data_f.close()
    print(f"## Process {postfix} has finished!")
    # print(f"Average zone integration time: {np.average(time_array)}")
    
    pass