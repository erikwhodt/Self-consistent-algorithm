from real_space_utilities.functions import *
from real_space_utilities.source import *
from momentum_space_utilities.source import Temp
import csv
import os
def real_algorithm(file_path, Q_array, Q_text_array, mu_array, alpha_array, polarization_array, N, density_init, magnetization_init, threshold, beta, psi, U, t, postfix, n_entries, dynamic_path):
    try:
        os.mkdir(f'{file_path}U={U}')
    except FileExistsError:
        pass    



    for index_alpha, alpha in enumerate(alpha_array):
        for index_mu, mu in enumerate(mu_array): 
            try:
                os.mkdir(f'{file_path}U={U}/mu={mu}')
            except FileExistsError:
                pass
            

            data_f = open(f'{file_path}/U={U}/{postfix*n_entries+index_mu}.csv', 'w', newline='')
            data_writer = csv.writer(data_f)
            data_writer.writerow(['mu', 'alpha', 'angle', 'Q', 'n', 'm', 'TP', 'did it converge?'])
            
            for direction in polarization_array:
                for idx, Q in enumerate(Q_array): 
                    
                    ## The seed_flag is raised during the first iteration to allow for a seeding of the magnetix texture. After the first iteration, the
                    ## flag is lowered and the calculations performed using a general hamiltonian without a Q-restriction. 

                    seed_flag = True
                    PM_flag = False
                    iteration = 0              

                    if Q_text_array[idx] == 'PM': 
                        PM_flag = True
                    print(f"U: {U} Alpha: {alpha:.2f}  Dir: {direction} Chem. pot: {mu:.2f}\n## Current Q-vector: {Q_text_array[idx]}")

                    current_system = Lattice(N, magnetization_init, density_init, threshold, mu, beta, psi, Q, Q_text_array[idx], U, t, alpha, direction, PM_flag, seed_flag)

                    temp_1, temp_2, temp_3, temp_4 = Temp(), Temp(), Temp(), Temp()



                    while current_system.has_not_converged():
                        if iteration == 1: 
                            current_system.save_plot(iteration, file_path)

                        print(f"Iteration: {iteration + 1}")
                        current_system.update_values()
                        
                        current_system.calculate_free_energy()
                        
                        current_system.create_matrix() 
                        current_system.obtain_new_values()
                        
                        print(current_system)
                        
                        iteration += 1 
                        
                    
                        if iteration % 4 == 0: 
                            temp_1.set(np.average(current_system.n_array_new), np.average(current_system.m_array_new), current_system.energy)
                        elif iteration % 4 == 1:
                            temp_2.set(np.average(current_system.n_array_new), np.average(current_system.m_array_new), current_system.energy)
                        elif iteration % 4 == 2: 
                            temp_3.set(np.average(current_system.n_array_new), np.average(current_system.m_array_new), current_system.energy)
                        elif iteration  % 4 == 3: 
                            temp_4.set(np.average(current_system.n_array_new), np.average(current_system.m_array_new), current_system.energy)

                        ## Statement 1

                        if temp_1 == temp_3 and temp_2 == temp_4 and iteration > 10:       
                            current_system.did_it_converge = False
                            print("Statement 1 - breaking...")
                            break
                        
                        ## Statement 2

                        # elif np.round(temp_1.density, 7) == np.round(temp_3.density, 7) and np.round(temp_2.density, 7) == np.round(temp_4.density, 7) and np.round(temp_1.magnetization, 7) == np.round(temp_3.magnetization, 7) and np.round(temp_2.magnetization, 7) == np.round(temp_4.magnetization, 7) and iteration > 100:        #Fails to converge, stuck in loop
                        #     current_system.did_it_converge = False
                        #     print("Statement 2 - breaking...")
                        #     break
                        
                        # Statement 3

                        # elif np.round(temp_1.energy, 5) == np.round(temp_2.energy, 5) == np.round(temp_3.energy, 5) == np.round(temp_4.energy, 5): 
                        #     current_system.did_it_converge = False
                        #     print("Statement 3 - breaking...")
                        #     break

                        if iteration == 10000: 
                            current_system.did_it_converge = False
                            print("Took too long...")
                            break
                        # current_system.plot()
                    
                    
                    current_system.update_values()
                    current_system.calculate_free_energy()
                    current_system.write_initial_conditions(dynamic_path)
                    # data_writer.writerow(current_system.output())
                    
                    # print(current_system)
                    # current_system.plot()
                    # current_system.plot3d()
                    current_system.save_plot(iteration, file_path)
                    # current_system.plot()
                    