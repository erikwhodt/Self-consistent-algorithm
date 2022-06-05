import csv
import numpy as np
from real_space_utilities.source import Dynamic_lattice
import matplotlib.pyplot as plt



U_init = 9
U = 9

data = open(f"data/quench_data/testing/U=5.csv", 'w', newline='')
writer = csv.writer(data, delimiter=',')
writer.writerow(['time', 'mag_avg', 'mag_x', 'mag_y', 'mag_z', 'n_up', 'n_down'])


t = 1
alpha_init=0.0
alpha = 0
Q_array = '[pi_16, pi]'
time_step = 0.04
no_steps = 8000
filepath = f'data/dynamics_initial_values/U={U_init}_alpha={alpha_init}_{Q_array}.csv'

N = 20


Lattice = Dynamic_lattice(N, U, t, alpha, filepath)

for i in range(no_steps):
    title_string = fr"Quench from $U$ = {U_init} $\rightarrow$ {U}, $\alpha$ = {alpha_init} $\rightarrow$ {alpha} timestep: {i}"

    # if (i-1) % 10 == 0: 
    #     Lattice.save(i+1, title_string)
    # Lattice.save(i+1, title_string)
    
    print(f"timestep: {i+1}")
    Lattice.iterate_one_timestep_alternative(time_step)
    Lattice.update_values_alt(i, time_step)

    # if i%10 == 0:
    #     Lattice.plot()
    #     Lattice.plot_array(i)

time_array = np.linspace(0, no_steps*time_step, no_steps)

[writer.writerow([time_array[i], float(Lattice.m_tot_array[i]), float(Lattice.m_x_tot_array[i]), float(Lattice.m_y_tot_array[i]), float(Lattice.m_z_tot_array[i]), float(Lattice.n_up_avg[i]), float(Lattice.n_down_avg[i])]) for i in range(no_steps)]

Lattice.plot(1, title_string)
Lattice.plot(0, title_string)

Lattice.plot_array(i, title_string)
# ax.set_ylim(0.0, 0.5)


a = 2

