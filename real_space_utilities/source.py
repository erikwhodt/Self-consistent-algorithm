import numpy as np
from real_space_utilities.functions import *
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
import csv
from functions import matrix_wrapper_real, evolution_function
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, AutoMinorLocator
class Matrix:
    def __init__(self, lattice):
        self.matrix = matrix(lattice.n_array, lattice.m_array, lattice.m_x_array, lattice.m_y_array, lattice.m_z_array, lattice.psi, lattice.Q, lattice.N, lattice.U, lattice.mu, 0, lattice.t, lattice.alpha, lattice.direction, lattice.PM_flag, lattice.seed_flag)
        self.eig_val, self.eig_vec = np.linalg.eigh(self.matrix)
        self.lattice = lattice

    def obtain_new_values(self):
        self.lattice.energy = 0
        self.eig_vec = np.conj(self.eig_vec.T)
        if self.lattice.PM_flag:
            self.lattice.energy, self.lattice.n_array_new = matrix_wrapper_real(self.lattice.energy, self.lattice.PM_flag, self.lattice.N, self.lattice.beta, self.lattice.n_array_new, self.eig_vec, self.eig_val, self.lattice.m_x_array_new, self.lattice.m_y_array_new, self.lattice.m_z_array_new, self.lattice.m_array_new)
        else:
            self.lattice.energy, self.lattice.n_array_new, self.lattice.m_x_array_new, self.lattice.m_y_array_new, self.lattice.m_z_array_new, self.lattice.m_array_new = matrix_wrapper_real(self.lattice.energy, self.lattice.PM_flag, self.lattice.N, self.lattice.beta, self.lattice.n_array_new, self.eig_vec, self.eig_val, self.lattice.m_x_array_new, self.lattice.m_y_array_new, self.lattice.m_z_array_new, self.lattice.m_array_new)

class Lattice:

    def __init__(self, N, h_init, n_init, threshold, mu, beta, psi, Q, Q_text, U, t, alpha, direction, PM_flag, seed_flag):

        self.threshold = threshold
        self.N          = N
        self.mu         = mu
        self.beta       = beta
        self.psi        = psi
        self.Q          = Q
        self.Q_text     = Q_text
        self.U          = U
        self.t          = t
        self.energy     = 0
        self.alpha      = alpha
        self.direction  = direction
        self.did_it_converge = True
        self.PM_flag = PM_flag
        self.seed_flag = seed_flag
        self.first_iteration = True
        ################################################################################
        ## Initializing containers to hold current and new values between iterations
        ################################################################################

        self.m_array    = np.full(N**2, 0, dtype=np.double)
        self.m_x_array  = np.full(N**2, 0, dtype=np.double)
        self.m_y_array  = np.full(N**2,0, dtype=np.double)
        self.m_z_array  = np.full(N**2,0, dtype=np.double)

        if PM_flag:
            self.m_array_new    = np.full(N**2, 0, dtype=np.double)
        else:
            self.m_array_new = np.full(N**2, h_init, dtype=np.double)
        self.m_x_array_new  = np.full(N**2, 0, dtype=np.double)
        self.m_y_array_new  = np.full(N**2,0, dtype=np.double)
        self.m_z_array_new  = np.full(N**2,0, dtype=np.double)

        self.n_array        = np.full(N**2, n_init, dtype=np.double)
        self.n_array_new    = np.full(N**2, n_init, dtype=np.double)

    # def has_not_converged(self) -> bool:
    #     if (np.max(self.h_x_array-self.h_x_array_new)) > self.threshold or  abs(np.max(self.h_y_array-self.h_y_array_new)) > self.threshold or abs(np.max(self.h_z_array-self.h_z_array_new)) > self.threshold or abs(np.max(self.h_array-self.h_array_new)) > self.threshold or abs(np.max(self.n_array-self.n_array_new)) > self.threshold:
    #         return True
    #     else: return False

    def has_not_converged(self) -> bool:
        if np.max(np.abs(self.m_x_array-self.m_x_array_new)) > self.threshold or np.max(np.abs(self.m_y_array-self.m_y_array_new)) > self.threshold or np.max(np.abs(self.m_z_array-self.m_z_array_new)) > self.threshold or np.max(np.abs(self.n_array - self.n_array_new)) > self.threshold or self.first_iteration:
            self.first_iteration = False
            return True
        else: return False

    def update_values(self) -> None:
        self.m_array    = self.m_array_new.copy()
        self.m_x_array  = self.m_x_array_new.copy()
        self.m_y_array  = self.m_y_array_new.copy()
        self.m_z_array  = self.m_z_array_new.copy()
        self.n_array    = self.n_array_new.copy()

    def create_matrix(self) -> None:
        self.lattice_matrix = Matrix(self)
        self.seed_flag = False

    def obtain_new_values(self):
        self.lattice_matrix.obtain_new_values()

    # def calculate_free_energy(self):
    #     temp = 0
    #     for i in range(self.N**2):
    #         temp    += -self.U*self.n_array[i]**2 / 4 + self.U*(self.m_array[i])**2
    #     temp_2 = (1/(np.average(self.n_array)*self.N**2))*(self.energy + temp)
    #     self.energy = temp_2

    def calculate_free_energy(self):
        temp = 0
        for i in range(self.N**2):
            temp    += -self.U*self.n_array[i]**2 / 4 + self.U*(self.m_array[i])**2
        temp_2 = (1/(self.N**2)) * (self.energy + temp)
        self.energy = temp_2


    def __repr__(self):
        return f"density(avg): {np.average(self.n_array):.4f} mag_x: {np.average(np.abs(self.m_x_array)):.4f} mag_y: {np.average(np.abs(self.m_y_array)):.4f} mag_z {np.average(np.abs(self.m_z_array)):.4f} Engy: {self.energy:.4f}\nError: mag(avg): {abs(np.max(self.m_array-self.m_array_new)):.5f} n(avg): {abs(np.max(self.n_array - self.n_array_new)):.5f}\n "

    def output(self):
        return [self.mu, self.alpha, self.direction, self.Q_text, np.average(self.n_array_new), np.average(self.m_array_new), self.energy, self.did_it_converge]

    def write_initial_conditions(self, dynamic_path):
        print("\n## Writing data to initial values file...")
        initial_file = open(f"{dynamic_path}U={self.U}_alpha={self.alpha}_{self.Q_text}.csv", 'w', newline='')
        writer = csv.writer(initial_file, delimiter=',')
        writer.writerow(['n_up', 'n_down', 'n_flip', 'm_x', 'm_y', 'm_z', '+x_noflip_up','+x_noflip_down', '-x_noflip_up','-x_noflip_down', '+y_noflip_up','+y_noflip_down', '-y_noflip_up', '-y_noflip_down', '+x_flip_ud','+x_flip_du', '-x_flip_ud','-x_flip_du', '+y_flip_ud','+y_flip_du', '-y_flip_ud','-y_flip_du'])

        for i in range(self.N**2):
            n_u                 = 0
            n_d                 = 0
            n_flip              = 0
            m_x                 = 0
            m_y                 = 0
            m_z                 = 0
            ## Spin-conserving hopping elements
            plus_x_noflip_up    = 0
            minus_x_noflip_up   = 0
            plus_x_noflip_down  = 0
            minus_x_noflip_down = 0
            plus_y_noflip_up    = 0
            minus_y_noflip_up   = 0
            plus_y_noflip_down  = 0
            minus_y_noflip_down = 0
            ## Spin-flipping hopping elements
            plus_x_flip_ud      = 0
            minus_x_flip_ud     = 0
            plus_x_flip_du      = 0
            minus_x_flip_du     = 0
            plus_y_flip_ud      = 0
            minus_y_flip_ud     = 0
            plus_y_flip_du      = 0
            minus_y_flip_du     = 0

            ## Temporary variables

            up                   = 0
            down                 = 0
            plus_x_up           = 0
            plus_x_down         = 0
            minus_x_up          = 0
            minus_x_down        = 0
            plus_y_up           = 0
            plus_y_down         = 0
            minus_y_up          = 0
            minus_y_down        = 0


            for j in range(2*self.N**2):
                eigval = fermi(self.beta, 0, self.lattice_matrix.eig_val[j])
                up = self.lattice_matrix.eig_vec[j][i*2]
                down = self.lattice_matrix.eig_vec[j][i*2+1]

                if i < (self.N**2 -self.N):
                    plus_x_up       = np.conj(self.lattice_matrix.eig_vec[j][i*2 + self.N*2])
                    plus_x_down     = np.conj(self.lattice_matrix.eig_vec[j][i*2 + 1 + self.N*2])
                if i >= self.N:
                    minus_x_up      = np.conj(self.lattice_matrix.eig_vec[j][i*2 - self.N*2])
                    minus_x_down    = np.conj(self.lattice_matrix.eig_vec[j][i*2 + 1 - self.N*2])
                if i % self.N != 0:
                    plus_y_up       = np.conj(self.lattice_matrix.eig_vec[j][i*2 - 2])
                    plus_y_down     = np.conj(self.lattice_matrix.eig_vec[j][i*2 - 1])
                if (i + 1) % self.N != 0:
                    minus_y_up      = np.conj(self.lattice_matrix.eig_vec[j][i*2 + 2])
                    minus_y_down    = np.conj(self.lattice_matrix.eig_vec[j][i*2 + 3])


                m_x     += 1/2 * (up*np.conj(down)+np.conj(up)*down)*eigval
                m_y     += -1j/2 * (up*np.conj(down)-np.conj(up)*down)*eigval
                m_z     += 1/2 *   (np.abs(up)**2-np.abs(down)**2)*eigval
                n_u     += (np.abs(up)**2) * eigval
                n_d     += (np.abs(down)**2) * eigval
                n_flip  += (up*np.conj(down)) * eigval

                ## spin-conserving hopping elements
                plus_x_noflip_up    += (up * plus_x_up) * eigval
                minus_x_noflip_up   += (up * minus_x_up) * eigval
                plus_x_noflip_down  += (down * plus_x_down) * eigval
                minus_x_noflip_down += (down * minus_x_down) * eigval
                plus_y_noflip_up    += (up * plus_y_up) * eigval
                minus_y_noflip_up   += (up * minus_y_up) * eigval
                plus_y_noflip_down  += (down * plus_y_down) * eigval
                minus_y_noflip_down += (down * minus_y_down) * eigval

                ## spin-flipping hopping elements
                plus_x_flip_ud      += (up * plus_x_down) * eigval
                minus_x_flip_ud     += (up * minus_x_down) * eigval
                plus_x_flip_du      += (down * plus_x_up) * eigval
                minus_x_flip_du     += (down * minus_x_up) * eigval
                plus_y_flip_ud      += (up * plus_y_down) * eigval
                minus_y_flip_ud     += (up * minus_y_down) * eigval
                plus_y_flip_du      += (down * plus_y_up) * eigval
                minus_y_flip_du     += (down * minus_y_up) * eigval

            writer.writerow([n_u, n_d, n_flip, m_x, m_y, m_z, plus_x_noflip_up, plus_x_noflip_down, minus_x_noflip_up, minus_x_noflip_down, plus_y_noflip_up, plus_y_noflip_down, minus_y_noflip_up, minus_y_noflip_down, plus_x_flip_ud, plus_x_flip_du, minus_x_flip_ud, minus_x_flip_du,  plus_y_flip_ud, plus_y_flip_du, minus_y_flip_ud, minus_y_flip_du])
        print("\n## Finished ")
        initial_file.close()

    def plot(self):
        fig, (ax_1,ax_2) = plt.subplots(1,2, figsize=(5.0,4.0), sharey=True, sharex=True)
        cbar_ax = fig.add_axes([0.85, 0.1075, 0.025, 0.775])
        norm = matplotlib.colors.Normalize(vmin=0.0,vmax=0.5,clip=False)
        X, Y    = np.meshgrid(np.linspace(1,self.N, self.N), np.linspace(1, self.N,self.N))
        ax_1.quiver(X,Y, np.reshape(self.m_x_array_new, (self.N, self.N)), np.reshape(self.m_y_array_new, (self.N, self.N)), pivot='middle', norm=norm, cmap='seismic')
        im = ax_2.pcolormesh(X, Y, np.reshape(self.m_z_array_new, (self.N, self.N)), shading='auto', cmap='seismic', vmax=0.5, vmin=-0.5)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'$\langle m_{z}\rangle$', rotation=270)
        fig.supxlabel('site number x', fontsize='medium')
        
        fig.supylabel('site number y', fontsize='medium')
        
        # ax_1.set_xlim(0.5, self.N+0.5)
        # ax_1.set_ylim(0.5, self.N+0.5)

        ax_1.yaxis.set_major_locator(MultipleLocator(1))
        ax_2.yaxis.set_major_locator(MultipleLocator(1))

        ax_1.xaxis.set_major_locator(MultipleLocator(1))
        ax_2.xaxis.set_major_locator(MultipleLocator(1))

        ax_1.tick_params(axis="both",direction="in", which='minor', length=2.0, right=True, top=True)
        ax_1.tick_params(axis="both",direction="in", which='major', length=2.0, right=True, top=True)

        ax_2.tick_params(axis="both",direction="in", which='minor', length=2.0, right=True, top=True)
        ax_2.tick_params(axis="both",direction="in", which='major', length=2.0, right=True, top=True)


        plt.show()

    def plot_xy(self, size):
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        norm = matplotlib.colors.Normalize(vmin=0.0,vmax=0.5,clip=False)
        X, Y    = np.meshgrid(np.linspace(1,self.N, self.N), np.linspace(1, self.N,self.N))
        ax.quiver(X,Y, np.reshape(self.m_x_array_new, (self.N, self.N)), np.reshape(self.m_y_array_new, (self.N, self.N)), pivot='middle', norm=norm, cmap='seismic')
        ax.set_xlabel('site number x')
        ax.set_xlabel('site number x')
        ax.set_ylabel('site number y')
        ax.set_ylabel('site number y')
        # ax.set_xlim(0.5, self.N+0.5)
        # ax.set_xticks(np.arange(1, self.N,1))
        # ax.set_ylim(0.5, self.N+0.5)
        # ax.set_yticks(np.arange(1, self.N,1))

        ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_minor_locator(MultipleLocator(1))

        ax.xaxis.set_major_locator(MultipleLocator(1))
        # ax.xaxis.set_minor_locator(MultipleLocator(1))

        ax.tick_params(axis="both",direction="in", which='minor', length=2.0, right=True, top=True)
        ax.tick_params(axis="both",direction="in", which='major', length=2.0, right=True, top=True)
        plt.show()

 


    def save_plot(self, iteration, file_path):
        fig, (ax_1,ax_2) = plt.subplots(1,2)
        X, Y    = np.meshgrid(np.linspace(0,self.N, self.N), np.linspace(0, self.N,self.N))
        ax_1.quiver(X,Y, np.reshape(self.m_x_array_new, (self.N, self.N)), np.reshape(self.m_y_array_new, (self.N, self.N)), pivot='middle')
        ax_2.pcolormesh(X, Y, np.reshape(self.m_z_array_new, (self.N, self.N)), shading='auto', cmap='seismic', vmax=1.0, vmin=-1.0)
        plt.savefig(f'{file_path}/U={self.U}/mu={self.mu}/{self.Q_text}_{self.direction}_{iteration}.pdf')
        plt.clf()

    def plot3d(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y, Z    = np.meshgrid(np.linspace(0,self.N,self.N), np.linspace(0,self.N,self.N), np.array([0]))
        # ax.quiver(np.linspace(0,self.N,self.N), np.linspace(0,self.N,self.N), np.meshgrid(np.arange(0,1,1),self.m_x_array_new, self.m_y_array_new,self.m_z_array_new, color='blue', aa=True)
        ax.quiver(X,Y, Z, np.reshape(self.m_x_array_new, (self.N, self.N)), np.reshape(self.m_y_array_new, (self.N, self.N)),np.reshape(self.m_z_array_new, (self.N, self.N)), color='blue', aa=True)
        ax.set_zlim(-2.0,2.0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        pass

class Dynamic_lattice:

    def __init__(self, N, U , t, alpha, file_path):
        print("\n## Reading values...")
        self.N = N
        self.U = U
        self.t = t
        self.alpha = alpha

        ## Old arrays

        self.m_x_array          = np.full(N**2, 0, dtype=complex)
        self.m_y_array          = np.full(N**2, 0, dtype=complex)
        self.m_z_array          = np.full(N**2, 0, dtype=np.double)
        self.n_up_array         = np.full(N**2, 0, dtype=np.double)
        self.n_down_array       = np.full(N**2, 0, dtype=np.double)
        self.n_flip_array       = np.full(N**2, 0, dtype=complex)

        self.plus_x_noflip_u    = np.full(N**2, 0, dtype=complex)
        self.plus_x_noflip_d    = np.full(N**2, 0, dtype=complex)
        self.minus_x_noflip_u   = np.full(N**2, 0, dtype=complex)
        self.minus_x_noflip_d   = np.full(N**2, 0, dtype=complex)
        self.plus_y_noflip_u    = np.full(N**2, 0, dtype=complex)
        self.plus_y_noflip_d    = np.full(N**2, 0, dtype=complex)
        self.minus_y_noflip_u   = np.full(N**2, 0, dtype=complex)
        self.minus_y_noflip_d   = np.full(N**2, 0, dtype=complex)

        self.plus_x_flip_ud     = np.full(N**2, 0, dtype=complex)
        self.plus_x_flip_du     = np.full(N**2, 0, dtype=complex)
        self.minus_x_flip_ud    = np.full(N**2, 0, dtype=complex)
        self.minus_x_flip_du    = np.full(N**2, 0, dtype=complex)
        self.plus_y_flip_ud     = np.full(N**2, 0, dtype=complex)
        self.plus_y_flip_du     = np.full(N**2, 0, dtype=complex)
        self.minus_y_flip_ud    = np.full(N**2, 0, dtype=complex)
        self.minus_y_flip_du    = np.full(N**2, 0, dtype=complex)

        ## New arrays

        self.m_x_array_new          = np.full(N**2, 0, dtype=complex)
        self.m_y_array_new          = np.full(N**2, 0, dtype=complex)
        self.m_z_array_new          = np.full(N**2, 0, dtype=np.double)
        self.n_up_array_new         = np.full(N**2, 0, dtype=np.double)
        self.n_down_array_new       = np.full(N**2, 0, dtype=np.double)
        self.n_flip_array_new       = np.full(N**2, 0, dtype=complex)

        self.plus_x_noflip_u_new    = np.full(N**2, 0, dtype=complex)
        self.plus_x_noflip_d_new    = np.full(N**2, 0, dtype=complex)
        self.minus_x_noflip_u_new   = np.full(N**2, 0, dtype=complex)
        self.minus_x_noflip_d_new   = np.full(N**2, 0, dtype=complex)
        self.plus_y_noflip_u_new    = np.full(N**2, 0, dtype=complex)
        self.plus_y_noflip_d_new    = np.full(N**2, 0, dtype=complex)
        self.minus_y_noflip_u_new   = np.full(N**2, 0, dtype=complex)
        self.minus_y_noflip_d_new   = np.full(N**2, 0, dtype=complex)

        self.plus_x_flip_ud_new     = np.full(N**2, 0, dtype=complex)
        self.plus_x_flip_du_new     = np.full(N**2, 0, dtype=complex)
        self.minus_x_flip_ud_new    = np.full(N**2, 0, dtype=complex)
        self.minus_x_flip_du_new    = np.full(N**2, 0, dtype=complex)
        self.plus_y_flip_ud_new     = np.full(N**2, 0, dtype=complex)
        self.plus_y_flip_du_new     = np.full(N**2, 0, dtype=complex)
        self.minus_y_flip_ud_new    = np.full(N**2, 0, dtype=complex)
        self.minus_y_flip_du_new    = np.full(N**2, 0, dtype=complex)

        self.m_x_array_der          = np.full((N**2, 4), 0, dtype=complex)
        self.m_y_array_der          = np.full((N**2, 4), 0, dtype=complex)
        self.m_z_array_der          = np.full((N**2, 4), 0, dtype=np.double)
        self.n_up_array_der         = np.full((N**2, 4), 0, dtype=np.double)
        self.n_down_array_der       = np.full((N**2, 4), 0, dtype=np.double)
        self.n_flip_array_der       = np.full((N**2, 4), 0, dtype=complex)

        self.plus_x_noflip_u_der    = np.full((N**2, 4), 0, dtype=complex)
        self.plus_x_noflip_d_der    = np.full((N**2, 4), 0, dtype=complex)
        self.minus_x_noflip_u_der   = np.full((N**2, 4), 0, dtype=complex)
        self.minus_x_noflip_d_der   = np.full((N**2, 4), 0, dtype=complex)
        self.plus_y_noflip_u_der    = np.full((N**2, 4), 0, dtype=complex)
        self.plus_y_noflip_d_der    = np.full((N**2, 4), 0, dtype=complex)
        self.minus_y_noflip_u_der   = np.full((N**2, 4), 0, dtype=complex)
        self.minus_y_noflip_d_der   = np.full((N**2, 4), 0, dtype=complex)

        self.plus_x_flip_ud_der     = np.full((N**2, 4), 0, dtype=complex)
        self.plus_x_flip_du_der     = np.full((N**2, 4), 0, dtype=complex)
        self.minus_x_flip_ud_der    = np.full((N**2, 4), 0, dtype=complex)
        self.minus_x_flip_du_der    = np.full((N**2, 4), 0, dtype=complex)
        self.plus_y_flip_ud_der     = np.full((N**2, 4), 0, dtype=complex)
        self.plus_y_flip_du_der     = np.full((N**2, 4), 0, dtype=complex)
        self.minus_y_flip_ud_der    = np.full((N**2, 4), 0, dtype=complex)
        self.minus_y_flip_du_der    = np.full((N**2, 4), 0, dtype=complex)

        self.m_x_array_temp         = np.full((N**2,4), 0, dtype=complex)
        self.m_y_array_temp         = np.full((N**2,4), 0, dtype=complex)
        self.m_z_array_temp         = np.full((N**2,4), 0, dtype=np.double)
        self.n_up_array_temp        = np.full((N**2,4), 0, dtype=np.double)
        self.n_down_array_temp      = np.full((N**2,4), 0, dtype=np.double)
        self.n_flip_array_temp      = np.full((N**2,4), 0, dtype=complex)

        self.plus_x_noflip_u_temp   = np.full((N**2,4), 0, dtype=complex)
        self.plus_x_noflip_d_temp   = np.full((N**2,4), 0, dtype=complex)
        self.minus_x_noflip_u_temp  = np.full((N**2,4), 0, dtype=complex)
        self.minus_x_noflip_d_temp  = np.full((N**2,4), 0, dtype=complex)
        self.plus_y_noflip_u_temp   = np.full((N**2,4), 0, dtype=complex)
        self.plus_y_noflip_d_temp   = np.full((N**2,4), 0, dtype=complex)
        self.minus_y_noflip_u_temp  = np.full((N**2,4), 0, dtype=complex)
        self.minus_y_noflip_d_temp  = np.full((N**2,4), 0, dtype=complex)

        self.plus_x_flip_ud_temp    = np.full((N**2,4), 0, dtype=complex)
        self.plus_x_flip_du_temp    = np.full((N**2,4), 0, dtype=complex)
        self.minus_x_flip_ud_temp   = np.full((N**2,4), 0, dtype=complex)
        self.minus_x_flip_du_temp   = np.full((N**2,4), 0, dtype=complex)
        self.plus_y_flip_ud_temp    = np.full((N**2,4), 0, dtype=complex)
        self.plus_y_flip_du_temp    = np.full((N**2,4), 0, dtype=complex)
        self.minus_y_flip_ud_temp   = np.full((N**2,4), 0, dtype=complex)
        self.minus_y_flip_du_temp   = np.full((N**2,4), 0, dtype=complex)

        self.n_up_avg = []
        self.n_down_avg = []
        self.m_tot_array = []
        self.m_x_tot_array = []
        self.m_y_tot_array = []
        self.m_z_tot_array = []
        self.time_array = []
        self.n_avg = []

        initial_values = open(file_path, 'r', newline='')
        reader = csv.reader(initial_values, delimiter=',')
        next(reader, None)
        row_no = 0
        for row in reader:
            self.m_x_array[row_no]      = complex(row[3])
            self.m_y_array[row_no]      = complex(row[4])
            self.m_z_array[row_no]      = float(row[5])
            self.n_up_array[row_no]     = float(row[0])
            self.n_down_array[row_no]   = float(row[1])
            self.n_flip_array[row_no]   = complex(row[2])

            self.plus_x_noflip_u[row_no]    = complex(row[6])
            self.plus_x_noflip_d[row_no]    = complex(row[7])
            self.minus_x_noflip_u[row_no]   = complex(row[8])
            self.minus_x_noflip_d[row_no]   = complex(row[9])
            self.plus_y_noflip_u[row_no]    = complex(row[10])
            self.plus_y_noflip_d[row_no]    = complex(row[11])
            self.minus_y_noflip_u[row_no]   = complex(row[12])
            self.minus_y_noflip_d[row_no]   = complex(row[13])

            self.plus_x_flip_ud[row_no]    = complex(row[14])
            self.plus_x_flip_du[row_no]    = complex(row[15])
            self.minus_x_flip_ud[row_no]   = complex(row[16])
            self.minus_x_flip_du[row_no]   = complex(row[17])
            self.plus_y_flip_ud[row_no]    = complex(row[18])
            self.plus_y_flip_du[row_no]    = complex(row[19])
            self.minus_y_flip_ud[row_no]   = complex(row[20])
            self.minus_y_flip_du[row_no]   = complex(row[21])

            self.m_x_array_temp[row_no][0]      = complex(row[3])
            self.m_y_array_temp[row_no][0]      = complex(row[4])
            self.m_z_array_temp[row_no][0]      = float(row[5])
            self.n_up_array_temp[row_no][0]     = float(row[0])
            self.n_down_array_temp[row_no][0]   = float(row[1])
            self.n_flip_array_temp[row_no][0]   = complex(row[2])

            self.plus_x_noflip_u_temp[row_no][0]    = complex(row[6])
            self.plus_x_noflip_d_temp[row_no][0]    = complex(row[7])
            self.minus_x_noflip_u_temp[row_no][0]   = complex(row[8])
            self.minus_x_noflip_d_temp[row_no][0]   = complex(row[9])
            self.plus_y_noflip_u_temp[row_no][0]    = complex(row[10])
            self.plus_y_noflip_d_temp[row_no][0]    = complex(row[11])
            self.minus_y_noflip_u_temp[row_no][0]   = complex(row[12])
            self.minus_y_noflip_d_temp[row_no][0]   = complex(row[13])

            self.plus_x_flip_ud_temp[row_no][0]    = complex(row[14])
            self.plus_x_flip_du_temp[row_no][0]    = complex(row[15])
            self.minus_x_flip_ud_temp[row_no][0]   = complex(row[16])
            self.minus_x_flip_du_temp[row_no][0]   = complex(row[17])
            self.plus_y_flip_ud_temp[row_no][0]    = complex(row[18])
            self.plus_y_flip_du_temp[row_no][0]    = complex(row[19])
            self.minus_y_flip_ud_temp[row_no][0]   = complex(row[20])
            self.minus_y_flip_du_temp[row_no][0]   = complex(row[21])

            row_no += 1
        print("\n## Finished")

    def iterate_one_timestep(self, time_step):
        for i in range(self.N**2):

            n_x_u       =   0
            n_minx_u    =   0
            n_y_u       =   0
            n_miny_u    =   0
            n_x_d       =   0
            n_minx_d    =   0
            n_y_d       =   0
            n_miny_d    =   0
            n_x_flip    =   0
            n_minx_flip =   0
            n_y_flip    =   0
            n_miny_flip =   0

            m_z_x       =   0
            m_z_minx    =   0
            m_z_y       =   0
            m_z_miny    =   0
            m_x_x       =   0
            m_x_minx    =   0
            m_x_y       =   0
            m_x_miny    =   0
            m_y_x       =   0
            m_y_minx    =   0
            m_y_y       =   0
            m_y_miny    =   0

            if i >= self.N:
                n_minx_u    = self.n_up_array[i-self.N]
                n_minx_d    = self.n_down_array[i-self.N]
                n_minx_flip = self.n_flip_array[i-self.N]
                m_z_minx    = self.m_z_array[i-self.N]
                m_x_minx    = self.m_x_array[i-self.N]
                m_y_minx    = self.m_y_array[i-self.N]
            if i < (self.N**2 - self.N):
                n_x_u       = self.n_up_array[i+self.N]
                n_x_d       = self.n_down_array[i+self.N]
                n_x_flip    = self.n_flip_array[i+self.N]
                m_z_x       = self.m_z_array[i+self.N]
                m_x_x       = self.m_x_array[i+self.N]
                m_y_x       = self.m_y_array[i+self.N]
            if (i % self.N) != 0:
                n_y_u       = self.n_up_array[i-1]
                n_y_d       = self.n_down_array[i-1]
                n_y_flip    = self.n_flip_array[i-1]
                m_z_y       = self.m_z_array[i-1]
                m_x_y       = self.m_x_array[i-1]
                m_y_y       = self.m_y_array[i-1]
            if (i + 1) % self.N != 0:
                n_miny_u    = self.n_up_array[i+1]
                n_miny_d    = self.n_down_array[i+1]
                n_miny_flip = self.n_flip_array[i+1]
                m_z_miny    = self.m_z_array[i+1]
                m_x_miny    = self.m_x_array[i+1]
                m_y_miny    = self.m_y_array[i+1]

            n_u_der_1, n_d_der_1, n_flip_der_1, plus_x_noflip_up_der_1, plus_x_noflip_down_der_1, minus_x_noflip_up_der_1, minus_x_noflip_down_der_1, \
            plus_y_noflip_up_der_1, plus_y_noflip_down_der_1, minus_y_noflip_up_der_1, minus_y_noflip_down_der_1, plus_x_flip_ud_der_1, plus_x_flip_du_der_1, \
            minus_x_flip_ud_der_1, minus_x_flip_du_der_1, plus_y_flip_ud_der_1, plus_y_flip_du_der_1, minus_y_flip_ud_der_1, minus_y_flip_du_der_1 \
            = evolution_function(self.t, self.alpha, self.U, self.n_up_array[i], self.n_down_array[i], n_x_u, n_minx_u, n_y_u, n_miny_u, n_x_d, n_minx_d, n_y_d, n_miny_d, n_x_flip, n_minx_flip, \
                        n_y_flip, n_miny_flip, self.n_flip_array[i], self.m_x_array[i], self.m_y_array[i], self.m_z_array[i], m_x_x, m_x_minx, m_x_y, m_x_miny, m_y_x, m_y_minx, m_y_y, m_y_miny, m_z_x, m_z_minx, m_z_y, m_z_miny ,self.plus_x_noflip_u[i], self.plus_x_noflip_d[i], self.minus_x_noflip_u[i], self.minus_x_noflip_d[i], \
                        self.plus_y_noflip_u[i], self.plus_y_noflip_d[i], self.minus_y_noflip_u[i], self.minus_y_noflip_d[i], self.plus_x_flip_ud[i], self.plus_x_flip_du[i], self.minus_x_flip_ud[i], \
                        self.minus_x_flip_du[i],  self.plus_y_flip_ud[i], self.plus_y_flip_du[i], self.minus_y_flip_ud[i], self.minus_y_flip_du[i])



            n_u_1                   = self.n_up_array[i]        + n_u_der_1                     * time_step/2
            n_d_1                   = self.n_down_array[i]      + n_d_der_1                     * time_step/2
            n_flip_1                = self.n_flip_array[i]      + n_flip_der_1                  * time_step/2

            m_x_1                   =  1/2 * (n_flip_1+np.conj(n_flip_1))
            m_y_1                   = -1j * 1/2 * (n_flip_1-np.conj(n_flip_1))
            m_z_1                   = 1/2 * (n_u_1 - n_d_1)


            if i < (self.N**2 - self.N):
                plus_x_noflip_up_1      = self.plus_x_noflip_u[i]   + plus_x_noflip_up_der_1        * time_step/2
                plus_x_noflip_down_1    = self.plus_x_noflip_d[i]   + plus_x_noflip_down_der_1      * time_step/2
                plus_x_flip_ud_1        = self.plus_x_flip_ud[i]    + plus_x_flip_ud_der_1          * time_step/2
                plus_x_flip_du_1        = self.plus_x_flip_du[i]    + plus_x_flip_du_der_1          * time_step/2
            else:
                plus_x_noflip_up_1      = 0
                plus_x_noflip_down_1    = 0
                plus_x_flip_ud_1        = 0
                plus_x_flip_du_1        = 0
            if i >= self.N:
                minus_x_noflip_up_1     = self.minus_x_noflip_u[i]  + minus_x_noflip_up_der_1       * time_step/2
                minus_x_noflip_down_1   = self.minus_x_noflip_d[i]  + minus_x_noflip_down_der_1     * time_step/2
                minus_x_flip_ud_1       = self.minus_x_flip_ud[i]   + minus_x_flip_ud_der_1         * time_step/2
                minus_x_flip_du_1       = self.minus_x_flip_du[i]   + minus_x_flip_du_der_1         * time_step/2
            else:
                minus_x_noflip_up_1     = 0
                minus_x_noflip_down_1   = 0
                minus_x_flip_ud_1       = 0
                minus_x_flip_du_1       = 0
            if (i % self.N) != 0:
                plus_y_noflip_up_1      = self.plus_y_noflip_u[i]   + plus_y_noflip_up_der_1        * time_step/2
                plus_y_noflip_down_1    = self.plus_y_noflip_d[i]   + plus_y_noflip_down_der_1      * time_step/2
                plus_y_flip_ud_1        = self.plus_y_flip_ud[i]    + plus_y_flip_ud_der_1          * time_step/2
                plus_y_flip_du_1        = self.plus_y_flip_du[i]    + plus_y_flip_du_der_1          * time_step/2
            else:
                plus_y_noflip_up_1      = 0
                plus_y_noflip_down_1    = 0
                plus_y_flip_ud_1        = 0
                plus_y_flip_du_1        = 0
            if (i + 1) % self.N != 0:
                minus_y_noflip_up_1     = self.minus_y_noflip_u[i]  + minus_y_noflip_up_der_1       * time_step/2
                minus_y_noflip_down_1   = self.minus_y_noflip_d[i]  + minus_y_noflip_down_der_1     * time_step/2
                minus_y_flip_ud_1       = self.minus_y_flip_ud[i]   + minus_y_flip_ud_der_1         * time_step/2
                minus_y_flip_du_1       = self.minus_y_flip_du[i]   + minus_y_flip_du_der_1         * time_step/2
            else:
                minus_y_noflip_up_1     = 0
                minus_y_noflip_down_1   = 0
                minus_y_flip_ud_1       = 0
                minus_y_flip_du_1       = 0


            n_u_der_2, n_d_der_2, n_flip_der_2, plus_x_noflip_up_der_2, plus_x_noflip_down_der_2, minus_x_noflip_up_der_2, minus_x_noflip_down_der_2, \
            plus_y_noflip_up_der_2, plus_y_noflip_down_der_2, minus_y_noflip_up_der_2, minus_y_noflip_down_der_2, plus_x_flip_ud_der_2, plus_x_flip_du_der_2, \
            minus_x_flip_ud_der_2, minus_x_flip_du_der_2, plus_y_flip_ud_der_2, plus_y_flip_du_der_2, minus_y_flip_ud_der_2, minus_y_flip_du_der_2 \
            = evolution_function(self.t, self.alpha, self.U, n_u_1, n_d_1, n_x_u, n_minx_u, n_y_u, n_miny_u, n_x_d, n_minx_d, n_y_d, n_miny_d, n_x_flip, n_minx_flip, \
                        n_y_flip, n_miny_flip, n_flip_1, m_x_1, m_y_1, m_z_1, m_x_x, m_x_minx, m_x_y, m_x_miny, m_y_x, m_y_minx, m_y_y, m_y_miny, m_z_x, m_z_minx, m_z_y, m_z_miny ,plus_x_noflip_up_1, plus_x_noflip_down_1, minus_x_noflip_up_1, minus_x_noflip_down_1, \
                        plus_y_noflip_up_1, plus_y_noflip_down_1, minus_y_noflip_up_1, minus_y_noflip_down_1, plus_x_flip_ud_1, plus_x_flip_du_1, minus_x_flip_ud_1, \
                        minus_x_flip_du_1,  plus_y_flip_ud_1, plus_y_flip_du_1, minus_y_flip_ud_1, minus_y_flip_du_1)

            n_u_2                   = self.n_up_array[i]        + n_u_der_2                     * time_step/2
            n_d_2                   = self.n_down_array[i]      + n_d_der_2                     * time_step/2
            n_flip_2                = self.n_flip_array[i]      + n_flip_der_2                  * time_step/2

            m_x_2                   =  1/2 * (n_flip_2+np.conj(n_flip_2))
            m_y_2                   = -1j * 1/2 * (n_flip_2-np.conj(n_flip_2))
            m_z_2                   = 1/2 * (n_u_2 - n_d_2)


            if i < (self.N**2 - self.N):
                plus_x_noflip_up_2      = self.plus_x_noflip_u[i]   + plus_x_noflip_up_der_2        * time_step/2
                plus_x_noflip_down_2    = self.plus_x_noflip_d[i]   + plus_x_noflip_down_der_2      * time_step/2
                plus_x_flip_ud_2        = self.plus_x_flip_ud[i]    + plus_x_flip_ud_der_2          * time_step/2
                plus_x_flip_du_2        = self.plus_x_flip_du[i]    + plus_x_flip_du_der_2          * time_step/2
            else:
                plus_x_noflip_up_2      = 0
                plus_x_noflip_down_2    = 0
                plus_x_flip_ud_2        = 0
                plus_x_flip_du_2        = 0
            if i >= self.N:
                minus_x_noflip_up_2     = self.minus_x_noflip_u[i]  + minus_x_noflip_up_der_2       * time_step/2
                minus_x_noflip_down_2   = self.minus_x_noflip_d[i]  + minus_x_noflip_down_der_2     * time_step/2
                minus_x_flip_ud_2       = self.minus_x_flip_ud[i]   + minus_x_flip_ud_der_2         * time_step/2
                minus_x_flip_du_2       = self.minus_x_flip_du[i]   + minus_x_flip_du_der_2         * time_step/2
            else:
                minus_x_noflip_up_2     = 0
                minus_x_noflip_down_2   = 0
                minus_x_flip_ud_2       = 0
                minus_x_flip_du_2       = 0
            if (i % self.N) != 0:
                plus_y_noflip_up_2      = self.plus_y_noflip_u[i]   + plus_y_noflip_up_der_2        * time_step/2
                plus_y_noflip_down_2    = self.plus_y_noflip_d[i]   + plus_y_noflip_down_der_2      * time_step/2
                plus_y_flip_ud_2        = self.plus_y_flip_ud[i]    + plus_y_flip_ud_der_2          * time_step/2
                plus_y_flip_du_2        = self.plus_y_flip_du[i]    + plus_y_flip_du_der_2          * time_step/2
            else:
                plus_y_noflip_up_2      = 0
                plus_y_noflip_down_2    = 0
                plus_y_flip_ud_2        = 0
                plus_y_flip_du_2        = 0
            if (i + 1) % self.N != 0:
                minus_y_noflip_up_2     = self.minus_y_noflip_u[i]  + minus_y_noflip_up_der_2       * time_step/2
                minus_y_noflip_down_2   = self.minus_y_noflip_d[i]  + minus_y_noflip_down_der_2     * time_step/2
                minus_y_flip_ud_2       = self.minus_y_flip_ud[i]   + minus_y_flip_ud_der_2         * time_step/2
                minus_y_flip_du_2       = self.minus_y_flip_du[i]   + minus_y_flip_du_der_2         * time_step/2
            else:
                minus_y_noflip_up_2     = 0
                minus_y_noflip_down_2   = 0
                minus_y_flip_ud_2       = 0
                minus_y_flip_du_2       = 0

            n_u_der_3, n_d_der_3, n_flip_der_3, plus_x_noflip_up_der_3, plus_x_noflip_down_der_3, minus_x_noflip_up_der_3, minus_x_noflip_down_der_3, \
            plus_y_noflip_up_der_3, plus_y_noflip_down_der_3, minus_y_noflip_up_der_3, minus_y_noflip_down_der_3, plus_x_flip_ud_der_3, plus_x_flip_du_der_3, \
            minus_x_flip_ud_der_3, minus_x_flip_du_der_3, plus_y_flip_ud_der_3, plus_y_flip_du_der_3, minus_y_flip_ud_der_3, minus_y_flip_du_der_3 \
            = evolution_function(self.t, self.alpha, self.U, n_u_2, n_d_2, n_x_u, n_minx_u, n_y_u, n_miny_u, n_x_d, n_minx_d, n_y_d, n_miny_d, n_x_flip, n_minx_flip, \
                        n_y_flip, n_miny_flip, n_flip_2, m_x_2, m_y_2, m_z_2, m_x_x, m_x_minx, m_x_y, m_x_miny, m_y_x, m_y_minx, m_y_y, m_y_miny, m_z_x, m_z_minx, m_z_y, m_z_miny ,plus_x_noflip_up_2, plus_x_noflip_down_2, minus_x_noflip_up_2, minus_x_noflip_down_2, \
                        plus_y_noflip_up_2, plus_y_noflip_down_2, minus_y_noflip_up_2, minus_y_noflip_down_2, plus_x_flip_ud_2, plus_x_flip_du_2, minus_x_flip_ud_2, \
                        minus_x_flip_du_2,  plus_y_flip_ud_2, plus_y_flip_du_2, minus_y_flip_ud_2, minus_y_flip_du_2)

            n_u_3                   = self.n_up_array[i]        + n_u_der_3                     * time_step
            n_d_3                   = self.n_down_array[i]      + n_d_der_3                     * time_step
            n_flip_3                = self.n_flip_array[i]      + n_flip_der_3                  * time_step

            m_x_3                   =  1/2 * (n_flip_3+np.conj(n_flip_3))
            m_y_3                   = -1j * 1/2 * (n_flip_3-np.conj(n_flip_3))
            m_z_3                   = 1/2 * (n_u_3 - n_d_3)

            if i < (self.N**2 - self.N):
                plus_x_noflip_up_3      = self.plus_x_noflip_u[i]   + plus_x_noflip_up_der_3        * time_step
                plus_x_noflip_down_3    = self.plus_x_noflip_d[i]   + plus_x_noflip_down_der_3      * time_step
                plus_x_flip_ud_3        = self.plus_x_flip_ud[i]    + plus_x_flip_ud_der_3          * time_step
                plus_x_flip_du_3        = self.plus_x_flip_du[i]    + plus_x_flip_du_der_3          * time_step
            else:
                plus_x_noflip_up_3      = 0
                plus_x_noflip_down_3    = 0
                plus_x_flip_ud_3        = 0
                plus_x_flip_du_3        = 0
            if i >= self.N:
                minus_x_noflip_up_3     = self.minus_x_noflip_u[i]  + minus_x_noflip_up_der_3       * time_step
                minus_x_noflip_down_3   = self.minus_x_noflip_d[i]  + minus_x_noflip_down_der_3     * time_step
                minus_x_flip_ud_3       = self.minus_x_flip_ud[i]   + minus_x_flip_ud_der_3         * time_step
                minus_x_flip_du_3       = self.minus_x_flip_du[i]   + minus_x_flip_du_der_3         * time_step
            else:
                minus_x_noflip_up_3     = 0
                minus_x_noflip_down_3   = 0
                minus_x_flip_ud_3       = 0
                minus_x_flip_du_3       = 0
            if (i % self.N) != 0:
                plus_y_noflip_up_3      = self.plus_y_noflip_u[i]   + plus_y_noflip_up_der_3        * time_step
                plus_y_noflip_down_3    = self.plus_y_noflip_d[i]   + plus_y_noflip_down_der_3      * time_step
                plus_y_flip_ud_3        = self.plus_y_flip_ud[i]    + plus_y_flip_ud_der_3          * time_step
                plus_y_flip_du_3        = self.plus_y_flip_du[i]    + plus_y_flip_du_der_3          * time_step
            else:
                plus_y_noflip_up_3      = 0
                plus_y_noflip_down_3    = 0
                plus_y_flip_ud_3        = 0
                plus_y_flip_du_3        = 0
            if (i + 1) % self.N != 0:
                minus_y_noflip_up_3     = self.minus_y_noflip_u[i]  + minus_y_noflip_up_der_3       * time_step
                minus_y_noflip_down_3   = self.minus_y_noflip_d[i]  + minus_y_noflip_down_der_3     * time_step
                minus_y_flip_ud_3       = self.minus_y_flip_ud[i]   + minus_y_flip_ud_der_3         * time_step
                minus_y_flip_du_3       = self.minus_y_flip_du[i]   + minus_y_flip_du_der_3         * time_step
            else:
                minus_y_noflip_up_3     = 0
                minus_y_noflip_down_3   = 0
                minus_y_flip_ud_3       = 0
                minus_y_flip_du_3       = 0

            n_u_der_4, n_d_der_4, n_flip_der_4, plus_x_noflip_up_der_4, plus_x_noflip_down_der_4, minus_x_noflip_up_der_4, minus_x_noflip_down_der_4, \
            plus_y_noflip_up_der_4, plus_y_noflip_down_der_4, minus_y_noflip_up_der_4, minus_y_noflip_down_der_4, plus_x_flip_ud_der_4, plus_x_flip_du_der_4, \
            minus_x_flip_ud_der_4, minus_x_flip_du_der_4, plus_y_flip_ud_der_4, plus_y_flip_du_der_4, minus_y_flip_ud_der_4, minus_y_flip_du_der_4 \
            = evolution_function(self.t, self.alpha, self.U, n_u_3, n_d_3, n_x_u, n_minx_u, n_y_u, n_miny_u, n_x_d, n_minx_d, n_y_d, n_miny_d, n_x_flip, n_minx_flip, \
                        n_y_flip, n_miny_flip, n_flip_3, m_x_3, m_y_3, m_z_3, m_x_x, m_x_minx, m_x_y, m_x_miny, m_y_x, m_y_minx, m_y_y, m_y_miny, m_z_x, m_z_minx, m_z_y, m_z_miny ,plus_x_noflip_up_3, plus_x_noflip_down_3, minus_x_noflip_up_3, minus_x_noflip_down_3, \
                        plus_y_noflip_up_3, plus_y_noflip_down_3, minus_y_noflip_up_3, minus_y_noflip_down_3, plus_x_flip_ud_3, plus_x_flip_du_3, minus_x_flip_ud_3, \
                        minus_x_flip_du_3,  plus_y_flip_ud_3, plus_y_flip_du_3, minus_y_flip_ud_3, minus_y_flip_du_3)

            self.n_up_array_new[i]          = self.n_up_array[i]    + (n_u_der_1 + 2*n_u_der_2 + 2*n_u_der_3 + n_u_der_4)               * time_step / 6
            self.n_down_array_new[i]        = self.n_down_array[i]  + (n_d_der_1 + 2*n_d_der_2 + 2*n_d_der_3 + n_d_der_4)               * time_step / 6
            self.n_flip_array_new[i]        = self.n_flip_array[i]  + (n_flip_der_1 + 2* n_flip_der_2 + 2* n_flip_der_3 + n_flip_der_4) * time_step / 6

            self.m_x_array_new[i] = 1/2 * (self.n_flip_array_new[i]+np.conj(self.n_flip_array_new[i]))
            self.m_y_array_new[i] = -1/2 * 1j*(self.n_flip_array_new[i]-np.conj(self.n_flip_array_new[i]))
            self.m_z_array_new[i] = 1/2 * (self.n_up_array_new[i]-self.n_down_array_new[i])

            if i < (self.N**2 - self.N):
                self.plus_x_noflip_u_new[i]    = self.plus_x_noflip_u[i]    + (plus_x_noflip_up_der_1       +plus_x_noflip_up_der_2 * 2     + plus_x_noflip_up_der_3 * 2    + plus_x_noflip_up_der_4)       * time_step / 6
                self.plus_x_noflip_d_new[i]    = self.plus_x_noflip_d[i]    + (plus_x_noflip_down_der_1     +plus_x_noflip_down_der_2 * 2   + plus_x_noflip_down_der_3 * 2  + plus_x_noflip_down_der_4)     * time_step / 6
                self.plus_x_flip_ud_new[i]     = self.plus_x_flip_ud[i]     + (plus_x_flip_ud_der_1         + 2*plus_x_flip_ud_der_2        + 2* plus_x_flip_ud_der_3       + plus_x_flip_ud_der_4)         * time_step / 6
                self.plus_x_flip_du_new[i]     = self.plus_x_flip_du[i]     + (plus_x_flip_du_der_1         + 2*plus_x_flip_du_der_2        + 2* plus_x_flip_du_der_3       + plus_x_flip_du_der_4)         * time_step / 6
            else:
                self.plus_x_noflip_u_new[i]    = 0
                self.plus_x_noflip_d_new[i]    = 0
                self.plus_x_flip_ud_new[i]     = 0
                self.plus_x_flip_du_new[i]     = 0

            if i >= self.N:
                self.minus_x_flip_ud_new[i]    = self.minus_x_flip_ud[i]    + (minus_x_flip_ud_der_1        + 2*minus_x_flip_ud_der_2       + 2* minus_x_flip_ud_der_3      + minus_x_flip_ud_der_4)        * time_step / 6
                self.minus_x_flip_du_new[i]    = self.minus_x_flip_du[i]    + (minus_x_flip_du_der_1        + 2*minus_x_flip_du_der_2       + 2* minus_x_flip_du_der_3      + minus_x_flip_du_der_4)        * time_step / 6
                self.minus_x_noflip_u_new[i]   = self.minus_x_noflip_u[i]   + (minus_x_noflip_up_der_1      +minus_x_noflip_up_der_2 * 2    + minus_x_noflip_up_der_3 * 2   + minus_x_noflip_up_der_4)      * time_step / 6
                self.minus_x_noflip_d_new[i]   = self.minus_x_noflip_d[i]   + (minus_x_noflip_down_der_1    +minus_x_noflip_down_der_2 * 2  + minus_x_noflip_down_der_3 * 2 + minus_x_noflip_down_der_4)    * time_step / 6

            else:
                self.minus_x_flip_ud_new[i]    = 0
                self.minus_x_flip_du_new[i]    = 0
                self.minus_x_noflip_u_new[i]   = 0
                self.minus_x_noflip_d_new[i]   = 0

            if (i % self.N) != 0:
                self.plus_y_noflip_u_new[i]    = self.plus_y_noflip_u[i]    + (plus_y_noflip_up_der_1       +plus_y_noflip_up_der_2 * 2     + plus_y_noflip_up_der_3 * 2    + plus_y_noflip_up_der_4)       * time_step / 6
                self.plus_y_noflip_d_new[i]    = self.plus_y_noflip_d[i]    + (plus_y_noflip_down_der_1     +plus_y_noflip_down_der_2 * 2   + plus_y_noflip_down_der_3 * 2  + plus_y_noflip_down_der_4)     * time_step / 6
                self.plus_y_flip_ud_new[i]     = self.plus_y_flip_ud[i]     + (plus_y_flip_ud_der_1         + 2*plus_y_flip_ud_der_2        + 2* plus_y_flip_ud_der_3       + plus_y_flip_ud_der_4)         * time_step / 6
                self.plus_y_flip_du_new[i]     = self.plus_y_flip_du[i]     + (plus_y_flip_du_der_1         + 2*plus_y_flip_du_der_2        + 2* plus_y_flip_du_der_3       + plus_y_flip_du_der_4)         * time_step / 6

            else:
                self.plus_y_noflip_u_new[i]    = 0
                self.plus_y_noflip_d_new[i]    = 0
                self.plus_y_flip_ud_new[i]     = 0
                self.plus_y_flip_du_new[i]     = 0

            if (i + 1) % self.N != 0:
                self.minus_y_flip_ud_new[i]    = self.minus_y_flip_ud[i]    + (minus_y_flip_ud_der_1        + 2*minus_y_flip_ud_der_2       + 2* minus_y_flip_ud_der_3      + minus_y_flip_ud_der_4)        * time_step / 6
                self.minus_y_flip_du_new[i]    = self.minus_y_flip_du[i]    + (minus_y_flip_du_der_1        + 2*minus_y_flip_du_der_2       + 2* minus_y_flip_du_der_3      + minus_y_flip_du_der_4)        * time_step / 6
                self.minus_y_noflip_u_new[i]   = self.minus_y_noflip_u[i]   + (minus_y_noflip_up_der_1      +minus_y_noflip_up_der_2 * 2    + minus_y_noflip_up_der_3 * 2   + minus_y_noflip_up_der_4)      * time_step / 6
                self.minus_y_noflip_d_new[i]   = self.minus_y_noflip_d[i]   + (minus_y_noflip_down_der_1    +minus_y_noflip_down_der_2 * 2  + minus_y_noflip_down_der_3 * 2 + minus_y_noflip_down_der_4)    * time_step / 6

            else:
                self.minus_y_flip_ud_new[i]    = 0
                self.minus_y_flip_du_new[i]    = 0
                self.minus_y_noflip_u_new[i]   = 0
                self.minus_y_noflip_d_new[i]   = 0


    def iterate_one_timestep_alternative(self, time_step):
        for j in range(4):
            for i in range(self.N**2):

                if i == 0:
                    self.n_up_array_der[i][j], self.n_down_array_der[i][j], self.n_flip_array_der[i][j], self.plus_x_noflip_u_der[i][j], self.plus_x_noflip_d_der[i][j], self.minus_x_noflip_u_der[i][j], self.minus_x_noflip_d_der[i][j], \
                    self.plus_y_noflip_u_der[i][j], self.plus_y_noflip_d_der[i][j], self.minus_y_noflip_u_der[i][j], self.minus_y_noflip_d_der[i][j], self.plus_x_flip_ud_der[i][j], self.plus_x_flip_du_der[i][j], \
                    self.minus_x_flip_ud_der[i][j], self.minus_x_flip_du_der[i][j], self.plus_y_flip_ud_der[i][j], self.plus_y_flip_du_der[i][j], self.minus_y_flip_ud_der[i][j], self.minus_y_flip_du_der[i][j] \
                    = evolution_function(self.t, self.alpha, self.U, self.n_up_array_temp[i][j], self.n_down_array_temp[i][j], self.n_up_array_temp[i+self.N][j], 0, 0, self.n_up_array_temp[i+1][j], \
                        self.n_down_array_temp[i+self.N][j], 0, 0, self.n_down_array_temp[i+1][j], self.n_flip_array_temp[i+self.N][j], 0, \
                        0, self.n_flip_array_temp[i+1][j], self.n_flip_array_temp[i][j], self.m_x_array_temp[i][j], self.m_y_array_temp[i][j], self.m_z_array_temp[i][j], self.m_x_array_temp[i+self.N][j], \
                        0, 0, self.m_x_array_temp[i+1][j], self.m_y_array_temp[i+self.N][j], 0, 0, self.m_y_array_temp[i+1][j], \
                        self.m_z_array_temp[i+self.N][j], 0, 0, self.m_z_array_temp[i+1][j] ,self.plus_x_noflip_u_temp[i][j], self.plus_x_noflip_d_temp[i][j], 0, 0, \
                        0, 0, self.minus_y_noflip_u_temp[i][j], self.minus_y_noflip_d_temp[i][j], self.plus_x_flip_ud_temp[i][j], self.plus_x_flip_du_temp[i][j], \
                        0, 0, 0, 0, self.minus_y_flip_ud_temp[i][j], self.minus_y_flip_du_temp[i][j])
                elif i == self.N-1:
                    self.n_up_array_der[i][j], self.n_down_array_der[i][j], self.n_flip_array_der[i][j], self.plus_x_noflip_u_der[i][j], self.plus_x_noflip_d_der[i][j], self.minus_x_noflip_u_der[i][j], self.minus_x_noflip_d_der[i][j], \
                    self.plus_y_noflip_u_der[i][j], self.plus_y_noflip_d_der[i][j], self.minus_y_noflip_u_der[i][j], self.minus_y_noflip_d_der[i][j], self.plus_x_flip_ud_der[i][j], self.plus_x_flip_du_der[i][j], \
                    self.minus_x_flip_ud_der[i][j], self.minus_x_flip_du_der[i][j], self.plus_y_flip_ud_der[i][j], self.plus_y_flip_du_der[i][j], self.minus_y_flip_ud_der[i][j], self.minus_y_flip_du_der[i][j] \
                    = evolution_function(self.t, self.alpha, self.U, self.n_up_array_temp[i][j], self.n_down_array_temp[i][j], self.n_up_array_temp[i+self.N][j], 0, self.n_up_array_temp[i-1][j], 0, \
                        self.n_down_array_temp[i+self.N][j], 0, self.n_down_array_temp[i-1][j], 0, self.n_flip_array_temp[i+self.N][j], 0, \
                        self.n_flip_array_temp[i-1][j], 0, self.n_flip_array_temp[i][j], self.m_x_array_temp[i][j], self.m_y_array_temp[i][j], self.m_z_array_temp[i][j], self.m_x_array_temp[i+self.N][j], \
                        0, self.m_x_array_temp[i-1][j], 0, self.m_y_array_temp[i+self.N][j], 0, self.m_y_array_temp[i-1][j], 0, \
                        self.m_z_array_temp[i+self.N][j], 0, self.m_z_array_temp[i-1][j], 0 ,self.plus_x_noflip_u_temp[i][j], self.plus_x_noflip_d_temp[i][j], 0, 0, \
                        self.plus_y_noflip_u_temp[i][j], self.plus_y_noflip_d_temp[i][j], 0, 0, self.plus_x_flip_ud_temp[i][j], self.plus_x_flip_du_temp[i][j], \
                        0, 0, self.plus_y_flip_ud_temp[i][j], self.plus_y_flip_du_temp[i][j], 0, 0)
                elif i == self.N**2-1:
                    self.n_up_array_der[i][j], self.n_down_array_der[i][j], self.n_flip_array_der[i][j], self.plus_x_noflip_u_der[i][j], self.plus_x_noflip_d_der[i][j], self.minus_x_noflip_u_der[i][j], self.minus_x_noflip_d_der[i][j], \
                    self.plus_y_noflip_u_der[i][j], self.plus_y_noflip_d_der[i][j], self.minus_y_noflip_u_der[i][j], self.minus_y_noflip_d_der[i][j], self.plus_x_flip_ud_der[i][j], self.plus_x_flip_du_der[i][j], \
                    self.minus_x_flip_ud_der[i][j], self.minus_x_flip_du_der[i][j], self.plus_y_flip_ud_der[i][j], self.plus_y_flip_du_der[i][j], self.minus_y_flip_ud_der[i][j], self.minus_y_flip_du_der[i][j] \
                    = evolution_function(self.t, self.alpha, self.U, self.n_up_array_temp[i][j], self.n_down_array_temp[i][j], 0, self.n_up_array_temp[i-self.N][j], self.n_up_array_temp[i-1][j], 0, \
                        0, self.n_down_array_temp[i-self.N][j], self.n_down_array_temp[i-1][j], 0, 0, self.n_flip_array_temp[i-self.N][j], \
                        self.n_flip_array_temp[i-1][j], 0, self.n_flip_array_temp[i][j], self.m_x_array_temp[i][j], self.m_y_array_temp[i][j], self.m_z_array_temp[i][j], 0, \
                        self.m_x_array_temp[i-self.N][j], self.m_x_array_temp[i-1][j], 0,0, self.m_y_array_temp[i-self.N][j], self.m_y_array_temp[i-1][j], 0, \
                        0, self.m_z_array_temp[i-self.N][j], 0,0 ,0, 0, self.minus_x_noflip_u_temp[i][j], self.minus_x_noflip_d_temp[i][j], \
                        self.plus_y_noflip_u_temp[i][j], self.plus_y_noflip_d_temp[i][j], 0, 0, 0, 0, \
                        self.minus_x_flip_ud_temp[i][j], self.minus_x_flip_du_temp[i][j], self.plus_y_flip_ud_temp[i][j], self.plus_y_flip_du_temp[i][j], 0, 0)
                elif i == (self.N**2-self.N):
                    self.n_up_array_der[i][j], self.n_down_array_der[i][j], self.n_flip_array_der[i][j], self.plus_x_noflip_u_der[i][j], self.plus_x_noflip_d_der[i][j], self.minus_x_noflip_u_der[i][j], self.minus_x_noflip_d_der[i][j], \
                    self.plus_y_noflip_u_der[i][j], self.plus_y_noflip_d_der[i][j], self.minus_y_noflip_u_der[i][j], self.minus_y_noflip_d_der[i][j], self.plus_x_flip_ud_der[i][j], self.plus_x_flip_du_der[i][j], \
                    self.minus_x_flip_ud_der[i][j], self.minus_x_flip_du_der[i][j], self.plus_y_flip_ud_der[i][j], self.plus_y_flip_du_der[i][j], self.minus_y_flip_ud_der[i][j], self.minus_y_flip_du_der[i][j] \
                    = evolution_function(self.t, self.alpha, self.U, self.n_up_array_temp[i][j], self.n_down_array_temp[i][j], 0, self.n_up_array_temp[i-self.N][j], 0, self.n_up_array_temp[i+1][j], \
                        0, self.n_down_array_temp[i-self.N][j], 0, self.n_down_array_temp[i+1][j], 0, self.n_flip_array_temp[i-self.N][j], \
                        0, self.n_flip_array_temp[i+1][j], self.n_flip_array_temp[i][j], self.m_x_array_temp[i][j], self.m_y_array_temp[i][j], self.m_z_array_temp[i][j], 0, \
                        self.m_x_array_temp[i-self.N][j], 0, self.m_x_array_temp[i+1][j], 0, self.m_y_array_temp[i-self.N][j], 0, self.m_y_array_temp[i+1][j], \
                        0, self.m_z_array_temp[i-self.N][j], 0, self.m_z_array_temp[i+1][j] ,0, 0, self.minus_x_noflip_u_temp[i][j], self.minus_x_noflip_d_temp[i][j], \
                        0, 0, self.minus_y_noflip_u_temp[i][j], self.minus_y_noflip_d_temp[i][j], 0, 0, \
                        self.minus_x_flip_ud_temp[i][j], self.minus_x_flip_du_temp[i][j], 0,0, self.minus_y_flip_ud_temp[i][j], self.minus_y_flip_du_temp[i][j])
                elif i < self.N:
                    self.n_up_array_der[i][j], self.n_down_array_der[i][j], self.n_flip_array_der[i][j], self.plus_x_noflip_u_der[i][j], self.plus_x_noflip_d_der[i][j], self.minus_x_noflip_u_der[i][j], self.minus_x_noflip_d_der[i][j], \
                    self.plus_y_noflip_u_der[i][j], self.plus_y_noflip_d_der[i][j], self.minus_y_noflip_u_der[i][j], self.minus_y_noflip_d_der[i][j], self.plus_x_flip_ud_der[i][j], self.plus_x_flip_du_der[i][j], \
                    self.minus_x_flip_ud_der[i][j], self.minus_x_flip_du_der[i][j], self.plus_y_flip_ud_der[i][j], self.plus_y_flip_du_der[i][j], self.minus_y_flip_ud_der[i][j], self.minus_y_flip_du_der[i][j] \
                    = evolution_function(self.t, self.alpha, self.U, self.n_up_array_temp[i][j], self.n_down_array_temp[i][j], self.n_up_array_temp[i+self.N][j], 0, self.n_up_array_temp[i-1][j], self.n_up_array_temp[i+1][j], \
                        self.n_down_array_temp[i+self.N][j], 0, self.n_down_array_temp[i-1][j], self.n_down_array_temp[i+1][j], self.n_flip_array_temp[i+self.N][j], 0, \
                        self.n_flip_array_temp[i-1][j], self.n_flip_array_temp[i+1][j], self.n_flip_array_temp[i][j], self.m_x_array_temp[i][j], self.m_y_array_temp[i][j], self.m_z_array_temp[i][j], self.m_x_array_temp[i+self.N][j], \
                        0, self.m_x_array_temp[i-1][j], self.m_x_array_temp[i+1][j], self.m_y_array_temp[i+self.N][j], 0, self.m_y_array_temp[i-1][j], self.m_y_array_temp[i+1][j], \
                        self.m_z_array_temp[i+self.N][j], 0, self.m_z_array_temp[i-1][j], self.m_z_array_temp[i+1][j] ,self.plus_x_noflip_u_temp[i][j], self.plus_x_noflip_d_temp[i][j], 0, 0, \
                        self.plus_y_noflip_u_temp[i][j], self.plus_y_noflip_d_temp[i][j], self.minus_y_noflip_u_temp[i][j], self.minus_y_noflip_d_temp[i][j], self.plus_x_flip_ud_temp[i][j], self.plus_x_flip_du_temp[i][j], \
                        0, 0, self.plus_y_flip_ud_temp[i][j], self.plus_y_flip_du_temp[i][j], self.minus_y_flip_ud_temp[i][j], self.minus_y_flip_du_temp[i][j])
                elif i > (self.N**2 - self.N):
                    self.n_up_array_der[i][j], self.n_down_array_der[i][j], self.n_flip_array_der[i][j], self.plus_x_noflip_u_der[i][j], self.plus_x_noflip_d_der[i][j], self.minus_x_noflip_u_der[i][j], self.minus_x_noflip_d_der[i][j], \
                    self.plus_y_noflip_u_der[i][j], self.plus_y_noflip_d_der[i][j], self.minus_y_noflip_u_der[i][j], self.minus_y_noflip_d_der[i][j], self.plus_x_flip_ud_der[i][j], self.plus_x_flip_du_der[i][j], \
                    self.minus_x_flip_ud_der[i][j], self.minus_x_flip_du_der[i][j], self.plus_y_flip_ud_der[i][j], self.plus_y_flip_du_der[i][j], self.minus_y_flip_ud_der[i][j], self.minus_y_flip_du_der[i][j] \
                    = evolution_function(self.t, self.alpha, self.U, self.n_up_array_temp[i][j], self.n_down_array_temp[i][j], 0, self.n_up_array_temp[i-self.N][j], self.n_up_array_temp[i-1][j], self.n_up_array_temp[i+1][j], \
                        0, self.n_down_array_temp[i-self.N][j], self.n_down_array_temp[i-1][j], self.n_down_array_temp[i+1][j], 0, self.n_flip_array_temp[i-self.N][j], \
                        self.n_flip_array_temp[i-1][j],self.n_flip_array_temp[i+1][j], self.n_flip_array_temp[i][j], self.m_x_array_temp[i][j], self.m_y_array_temp[i][j], self.m_z_array_temp[i][j], 0, \
                        self.m_x_array_temp[i-self.N][j], self.m_x_array_temp[i-1][j], self.m_x_array_temp[i+1][j], 0, self.m_y_array_temp[i-self.N][j], self.m_y_array_temp[i-1][j], self.m_y_array_temp[i+1][j], \
                        0, self.m_z_array_temp[i-self.N][j], self.m_z_array_temp[i-1][j], self.m_z_array_temp[i+1][j] ,0, 0, self.minus_x_noflip_u_temp[i][j], self.minus_x_noflip_d_temp[i][j], \
                        self.plus_y_noflip_u_temp[i][j], self.plus_y_noflip_d_temp[i][j], self.minus_y_noflip_u_temp[i][j], self.minus_y_noflip_d_temp[i][j], 0, 0, \
                        self.minus_x_flip_ud_temp[i][j], self.minus_x_flip_du_temp[i][j], self.plus_y_flip_ud_temp[i][j], self.plus_y_flip_du_temp[i][j], self.minus_y_flip_ud_temp[i][j], self.minus_y_flip_du_temp[i][j])
                elif i % self.N == 0:
                    self.n_up_array_der[i][j], self.n_down_array_der[i][j], self.n_flip_array_der[i][j], self.plus_x_noflip_u_der[i][j], self.plus_x_noflip_d_der[i][j], self.minus_x_noflip_u_der[i][j], self.minus_x_noflip_d_der[i][j], \
                    self.plus_y_noflip_u_der[i][j], self.plus_y_noflip_d_der[i][j], self.minus_y_noflip_u_der[i][j], self.minus_y_noflip_d_der[i][j], self.plus_x_flip_ud_der[i][j], self.plus_x_flip_du_der[i][j], \
                    self.minus_x_flip_ud_der[i][j], self.minus_x_flip_du_der[i][j], self.plus_y_flip_ud_der[i][j], self.plus_y_flip_du_der[i][j], self.minus_y_flip_ud_der[i][j], self.minus_y_flip_du_der[i][j] \
                    = evolution_function(self.t, self.alpha, self.U, self.n_up_array_temp[i][j], self.n_down_array_temp[i][j], self.n_up_array_temp[i+self.N][j], self.n_up_array_temp[i-self.N][j], 0, self.n_up_array_temp[i+1][j], \
                        self.n_down_array_temp[i+self.N][j], self.n_down_array_temp[i-self.N][j], 0, self.n_down_array_temp[i+1][j], self.n_flip_array_temp[i+self.N][j], self.n_flip_array_temp[i-self.N][j], \
                        0, self.n_flip_array_temp[i+1][j], self.n_flip_array_temp[i][j], self.m_x_array_temp[i][j], self.m_y_array_temp[i][j], self.m_z_array_temp[i][j], self.m_x_array_temp[i+self.N][j], \
                        self.m_x_array_temp[i-self.N][j], 0, self.m_x_array_temp[i+1][j], self.m_y_array_temp[i+self.N][j], self.m_y_array_temp[i-self.N][j], 0, self.m_y_array_temp[i+1][j], \
                        self.m_z_array_temp[i+self.N][j], self.m_z_array_temp[i-self.N][j], 0, self.m_z_array_temp[i+1][j] ,self.plus_x_noflip_u_temp[i][j], self.plus_x_noflip_d_temp[i][j], self.minus_x_noflip_u_temp[i][j], self.minus_x_noflip_d_temp[i][j], \
                        0, 0, self.minus_y_noflip_u_temp[i][j], self.minus_y_noflip_d_temp[i][j], self.plus_x_flip_ud_temp[i][j], self.plus_x_flip_du_temp[i][j], \
                        self.minus_x_flip_ud_temp[i][j], self.minus_x_flip_du_temp[i][j], 0, 0, self.minus_y_flip_ud_temp[i][j], self.minus_y_flip_du_temp[i][j])
                elif (i+1) % self.N == 0:
                    self.n_up_array_der[i][j], self.n_down_array_der[i][j], self.n_flip_array_der[i][j], self.plus_x_noflip_u_der[i][j], self.plus_x_noflip_d_der[i][j], self.minus_x_noflip_u_der[i][j], self.minus_x_noflip_d_der[i][j], \
                    self.plus_y_noflip_u_der[i][j], self.plus_y_noflip_d_der[i][j], self.minus_y_noflip_u_der[i][j], self.minus_y_noflip_d_der[i][j], self.plus_x_flip_ud_der[i][j], self.plus_x_flip_du_der[i][j], \
                    self.minus_x_flip_ud_der[i][j], self.minus_x_flip_du_der[i][j], self.plus_y_flip_ud_der[i][j], self.plus_y_flip_du_der[i][j], self.minus_y_flip_ud_der[i][j], self.minus_y_flip_du_der[i][j] \
                    = evolution_function(self.t, self.alpha, self.U, self.n_up_array_temp[i][j], self.n_down_array_temp[i][j], self.n_up_array_temp[i+self.N][j], self.n_up_array_temp[i-self.N][j], self.n_up_array_temp[i-1][j], 0, \
                        self.n_down_array_temp[i+self.N][j], self.n_down_array_temp[i-self.N][j], self.n_down_array_temp[i-1][j], 0, self.n_flip_array_temp[i+self.N][j], self.n_flip_array_temp[i-self.N][j], \
                        self.n_flip_array_temp[i-1][j], 0, self.n_flip_array_temp[i][j], self.m_x_array_temp[i][j], self.m_y_array_temp[i][j], self.m_z_array_temp[i][j], self.m_x_array_temp[i+self.N][j], \
                        self.m_x_array_temp[i-self.N][j], self.m_x_array_temp[i-1][j], 0, self.m_y_array_temp[i+self.N][j], self.m_y_array_temp[i-self.N][j], self.m_y_array_temp[i-1][j], 0, \
                        self.m_z_array_temp[i+self.N][j], self.m_z_array_temp[i-self.N][j], self.m_z_array_temp[i-1][j], 0 ,self.plus_x_noflip_u_temp[i][j], self.plus_x_noflip_d_temp[i][j], self.minus_x_noflip_u_temp[i][j], self.minus_x_noflip_d_temp[i][j], \
                        self.plus_y_noflip_u_temp[i][j], self.plus_y_noflip_d_temp[i][j],0,0, self.plus_x_flip_ud_temp[i][j], self.plus_x_flip_du_temp[i][j], \
                        self.minus_x_flip_ud_temp[i][j], self.minus_x_flip_du_temp[i][j], self.plus_y_flip_ud_temp[i][j], self.plus_y_flip_du_temp[i][j], 0, 0)
                else:
                    self.n_up_array_der[i][j], self.n_down_array_der[i][j], self.n_flip_array_der[i][j], self.plus_x_noflip_u_der[i][j], self.plus_x_noflip_d_der[i][j], self.minus_x_noflip_u_der[i][j], self.minus_x_noflip_d_der[i][j], \
                    self.plus_y_noflip_u_der[i][j], self.plus_y_noflip_d_der[i][j], self.minus_y_noflip_u_der[i][j], self.minus_y_noflip_d_der[i][j], self.plus_x_flip_ud_der[i][j], self.plus_x_flip_du_der[i][j], \
                    self.minus_x_flip_ud_der[i][j], self.minus_x_flip_du_der[i][j], self.plus_y_flip_ud_der[i][j], self.plus_y_flip_du_der[i][j], self.minus_y_flip_ud_der[i][j], self.minus_y_flip_du_der[i][j] \
                    = evolution_function(self.t, self.alpha, self.U, self.n_up_array_temp[i][j], self.n_down_array_temp[i][j], self.n_up_array_temp[i+self.N][j], self.n_up_array_temp[i-self.N][j], self.n_up_array_temp[i-1][j], self.n_up_array_temp[i+1][j], \
                        self.n_down_array_temp[i+self.N][j], self.n_down_array_temp[i-self.N][j], self.n_down_array_temp[i-1][j], self.n_down_array_temp[i+1][j], self.n_flip_array_temp[i+self.N][j], self.n_flip_array_temp[i-self.N][j], \
                        self.n_flip_array_temp[i-1][j], self.n_flip_array_temp[i+1][j], self.n_flip_array_temp[i][j], self.m_x_array_temp[i][j], self.m_y_array_temp[i][j], self.m_z_array_temp[i][j], self.m_x_array_temp[i+self.N][j], \
                        self.m_x_array_temp[i-self.N][j], self.m_x_array_temp[i-1][j], self.m_x_array_temp[i+1][j], self.m_y_array_temp[i+self.N][j], self.m_y_array_temp[i-self.N][j], self.m_y_array_temp[i-1][j], self.m_y_array_temp[i+1][j], \
                        self.m_z_array_temp[i+self.N][j], self.m_z_array_temp[i-self.N][j], self.m_z_array_temp[i-1][j], self.m_z_array_temp[i+1][j] ,self.plus_x_noflip_u_temp[i][j], self.plus_x_noflip_d_temp[i][j], self.minus_x_noflip_u_temp[i][j], self.minus_x_noflip_d_temp[i][j], \
                        self.plus_y_noflip_u_temp[i][j], self.plus_y_noflip_d_temp[i][j], self.minus_y_noflip_u_temp[i][j], self.minus_y_noflip_d_temp[i][j], self.plus_x_flip_ud_temp[i][j], self.plus_x_flip_du_temp[i][j], \
                        self.minus_x_flip_ud_temp[i][j], self.minus_x_flip_du_temp[i][j], self.plus_y_flip_ud_temp[i][j], self.plus_y_flip_du_temp[i][j], self.minus_y_flip_ud_temp[i][j], self.minus_y_flip_du_temp[i][j])

                if i < self.N:
                    self.minus_x_noflip_u_der[i][j]   = 0
                    self.minus_x_noflip_d_der[i][j]   = 0
                    self.minus_x_flip_ud_der[i][j]    = 0
                    self.minus_x_flip_du_der[i][j]    = 0
                if i > (self.N**2 - self.N):
                    self.plus_x_noflip_u_der[i][j]    = 0
                    self.plus_x_noflip_d_der[i][j]    = 0
                    self.plus_x_flip_ud_der[i][j]     = 0
                    self.plus_x_flip_du_der[i][j]     = 0
                if (i % self.N) == 0:
                    self.plus_y_noflip_u_der[i][j]    = 0
                    self.plus_y_noflip_d_der[i][j]    = 0
                    self.plus_y_flip_ud_der[i][j]     = 0
                    self.plus_y_flip_du_der[i][j]     = 0
                if (i + 1) % self.N == 0:
                    self.minus_y_noflip_u_der[i][j]   = 0
                    self.minus_y_noflip_d_der[i][j]   = 0
                    self.minus_y_flip_ud_der[i][j]    = 0
                    self.minus_y_flip_du_der[i][j]    = 0


                if j == 0 or j == 1:
                    self.n_up_array_temp[i][j+1]         = self.n_up_array_temp[i][0]   + self.n_up_array_der[i][j]           * time_step / 2
                    self.n_down_array_temp[i][j+1]       = self.n_down_array_temp[i][0] + self.n_down_array_der[i][j]         * time_step / 2
                    self.n_flip_array_temp[i][j+1]       = self.n_flip_array_temp[i][0] + self.n_flip_array_der[i][j]         * time_step / 2
                    self.m_x_array_temp[i][j+1]          = 1/2 * (self.n_flip_array_temp[i][j+1] + np.conj(self.n_flip_array_temp[i][j+1]))
                    self.m_y_array_temp[i][j+1]          = -1j*1/2 * (self.n_flip_array_temp[i][j+1] - np.conj(self.n_flip_array_temp[i][j+1]))
                    self.m_z_array_temp[i][j+1]          = 1/2 * (self.n_up_array_temp[i][j+1] - self.n_down_array_temp[i][j+1])

                    self.plus_x_noflip_u_temp[i][j+1]    = self.plus_x_noflip_u_temp[i][0]    + self.plus_x_noflip_u_der[i][j]      * time_step / 2
                    self.plus_x_noflip_d_temp[i][j+1]    = self.plus_x_noflip_d_temp[i][0]    + self.plus_x_noflip_d_der[i][j]      * time_step / 2
                    self.minus_x_noflip_u_temp[i][j+1]   = self.minus_x_noflip_u_temp[i][0]   + self.minus_x_noflip_u_der[i][j]     * time_step / 2
                    self.minus_x_noflip_d_temp[i][j+1]   = self.minus_x_noflip_d_temp[i][0]   + self.minus_x_noflip_d_der[i][j]     * time_step / 2
                    self.plus_y_noflip_u_temp[i][j+1]    = self.plus_y_noflip_u_temp[i][0]    + self.plus_y_noflip_u_der[i][j]      * time_step / 2
                    self.plus_y_noflip_d_temp[i][j+1]    = self.plus_y_noflip_d_temp[i][0]    + self.plus_y_noflip_d_der[i][j]      * time_step / 2
                    self.minus_y_noflip_u_temp[i][j+1]   = self.minus_y_noflip_u_temp[i][0]   + self.minus_y_noflip_u_der[i][j]     * time_step / 2
                    self.minus_y_noflip_d_temp[i][j+1]   = self.minus_y_noflip_d_temp[i][0]   + self.minus_y_noflip_d_der[i][j]     * time_step / 2

                    self.plus_x_flip_ud_temp[i][j+1]     = self.plus_x_flip_ud_temp[i][0]     + self.plus_x_flip_ud_der[i][j]       * time_step / 2
                    self.plus_x_flip_du_temp[i][j+1]     = self.plus_x_flip_du_temp[i][0]     + self.plus_x_flip_du_der[i][j]       * time_step / 2
                    self.minus_x_flip_ud_temp[i][j+1]    = self.minus_x_flip_ud_temp[i][0]    + self.minus_x_flip_ud_der[i][j]      * time_step / 2
                    self.minus_x_flip_du_temp[i][j+1]    = self.minus_x_flip_du_temp[i][0]    + self.minus_x_flip_du_der[i][j]      * time_step / 2
                    self.plus_y_flip_ud_temp[i][j+1]     = self.plus_y_flip_ud_temp[i][0]     + self.plus_y_flip_ud_der[i][j]       * time_step / 2
                    self.plus_y_flip_du_temp[i][j+1]     = self.plus_y_flip_du_temp[i][0]     + self.plus_y_flip_du_der[i][j]       * time_step / 2
                    self.minus_y_flip_ud_temp[i][j+1]    = self.minus_y_flip_ud_temp[i][0]    + self.minus_y_flip_ud_der[i][j]      * time_step / 2
                    self.minus_y_flip_du_temp[i][j+1]    = self.minus_y_flip_du_temp[i][0]    + self.minus_y_flip_du_der[i][j]      * time_step / 2

                if j == 2:
                    self.n_up_array_temp[i][j+1]         = self.n_up_array_temp[i][0]   + self.n_up_array_der[i][j]           * time_step
                    self.n_down_array_temp[i][j+1]       = self.n_down_array_temp[i][0] + self.n_down_array_der[i][j]         * time_step
                    self.n_flip_array_temp[i][j+1]       = self.n_flip_array_temp[i][0] + self.n_flip_array_der[i][j]         * time_step
                    self.m_x_array_temp[i][j+1]          = 1/2 * (self.n_flip_array_temp[i][j+1] + np.conj(self.n_flip_array_temp[i][j+1]))
                    self.m_y_array_temp[i][j+1]          = -1j*1/2 * (self.n_flip_array_temp[i][j+1] - np.conj(self.n_flip_array_temp[i][j+1]))
                    self.m_z_array_temp[i][j+1]          = 1/2 * (self.n_up_array_temp[i][j+1] - self.n_down_array_temp[i][j+1])

                    self.plus_x_noflip_u_temp[i][j+1]    = self.plus_x_noflip_u_temp[i][0]    + self.plus_x_noflip_u_der[i][j]      * time_step
                    self.plus_x_noflip_d_temp[i][j+1]    = self.plus_x_noflip_d_temp[i][0]    + self.plus_x_noflip_d_der[i][j]      * time_step
                    self.minus_x_noflip_u_temp[i][j+1]   = self.minus_x_noflip_u_temp[i][0]   + self.minus_x_noflip_u_der[i][j]     * time_step
                    self.minus_x_noflip_d_temp[i][j+1]   = self.minus_x_noflip_d_temp[i][0]   + self.minus_x_noflip_d_der[i][j]     * time_step
                    self.plus_y_noflip_u_temp[i][j+1]    = self.plus_y_noflip_u_temp[i][0]    + self.plus_y_noflip_u_der[i][j]      * time_step
                    self.plus_y_noflip_d_temp[i][j+1]    = self.plus_y_noflip_d_temp[i][0]    + self.plus_y_noflip_d_der[i][j]      * time_step
                    self.minus_y_noflip_u_temp[i][j+1]   = self.minus_y_noflip_u_temp[i][0]   + self.minus_y_noflip_u_der[i][j]     * time_step
                    self.minus_y_noflip_d_temp[i][j+1]   = self.minus_y_noflip_d_temp[i][0]   + self.minus_y_noflip_d_der[i][j]     * time_step

                    self.plus_x_flip_ud_temp[i][j+1]     = self.plus_x_flip_ud_temp[i][0]     + self.plus_x_flip_ud_der[i][j]       * time_step
                    self.plus_x_flip_du_temp[i][j+1]     = self.plus_x_flip_du_temp[i][0]     + self.plus_x_flip_du_der[i][j]       * time_step
                    self.minus_x_flip_ud_temp[i][j+1]    = self.minus_x_flip_ud_temp[i][0]    + self.minus_x_flip_ud_der[i][j]      * time_step
                    self.minus_x_flip_du_temp[i][j+1]    = self.minus_x_flip_du_temp[i][0]    + self.minus_x_flip_du_der[i][j]      * time_step
                    self.plus_y_flip_ud_temp[i][j+1]     = self.plus_y_flip_ud_temp[i][0]     + self.plus_y_flip_ud_der[i][j]       * time_step
                    self.plus_y_flip_du_temp[i][j+1]     = self.plus_y_flip_du_temp[i][0]     + self.plus_y_flip_du_der[i][j]       * time_step
                    self.minus_y_flip_ud_temp[i][j+1]    = self.minus_y_flip_ud_temp[i][0]    + self.minus_y_flip_ud_der[i][j]      * time_step
                    self.minus_y_flip_du_temp[i][j+1]    = self.minus_y_flip_du_temp[i][0]    + self.minus_y_flip_du_der[i][j]      * time_step





        for i in range(self.N**2):
            self.n_up_array_temp[i][0]         = self.n_up_array_temp[i][0]     + (self.n_up_array_der[i][0]        + 2* self.n_up_array_der[i][1]          + 2* self.n_up_array_der[i][2]      + self.n_up_array_der[i][3]   ) * time_step / 6
            self.n_down_array_temp[i][0]       = self.n_down_array_temp[i][0]    + (self.n_down_array_der[i][0]      + 2* self.n_down_array_der[i][1]        + 2* self.n_down_array_der[i][2]    + self.n_down_array_der[i][3] ) * time_step / 6
            self.n_flip_array_temp[i][0]       = self.n_flip_array_temp[i][0]    + (self.n_flip_array_der[i][0]      + 2* self.n_flip_array_der[i][1]        + 2* self.n_flip_array_der[i][2]    + self.n_flip_array_der[i][3] ) * time_step / 6
            self.m_x_array_temp[i][0]          = 1/2 * (self.n_flip_array_temp[i][0] + np.conj(self.n_flip_array_temp[i][0]))
            self.m_y_array_temp[i][0]          = -1j * 1/2 * (self.n_flip_array_temp[i][0] - np.conj(self.n_flip_array_temp[i][0]))
            self.m_z_array_temp[i][0]          = 1/2 * (self.n_up_array_temp[i][0] - self.n_down_array_temp[i][0])

            self.plus_x_noflip_u_temp[i][0]    = self.plus_x_noflip_u_temp[i][0]  + ( self.plus_x_noflip_u_der[i][0]   + 2* self.plus_x_noflip_u_der[i][1]  + 2 * self.plus_x_noflip_u_der[i][2]  + self.plus_x_noflip_u_der[i][3] ) * time_step / 6
            self.plus_x_noflip_d_temp[i][0]    = self.plus_x_noflip_d_temp[i][0]  + ( self.plus_x_noflip_d_der[i][0]   + 2* self.plus_x_noflip_d_der[i][1]  + 2 * self.plus_x_noflip_d_der[i][2]  + self.plus_x_noflip_d_der[i][3] ) * time_step / 6
            self.minus_x_noflip_u_temp[i][0]   = self.minus_x_noflip_u_temp[i][0] + ( self.minus_x_noflip_u_der[i][0]  + 2* self.minus_x_noflip_u_der[i][1] + 2 * self.minus_x_noflip_u_der[i][2] + self.minus_x_noflip_u_der[i][3]) * time_step / 6
            self.minus_x_noflip_d_temp[i][0]   = self.minus_x_noflip_d_temp[i][0] + ( self.minus_x_noflip_d_der[i][0]  + 2* self.minus_x_noflip_d_der[i][1] + 2 * self.minus_x_noflip_d_der[i][2] + self.minus_x_noflip_d_der[i][3]) * time_step / 6
            self.plus_y_noflip_u_temp[i][0]    = self.plus_y_noflip_u_temp[i][0]  + ( self.plus_y_noflip_u_der[i][0]   + 2* self.plus_y_noflip_u_der[i][1]  + 2 * self.plus_y_noflip_u_der[i][2]  + self.plus_y_noflip_u_der[i][3] ) * time_step / 6
            self.plus_y_noflip_d_temp[i][0]    = self.plus_y_noflip_d_temp[i][0]  + ( self.plus_y_noflip_d_der[i][0]   + 2* self.plus_y_noflip_d_der[i][1]  + 2 * self.plus_y_noflip_d_der[i][2]  + self.plus_y_noflip_d_der[i][3] ) * time_step / 6
            self.minus_y_noflip_u_temp[i][0]   = self.minus_y_noflip_u_temp[i][0] + ( self.minus_y_noflip_u_der[i][0]  + 2* self.minus_y_noflip_u_der[i][1] + 2 * self.minus_y_noflip_u_der[i][2] + self.minus_y_noflip_u_der[i][3]) * time_step / 6
            self.minus_y_noflip_d_temp[i][0]   = self.minus_y_noflip_d_temp[i][0] + ( self.minus_y_noflip_d_der[i][0]  + 2* self.minus_y_noflip_d_der[i][1] + 2 * self.minus_y_noflip_d_der[i][2] + self.minus_y_noflip_d_der[i][3]) * time_step / 6

            self.plus_x_flip_ud_temp[i][0]     = self.plus_x_flip_ud_temp[i][0]     + ( self.plus_x_flip_ud_der[i][0]  + 2* self.plus_x_flip_ud_der[i][1]   + 2* self.plus_x_flip_ud_der[i][2]   + self.plus_x_flip_ud_der[i][3]  )  * time_step / 6
            self.plus_x_flip_du_temp[i][0]     = self.plus_x_flip_du_temp[i][0]     + ( self.plus_x_flip_du_der[i][0]  + 2* self.plus_x_flip_du_der[i][1]   + 2* self.plus_x_flip_du_der[i][2]   + self.plus_x_flip_du_der[i][3]  )  * time_step / 6
            self.minus_x_flip_ud_temp[i][0]    = self.minus_x_flip_ud_temp[i][0]    + ( self.minus_x_flip_ud_der[i][0] + 2* self.minus_x_flip_ud_der[i][1]  + 2* self.minus_x_flip_ud_der[i][2]  + self.minus_x_flip_ud_der[i][3] )  * time_step / 6
            self.minus_x_flip_du_temp[i][0]    = self.minus_x_flip_du_temp[i][0]    + ( self.minus_x_flip_du_der[i][0] + 2* self.minus_x_flip_du_der[i][1]  + 2* self.minus_x_flip_du_der[i][2]  + self.minus_x_flip_du_der[i][3] )  * time_step / 6
            self.plus_y_flip_ud_temp[i][0]     = self.plus_y_flip_ud_temp[i][0]     + ( self.plus_y_flip_ud_der[i][0]  + 2* self.plus_y_flip_ud_der[i][1]   + 2* self.plus_y_flip_ud_der[i][2]   + self.plus_y_flip_ud_der[i][3]  )  * time_step / 6
            self.plus_y_flip_du_temp[i][0]     = self.plus_y_flip_du_temp[i][0]     + ( self.plus_y_flip_du_der[i][0]  + 2* self.plus_y_flip_du_der[i][1]   + 2* self.plus_y_flip_du_der[i][2]   + self.plus_y_flip_du_der[i][3]  )  * time_step / 6
            self.minus_y_flip_ud_temp[i][0]    = self.minus_y_flip_ud_temp[i][0]    + ( self.minus_y_flip_ud_der[i][0] + 2* self.minus_y_flip_ud_der[i][1]  + 2* self.minus_y_flip_ud_der[i][2]  + self.minus_y_flip_ud_der[i][3] )  * time_step / 6
            self.minus_y_flip_du_temp[i][0]    = self.minus_y_flip_du_temp[i][0]    + ( self.minus_y_flip_du_der[i][0] + 2* self.minus_y_flip_du_der[i][1]  + 2* self.minus_y_flip_du_der[i][2]  + self.minus_y_flip_du_der[i][3] )  * time_step / 6





    def update_values(self) -> None:
        temp = 0
        temp_1 = 0
        temp_2 = 0
        temp_3 = 0
        temp_4 = 0
        temp_5 = 0
        for i in range(self.N**2):
            temp    += np.sqrt(self.m_x_array[i]**2 + self.m_y_array[i]**2 + self.m_z_array[i]**2)
            temp_1  += np.abs(self.m_x_array[i])
            temp_2  += np.abs(self.m_y_array[i])
            temp_3  += np.abs(self.m_z_array[i])
            temp_4  +=  self.n_up_array[i]
            temp_5 += + self.n_down_array[i]
        self.m_tot_array.append(temp * (1/(self.N**2)))
        self.m_x_tot_array.append(temp_1 * (1/(self.N**2)))
        self.m_y_tot_array.append(temp_2 * (1/(self.N**2)))
        self.m_z_tot_array.append(temp_3 * (1/(self.N**2)))
        self.n_up_avg.append(temp_4 * (1/(self.N**2)))
        self.n_down_avg.append(temp_5 * (1/(self.N**2)))
        self.n_avg.append((temp_4+temp_5)  * (1/(self.N**2)))


        self.m_x_array      = self.m_x_array_new.copy()
        self.m_y_array      = self.m_y_array_new.copy()
        self.m_z_array      = self.m_z_array_new.copy()
        self.n_up_array     = self.n_up_array_new.copy()
        self.n_down_array   = self.n_down_array_new.copy()
        self.n_flip_array   = self.n_flip_array_new.copy()

        self.plus_x_noflip_u    = self.plus_x_noflip_u_new.copy()
        self.plus_x_noflip_d    = self.plus_x_noflip_d_new.copy()
        self.minus_x_noflip_u   = self.minus_x_noflip_u_new.copy()
        self.minus_x_noflip_d   = self.minus_x_noflip_d_new.copy()
        self.plus_y_noflip_u    = self.plus_y_noflip_u_new.copy()
        self.plus_y_noflip_d    = self.plus_y_noflip_d_new.copy()
        self.minus_y_noflip_u   = self.minus_y_noflip_u_new.copy()
        self.minus_y_noflip_d   = self.minus_y_noflip_d_new.copy()
        self.plus_x_flip_ud     = self.plus_x_flip_ud_new.copy()
        self.plus_x_flip_du     = self.plus_x_flip_du_new.copy()
        self.minus_x_flip_ud    = self.minus_x_flip_ud_new.copy()
        self.minus_x_flip_du    = self.minus_x_flip_du_new.copy()
        self.plus_y_flip_ud     = self.plus_y_flip_ud_new.copy()
        self.plus_y_flip_du     = self.plus_y_flip_du_new.copy()
        self.minus_y_flip_ud    = self.minus_y_flip_ud_new.copy()
        self.minus_y_flip_du    = self.minus_y_flip_du_new.copy()

    def update_values_alt(self, i, time_step) -> None:
        temp = 0
        temp_1 = 0
        temp_2 = 0
        temp_3 = 0
        temp_4 = 0
        temp_5 = 0
        for i in range(self.N**2):
            temp    += np.sqrt(self.m_x_array_temp[i][0]**2 + self.m_y_array_temp[i][0]**2 + self.m_z_array_temp[i][0]**2)
            temp_1  += np.abs(self.m_x_array_temp[i][0])
            temp_2  += np.abs(self.m_y_array_temp[i][0])
            temp_3  += np.abs(self.m_z_array_temp[i][0])
            temp_4  +=  self.n_up_array_temp[i][0]
            temp_5 += self.n_down_array_temp[i][0]
        self.m_tot_array.append(temp * (1/(self.N**2)))
        self.m_x_tot_array.append(temp_1 * (1/(self.N**2)))
        self.m_y_tot_array.append(temp_2 * (1/(self.N**2)))
        self.m_z_tot_array.append(temp_3 * (1/(self.N**2)))
        self.n_up_avg.append(temp_4 * (1/(self.N**2)))
        self.n_down_avg.append(temp_5 * (1/(self.N**2)))
        self.n_avg.append((temp_4+temp_5)  * (1/(self.N**2)))



    def plot(self, size, title_string = ''):
        # norm = matplotlib.colors.Normalize(vmin=0.0,vmax=0.5,clip=False)
        fig, (ax_1,ax_2) = plt.subplots(1,2, figsize=(5.0,4.0), sharex=True, sharey=True)
        # M = np.hypot(np.array(np.reshape(self.m_x_array_new, (self.N, self.N)), dtype=np.double), np.array(np.reshape(self.m_y_array_new, (self.N, self.N)), dtype=np.double))
        X, Y    = np.meshgrid(np.linspace(1,self.N, self.N), np.linspace(1, self.N,self.N))
        cbar_ax = fig.add_axes([0.85, 0.1075, 0.025, 0.775])
        # ax_1.scatter(X,Y, color='black', s=1)
        im = ax_2.pcolormesh(np.linspace(0,self.N, self.N+1), np.linspace(0,self.N, self.N+1), np.reshape(self.m_z_array_temp[:,0], (self.N, self.N)), shading='auto', cmap='seismic', vmax=0.5, vmin=-0.5)
        cbar = fig.colorbar(im, cax=cbar_ax)
        # fig.suptitle(title_string)
        fig.supxlabel('site number x', fontsize='medium')
        # ax_2.set_xlabel('site number x')
        fig.supylabel('site number y', fontsize='medium')
        # ax_2.set_ylabel('site number y')
        if size == 1:
            ax_1.quiver(X,Y, np.array(np.reshape(self.m_x_array_temp[:,0], (self.N, self.N)), dtype=np.double), np.array(np.reshape(self.m_y_array_temp[:,0], (self.N, self.N)), dtype=np.double), cmap='Reds', pivot='middle', scale=7.5, width=0.0075)
            ax_1.set_xlim(0.5,10.5)
            ax_1.set_ylim(0.5,10.5)
        else:
            ax_1.quiver(X,Y, np.array(np.reshape(self.m_x_array_temp[:,0], (self.N, self.N)), dtype=np.double), np.array(np.reshape(self.m_y_array_temp[:,0], (self.N, self.N)), dtype=np.double), cmap='Reds', pivot='middle')
            ax_1.set_xlim(0.5,self.N+0.5)
            ax_1.set_ylim(0.5,self.N+0.5)
        
        cbar.set_label(r'$\langle m_{i,z} \rangle$', rotation=270)

        ax_1.yaxis.set_major_locator(MultipleLocator(1))
        ax_2.yaxis.set_major_locator(MultipleLocator(1))

        ax_1.xaxis.set_major_locator(MultipleLocator(1))
        ax_2.xaxis.set_major_locator(MultipleLocator(1))

        ax_1.tick_params(axis="both",direction="in", which='minor', length=2.0, right=True, top=True)
        ax_1.tick_params(axis="both",direction="in", which='major', length=2.0, right=True, top=True)

        # ax_2.yaxis.set_major_locator(MultipleLocator(2))
        # ax_2.yaxis.set_minor_locator(MultipleLocator(1))

        # ax_2.xaxis.set_major_locator(MultipleLocator(2))
        # ax_2.xaxis.set_minor_locator(MultipleLocator(1))

        ax_2.tick_params(axis="both",direction="in", which='minor', length=2.0, right=True, top=True)
        ax_2.tick_params(axis="both",direction="in", which='major', length=2.0, right=True, top=True)

        # ax_1.margins(0.5)
        # ax_2.margins(0.5)





        plt.show()

# fig, (ax_1,ax_2) = plt.subplots(1,2, figsize=(5.0,4.0))
#         norm = matplotlib.colors.Normalize(vmin=0.0,vmax=0.5,clip=False)
#         X, Y    = np.meshgrid(np.linspace(0,self.N, self.N), np.linspace(0, self.N,self.N))
#         ax_1.quiver(X,Y, np.reshape(self.m_x_array_new, (self.N, self.N)), np.reshape(self.m_y_array_new, (self.N, self.N)), pivot='middle', norm=norm, cmap='seismic')
#         im = ax_2.pcolormesh(X, Y, np.reshape(self.m_z_array_new, (self.N, self.N)), shading='auto', cmap='seismic', vmax=0.7, vmin=-0.7)
#         fig.colorbar(im, ax=ax_2)
#         ax_1.set_xlabel('site number x')
#         ax_2.set_xlabel('site number x')
#         ax_1.set_ylabel('site number y')
#         ax_2.set_ylabel('site number y')
#         ax_1.set_xlim(-1.0, self.N+1.0)
#         ax_1.set_xticks(np.arange(0, self.N+1,2))
#         ax_1.set_ylim(-1.0, self.N+1.0)
#         ax_1.set_yticks(np.arange(0, self.N+1,2))
#         ax_2.set_xlim(-1.0, self.N+1.0)
#         ax_2.set_xticks(np.arange(0, self.N+1,2))
#         ax_2.set_ylim(-1.0, self.N+1.0)
#         ax_2.set_yticks(np.arange(0, self.N+1,2))
#         plt.show()



    def plot_array(self, i, title_string = ''):
        time_array = np.linspace(0, 1, (i+1))
        fig, ax = plt.subplots()

        ax.plot(time_array, self.m_tot_array, color='red', linewidth=1.0, label=r'$\langle m_i \rangle$')
        ax.plot(time_array, self.m_x_tot_array, color='blue', linewidth=0.75, label=r'$\langle m_{i,x} \rangle$')
        ax.plot(time_array, self.m_y_tot_array, color='green', linewidth=0.75, label=r'$\langle m_{i,y} \rangle$')
        ax.plot(time_array, self.m_z_tot_array, color='limegreen', linewidth=0.75, label=r'$\langle m_{i,z} \rangle$')
        ax.plot(time_array, self.n_up_avg, color='teal', linewidth=1.0, label=r"$\langle n_{u,\uparrow} \rangle$")
        ax.plot(time_array, self.n_down_avg, linewidth=1.0, label=r"$\langle n_{u,\downarrow} \rangle$")
        ax.plot(time_array, self.n_avg, linewidth=1.0, label=r"$\langle n_i \rangle$")
        ax.legend()
        fig.suptitle(title_string)
        plt.show()

    def save(self, i, title_string = ''):
        norm = matplotlib.colors.Normalize(vmin=0.0,vmax=0.5,clip=False)
        fig, (ax_1,ax_2) = plt.subplots(1,2)
        M = np.hypot(np.array(np.reshape(self.m_x_array_new, (self.N, self.N)), dtype=np.double), np.array(np.reshape(self.m_y_array_new, (self.N, self.N)), dtype=np.double))
        X, Y    = np.meshgrid(np.linspace(0,self.N, self.N), np.linspace(0, self.N,self.N))
        ax_1.quiver(X,Y, np.reshape(self.m_x_array_new, (self.N, self.N)), np.reshape(self.m_y_array_new, (self.N, self.N)), M, cmap='Reds', pivot='middle', norm=norm)
        # ax_1.scatter(X,Y, color='black', s=1)
        ax_2.pcolormesh(np.linspace(0,self.N, self.N+1), np.linspace(0,self.N, self.N+1), np.reshape(self.m_z_array_new, (self.N, self.N)), shading='auto', cmap='seismic', vmax=0.5, vmin=-0.5)
        fig.suptitle(title_string)
        ax_1.set_xlabel('$y$')
        ax_1.set_ylabel('$x$')
        ax_2.set_xlabel('$z$-component')

        plt.savefig(f'data/dynamics_initial_values/plots/{i}.jpg', dpi=300)
        plt.close()

    def plot_n(self, spin):
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        cbar_ax = fig.add_axes([0.85, 0.1075, 0.025, 0.775])
        norm = matplotlib.colors.Normalize(vmin=0.0,vmax=0.5,clip=False)
        X, Y    = np.meshgrid(np.linspace(1,self.N, self.N), np.linspace(1, self.N,self.N))
        if spin == 1:
            im = ax.pcolormesh(X, Y, np.reshape(self.n_up_array, (self.N, self.N)), shading='auto', cmap='Blues', vmax=0.30, vmin=0.40)
        else: 
            im = ax.pcolormesh(X, Y, np.reshape(self.n_down_array, (self.N, self.N)), shading='auto', cmap='Blues', vmax=0.30, vmin=0.40)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'$\langle n_{i}\rangle$', rotation=270)
        ax.set_xlabel('site number x')
        ax.set_ylabel('site number y')
        # ax.set_xlim(0.5, self.N+0.5)
        # ax.set_xticks(np.arange(1, self.N,1))
        # ax.set_ylim(0.5, self.N+0.5)
        # ax.set_yticks(np.arange(1, self.N,1))

        ax.yaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_minor_locator(MultipleLocator(1))

        ax.xaxis.set_major_locator(MultipleLocator(1))
        # ax.xaxis.set_minor_locator(MultipleLocator(1))

        ax.tick_params(axis="both",direction="in", which='minor', length=2.0, right=True, top=True)
        ax.tick_params(axis="both",direction="in", which='major', length=2.0, right=True, top=True)
        plt.show()