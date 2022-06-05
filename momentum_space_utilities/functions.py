import numpy as np
pi = np.pi
from functions import matrixMaster, matrix_wrapper, matrix_wrapper_PM, matrix_wrapper_FM

################################################################################
## Function declarations 
################################################################################

## The only function to be defined in Python is the brillouinMaster-function responsible for defining the correct Brillouin zone size and shape, depending on 
## the magnetic modulation vector Q. Each Q must be paired with the correct BZ in order to preserve the correct number of independent degrees of freedom.

def brillouinMaster(Q, k_res):
    if Q[0] == Q[1]:            # diagonal  
        if Q[0] == 0: 
            N = 1 
            k_res_temp = k_res 
            k_1 =np.linspace([-pi, -pi], [pi, -pi], k_res_temp)
            k_2 = np.linspace([-pi, pi], [pi, pi], k_res_temp)
            k_grid = np.linspace(k_1, k_2, k_res_temp)

            return k_grid, k_res_temp, N
        N = int(2*pi/Q[0])
        k_res_temp = int(k_res*1/np.sqrt(N))
        if N == 2: 
            k_1 =np.linspace([-pi, 0], [0, pi], k_res_temp)
            k_2 = np.linspace([0, -pi], [pi, 0], k_res_temp)
            k_grid = np.linspace(k_1, k_2, k_res_temp)

            return k_grid, k_res_temp, N
        elif N == 4: 
            k_1 =np.linspace([-pi/2, pi/2], [0, pi], int(k_res_temp/np.sqrt(2)))
            k_2 = np.linspace([0, -pi], [pi/2, -pi/2], int(k_res_temp/np.sqrt(2)))
            k_grid = np.linspace(k_1, k_2, int(k_res_temp*np.sqrt(2)))

            return k_grid, k_res_temp, N

        elif N == 8: 
            k_1 =np.linspace([-pi/4, 3*pi/4], [0, pi], int(k_res_temp/2))
            k_2 = np.linspace([0, -pi], [pi/4, -3*pi/4], int(k_res_temp/2))
            k_grid = np.linspace(k_1, k_2, k_res_temp*2)

            return k_grid, k_res_temp, N
        elif N == 16: 
            k_1 =np.linspace([-pi/8, 7*pi/8], [0, pi], int(k_res_temp/2))
            k_2 = np.linspace([0, -pi], [pi/8, -7*pi/8], int(k_res_temp/2))
            k_grid = np.linspace(k_1, k_2, k_res_temp*2)

            return k_grid, k_res_temp, N
        
        elif N == 32: 
            k_1 =np.linspace([-pi/16, 15*pi/16], [0, pi], int(k_res_temp/2))
            k_2 = np.linspace([0, -pi], [pi/16, -15*pi/16], int(k_res_temp/2))
            k_grid = np.linspace(k_1, k_2, k_res_temp*2)

            return k_grid, k_res_temp, N

    elif Q[0] == 0:             # parallel_X
        N = int(2*pi/Q[1])
        k_res_temp = int(k_res*1/np.sqrt(N))
        k_1 =np.linspace([-pi, pi/N], [pi, pi/N], k_res_temp*2)
        k_2 = np.linspace([-pi, -pi/N], [pi, -pi/N], k_res_temp*2)
        k_grid = np.linspace(k_1, k_2, int(k_res_temp/2))

        return k_grid, k_res_temp, N
    elif Q[1]==0:               # parallel_Y 
        N = int(2*pi/Q[0])
        k_res_temp = int(k_res*1/np.sqrt(N))
        k_1 =np.linspace([-pi/N, pi], [pi/N, pi], int(k_res_temp/2))
        k_2 = np.linspace([-pi/N, -pi], [pi/N, -pi], int(k_res_temp/2))
        k_grid = np.linspace(k_1, k_2, k_res_temp*2)

        return k_grid, k_res_temp, N

    elif Q[1] == pi: 
        N=int(2*pi/Q[0]) 
        k_res_temp = int(k_res*1/np.sqrt(N))
        k_1 =np.linspace([-2*pi/N, 0], [0, pi], k_res_temp)
        k_2 = np.linspace([0, -pi], [2*pi/N, 0], k_res_temp)
        k_grid = np.linspace(k_1, k_2, k_res_temp)

        return k_grid, k_res_temp, N

    elif Q[0] == pi: 
        N=int(2*pi/Q[1]) 
        k_res_temp = int(k_res*1/np.sqrt(N))
        k_1 =np.linspace([-pi, 0], [0, 2*pi/N], k_res_temp)
        k_2 = np.linspace([0, -2*pi/N], [pi, 0], k_res_temp)
        k_grid = np.linspace(k_1, k_2, k_res_temp)

        return k_grid, k_res_temp, N
        
if __name__=='__main__':
    pass