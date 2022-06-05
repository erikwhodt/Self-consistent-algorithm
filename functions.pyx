#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport abs
from libc.math cimport exp
from libc.math cimport sqrt
from libc.math cimport pow

cdef extern from "<complex.h>" namespace "std" nogil:
    double complex conj(double complex)
    
cdef matrix_func_xy(Q,  N, k, psi, h, alpha, t, xi):
    matrix = np.full((N*2,N*2), 0, dtype=complex) 
    cdef double cos_psi = cos(psi)
    cdef double sin_psi = sin(psi)
    for i in range(N):
        rashb = -2*alpha*(sin(k[0]+i*Q[0])+1j*sin(k[1]+i*Q[1]))
        toit = -2*t*(cos(k[0]+i*Q[0])+cos(k[1]+i*Q[1]))-xi
        matrix[2*i:2*i+2,2*i:2*i+2] = np.array([[toit-h/2.0 * cos_psi, rashb],[conj(rashb), toit+h/2.0 * cos_psi]]) 

        matrix[2*i][(2*i+3) % (2*N)] += -h/2.0 * sin_psi
        matrix[(2*i+3)%(2*N)][2*i] += -h/2.0 * sin_psi
      
    return matrix 

cdef matrix_func_xz(Q, N, k, phi, h, alpha, t, xi):
    matrix = np.full((N*2,N*2), 0, dtype=complex) 
    cdef double cos_phi = cos(phi)
    cdef double sin_phi = sin(phi)
    for i in range(N):
        rashb = -2*alpha*(sin(k[0]+i*Q[0])+1j*sin(k[1]+i*Q[1]))
        toit = -2*t*(cos(k[0]+i*Q[0])+cos(k[1]+i*Q[1]))-xi
        matrix[2*i:2*i+2,2*i:2*i+2] = np.array([[toit, rashb+1j*h/2 * cos_phi],[conj(rashb)-1j*h/2 * cos_phi, toit]]) 
        
        if N == 1:
            matrix[0][0] -= h/2.0 * sin_phi
            matrix[1][1] += h/2.0 * sin_phi

        elif N == 2:
            matrix[2*i][(2*i+2) % (2*N)] += -h/2.0 * sin_phi
            matrix[(2*i+1)][(2*i+3) % (2*N)] += h/2.0 * sin_phi

        else:
            matrix[2*i][(2*i+3) % (2*N)] += -1j*h/4.0 *sin_phi
            matrix[2*i+1][(2*i+2) % (2*N)] += -1j*h/4.0 *sin_phi

            matrix[(2*i+3) % (2*N)][2*i] += 1j*h/4.0 *sin_phi
            matrix[(2*i+2) % (2*N)][2*i+1] += 1j*h/4.0 *sin_phi
            
            matrix[2*i][(2*i+2) % (2*N)] += -h/4.0 * sin_phi
            matrix[(2*i+2) % (2*N)][2*i] += -h/4.0 * sin_phi

            matrix[2*i+1][(2*i+3) % (2*N)] += h/4.0 * sin_phi
            matrix[(2*i+3) % (2*N)][2*i+1] += h/4.0 * sin_phi
    
    return matrix 

cdef matrix_func_yz(Q, N, k, phi, h, alpha, t, xi):
    matrix = np.full((N*2,N*2), 0, dtype=complex) 
    cdef double cos_phi = cos(phi)
    cdef sin_phi = sin(phi)
    for i in range(N):
        rashb = -2*alpha*(sin(k[0]+i*Q[0])+1j*sin(k[1]+i*Q[1]))
        toit = -2*t*(cos(k[0]+i*Q[0])+cos(k[1]+i*Q[1]))-xi
        matrix[2*i:2*i+2,2*i:2*i+2] = np.array([[toit, rashb - h/2 * cos_phi],[conj(rashb) - h/2 * cos_phi, toit]]) 
        
        if N == 1: 
            matrix[0][1] += -h/2.0 * np.exp(-1j*phi)
            matrix[1][0] += -h/2.0 * np.exp(1j*phi)

        elif N == 2:
            matrix[2*i][(2*i+3) % (2*N)] += 1j*h/2.0 * sin_phi
            matrix[(2*i+3)%(2*N)][2*i] += -1j*h/2.0 * sin_phi

        else: 
            matrix[2*i][(2*i+3) % (2*N)] += 1j*h/4.0 *sin_phi
            matrix[2*i+1][(2*i+2) % (2*N)] += -1j*h/4.0 *sin_phi

            matrix[(2*i+3) % (2*N)][2*i] += -1j*h/4.0 *sin_phi
            matrix[(2*i+2) % (2*N)][2*i+1] += 1j*h/4.0 *sin_phi

            matrix[2*i][(2*i+2) % (2*N)] += -1j*h/4.0 *sin_phi
            matrix[(2*i+2) % (2*N)][2*i] += 1j*h/4.0 *sin_phi

            matrix[2*i+1][(2*i+3) % (2*N)] += 1j*h/4.0 *sin_phi
            matrix[(2*i+3) % (2*N)][2*i+1] += -1j*h/4.0 *sin_phi
       
    return matrix

cpdef matrixMaster(Q, N, k_array, t, xi, alpha, psi, h, mu, polarization):
    if polarization == 'x': 
        return matrix_func_xy(Q, N, k_array, psi, h, alpha, t, xi)
    elif polarization == 'y': 
        return matrix_func_yz(Q, N, k_array, psi, h, alpha, t, xi)
    elif polarization == 'z':
        return matrix_func_xz(Q, N, k_array, psi, h, alpha, t, xi)

cpdef matrix_wrapper(Q, N, k_grid, t, xi, alpha, psi, h, mu, polarization, beta, k_res_temp): 
    cdef double energy  = 0
    cdef double m_x     = 0
    cdef double m_y     = 0
    cdef double m_z     = 0
    cdef double m_tot   = 0
    cdef double n_temp  = 0
    for k_array in k_grid: 
        for k_coordinate in k_array: 

            matrix = matrixMaster(Q, N, k_coordinate, t, xi, alpha, psi, h, mu, polarization) 
            eigval, eigvec = np.linalg.eigh(matrix)  
            eigvec = np.conj(eigvec.T)           

            m_x = 0
            m_y = 0 
            m_z = 0

            for o in range(int(N/2.0)):
                
                for l in range(N*2):

                    eigval_fermi = 1/(exp(beta*(eigval[l]-mu))+1)
                    
                    if o == 0:
                        energy += (eigval[l]-mu)*eigval_fermi           

                    a = abs(eigvec[l][o*4])**2
                    b = abs(eigvec[l][o*4+1])**2
                    c = abs(eigvec[l][o*4+2])**2
                    d = abs(eigvec[l][o*4+3])**2

                    e = eigvec[l][o*4]*conj(eigvec[l][o*4+3])
                    f = conj(eigvec[l][o*4])*eigvec[l][o*4+3]
                    g = eigvec[l][o*4+2]*conj(eigvec[l][(o*4+5)%(2*N)])
                    h_2 = conj(eigvec[l][o*4+2])*eigvec[l][(o*4+5)%(2*N)]

                    i = eigvec[l][o*4]*conj(eigvec[l][o*4+2])
                    j = conj(eigvec[l][o*4])*eigvec[l][o*4+2]
                    k = eigvec[l][o*4+1]*conj(eigvec[l][(o*4+3)])
                    m = conj(eigvec[l][o*4+1])*eigvec[l][(o*4+3)]


                    n_temp += (a + b + c + d)*eigval_fermi
                    m_x += (e + f + g + h_2)*eigval_fermi
                    
                    if N > 2:
                        m_y += 2*(-1j * (e  -f + g - h_2))*eigval_fermi
                        m_z += 2*(a - b + c - d)*eigval_fermi	
                        
                    else: 
                        m_y += (-1j * (e  -f + g - h_2))*eigval_fermi
                        m_z += (i+j-k-m)*eigval_fermi


            m_tot += 1.0/(2.0*N*k_res_temp**2) * sqrt(pow(m_x,2) + pow(m_y,2)+pow(m_z,2))

    return m_tot, n_temp, energy

cpdef matrix_wrapper_FM(Q, N, k_grid, t, xi, alpha, psi, h, mu, polarization, beta, k_res_temp):
    cdef double energy  = 0.0
    cdef double m_x     = 0.0
    cdef double m_y     = 0.0   
    cdef double m_z     = 0.0 
    cdef double m_tot   = 0.0   
    cdef double n_temp  = 0.0 
    cdef double eigval_fermi = 0.0
    for k_array in k_grid: 
        for k_coordinate in k_array: 

            matrix = matrixMaster(Q, N, k_coordinate, t, xi, alpha, psi, h, mu, polarization) 
            eigval, eigvec = np.linalg.eigh(matrix)  

            m_x     = 0.0
            m_y     = 0.0   
            m_z     = 0.0 

            for l in range(N*2):

                eigval_fermi = 1/(exp(beta*(eigval[l]-mu))+1)
                energy += (eigval[l]-mu)*eigval_fermi                

                for o in range(N):

                    a = pow(abs(eigvec[o*2][l]), 2)
                    b = pow(abs(eigvec[o*2+1][l]),2)
                    c = conj(eigvec[o*2+1][l])
                    d = conj(eigvec[o*2][l])

                    n_temp += (a + b)*eigval_fermi
                    m_x += (1/(2*N * pow(k_res_temp, 2)))*((eigvec[o*2][l]*c + d*eigvec[o*2+1][l]))*eigval_fermi
                    m_y += (1/(2*N * pow(k_res_temp,2)))*(-1j * (d*eigvec[o*2+1][l]-eigvec[o*2][l]*c ))*eigval_fermi
                    m_z += (1/(2*N * pow(k_res_temp,2)))*(a - b)*eigval_fermi
            
            m_tot += sqrt(pow(m_x,2) + pow(m_y,2)+pow(m_z,2))
    return m_tot, n_temp, energy

cpdef matrix_wrapper_PM(Q, N, k_grid, t, xi, alpha, psi, h, mu, polarization, beta, k_res_temp):
    cdef double n_temp = 0.0
    cdef double energy  = 0.0
    cdef double eigval_fermi = 0.0
    for k_array in k_grid: 
        for k_coordinate in k_array:
            rashb = abs(-2*alpha*(sin(k_coordinate[0])+1j*sin(k_coordinate[1])))
            toit = -2*t*(cos(k_coordinate[0])+cos(k_coordinate[1]))-xi           

            e_plus = toit + rashb
            e_min = toit - rashb

            eigval_plus = 1.0/(exp(beta*(e_plus-mu))+1.0)
            eigval_min = 1.0/(exp(beta*(e_min-mu))+1.0)

            n_temp += (eigval_plus + eigval_min) 
            energy += ((e_plus-mu)*eigval_plus + (e_min-mu)*eigval_min)
    return n_temp, energy

cpdef matrix_wrapper_real(energy, PM_flag, N, beta, n_array_new, eig_vec, eig_val, m_x_array_new, m_y_array_new, m_z_array_new, m_array_new):
    if PM_flag: 
        for i in range(N**2):     
            n_temp      = 0
            
            for j in range(2*N**2): 
                eigval = 1/(exp(beta*(eig_val[j]))+1)
                u = eig_vec[j][i*2]
                v = eig_vec[j][i*2+1]
                
                if i == 0:
                    energy += (eig_val[j])*eigval
                n_temp +=(pow(abs(u),2)+pow(abs(v),2))*eigval

            n_array_new[i]     = n_temp
        return energy, n_array_new

    else:
        for i in range(N**2):     
            n_temp          = 0
            m_x_temp        = 0
            m_y_temp        = 0
            m_z_temp        = 0                        

            for j in range(2*N**2): 
                eigval = 1.0/(exp(beta*(eig_val[j]))+1.0)
                u = eig_vec[j][i*2]
                v = eig_vec[j][i*2+1]

                m_x_temp += (u*conj(v)+conj(u)*v)*eigval
                m_y_temp += -1j*(u*conj(v)-conj(u)*v)*eigval
                m_z_temp += (pow(abs(u),2)-pow(abs(v),2))*eigval
                n_temp   += (pow(abs(u),2)+pow(abs(v),2))*eigval
                
                if i == 0:
                    energy += (eig_val[j])*eigval
                   
                
        
            m_x_array_new[i]   = 0.5 * m_x_temp
            m_y_array_new[i]   = 0.5 * m_y_temp
            m_z_array_new[i]   = 0.5 * m_z_temp

            m_array_new[i]     = 0.5 * sqrt((pow(m_x_temp,2)+pow(m_y_temp,2)+pow(m_z_temp,2)))
            
            n_array_new[i]     = n_temp

        return energy, n_array_new, m_x_array_new, m_y_array_new, m_z_array_new, m_array_new

cpdef evolution_function(
    t, alpha, U, n_u, n_d, n_x_u, n_minx_u, n_y_u, n_miny_u, n_x_d, n_minx_d, n_y_d, n_miny_d, n_x_flip, n_minx_flip, n_y_flip, n_miny_flip, 
    n_flip, m_x, m_y,  m_z, m_x_x, m_x_minx, m_x_y, m_x_miny, m_y_x, m_y_minx, m_y_y, m_y_miny, m_z_x, m_z_minx, m_z_y, m_z_miny ,plus_x_noflip_up, plus_x_noflip_down, minus_x_noflip_up, minus_x_noflip_down, plus_y_noflip_up, plus_y_noflip_down, 
    minus_y_noflip_up, minus_y_noflip_down, plus_x_flip_ud, plus_x_flip_du, minus_x_flip_ud, minus_x_flip_du,  plus_y_flip_ud, plus_y_flip_du, 
    minus_y_flip_ud, minus_y_flip_du):

    n_u_der = 1/1j * (-t*(plus_x_noflip_up + minus_x_noflip_up + plus_y_noflip_up + minus_y_noflip_up \
        - conj(plus_x_noflip_up) - conj(minus_x_noflip_up) - conj(plus_y_noflip_up) - conj(minus_y_noflip_up)) \
        - 1j*alpha*(-1j)*plus_x_flip_ud + 1j*alpha*(-1j)*minus_x_flip_ud +1j*alpha*plus_y_flip_ud - 1j*alpha*minus_y_flip_ud \
        - 1j*alpha*(1j)*conj(plus_x_flip_ud) +1j*alpha*(1j)*conj(minus_x_flip_ud)+1j*alpha*conj(plus_y_flip_ud) \
        - 1j*alpha*conj(minus_y_flip_ud) \
        - U*(m_x * (n_flip-conj(n_flip))+ m_y * ((-1j)*n_flip - (1j) * conj(n_flip)) ))

    n_d_der = 1/1j * (-t*(plus_x_noflip_down + minus_x_noflip_down + plus_y_noflip_down + minus_y_noflip_down \
        - conj(plus_x_noflip_down) - conj(minus_x_noflip_down) - conj(plus_y_noflip_down) - conj(minus_y_noflip_down)) \
        - 1j*alpha*(1j)*plus_x_flip_du + 1j*alpha*(1j)*minus_x_flip_du +1j*alpha*plus_y_flip_du - 1j*alpha*minus_y_flip_du \
        - 1j*alpha*(-1j)*conj(plus_x_flip_du) +1j*alpha*(-1j)*conj(minus_x_flip_du)+1j*alpha*conj(plus_y_flip_du) \
        - 1j*alpha*conj(minus_y_flip_du) \
        - U*(m_x * (conj(n_flip) - n_flip)+ m_y * ((1j)*conj(n_flip) - (-1j) * n_flip)))

    n_flip_der = 1/1j * (-t*(plus_x_flip_ud + minus_x_flip_ud + plus_y_flip_ud + minus_y_flip_ud  
        - conj(plus_x_flip_du) - conj(minus_x_flip_du) - conj(plus_y_flip_du) - conj(minus_y_flip_du)) \
        - 1j*alpha*(1j)*plus_x_noflip_up +1j*alpha*(1j)*minus_x_noflip_up +1j*alpha*plus_y_noflip_up -1j*alpha*minus_y_noflip_up -1j*alpha*(1j)*conj(plus_x_noflip_down) \
        + 1j*alpha*(1j)*conj(minus_x_noflip_down)+1j*alpha*conj(plus_y_noflip_down)-1j*alpha*conj(minus_y_noflip_down) \
        + 2*U*m_z*n_flip \
        - U*(m_x + m_y * (1j)) * (n_u - n_d))

    plus_x_noflip_up_der = 1/1j * (-t*(n_u-n_x_u)  + 1j*alpha*(-1j)*n_flip - 1j*alpha*(1j)*conj(n_x_flip) \
        - U*(m_x_x + m_y_x*(-1j)) * plus_x_flip_ud + U*(m_x + m_y*(1j))*plus_x_flip_du \
        + U*( (n_x_u+n_x_d)/2.0 - (n_u + n_d)/2.0 - m_z_x + m_z )*plus_x_noflip_up) 
    
    plus_x_noflip_down_der = 1/1j * (-t*(n_d-n_x_d)  + 1j*alpha*(1j)*conj(n_flip)- 1j*alpha*(-1j)*n_x_flip \
        - U*(m_x_x + m_y_x*(1j)) * plus_x_flip_du + U*(m_x + m_y*(-1j))*plus_x_flip_ud\
        + U*( (n_x_u+n_x_d)/2.0 - (n_u + n_d)/2.0 + m_z_x - m_z )*plus_x_noflip_down) 

    minus_x_noflip_up_der = 1/1j * (-t*(n_u-n_minx_u)  - 1j*alpha*(-1j)*n_flip + 1j*alpha*(1j)*conj(n_minx_flip) \
        - U*(m_x_minx + m_y_minx*(-1j)) * minus_x_flip_ud + U*(m_x + m_y*(1j))*minus_x_flip_du \
        + U*( (n_minx_u+n_minx_d)/2.0 - (n_u + n_d)/2.0 - m_z_minx + m_z )*minus_x_noflip_up) 
    
    minus_x_noflip_down_der = 1/1j * (-t*(n_d-n_minx_d)  - 1j*alpha*(1j)*conj(n_flip) + 1j*alpha*(-1j)*n_minx_flip \
        - U*(m_x_minx + m_y_minx*(1j)) * minus_x_flip_du + U*(m_x + m_y*(-1j))*minus_x_flip_ud \
        + U*( (n_minx_u+n_minx_d)/2.0 - (n_u + n_d)/2.0 + m_z_minx - m_z )*minus_x_noflip_down) 

    plus_y_noflip_up_der = 1/1j * (-t*(n_u-n_y_u)  - 1j*alpha*n_flip + 1j*alpha*conj(n_y_flip) \
        - U*(m_x_y + m_y_y*(-1j)) * plus_y_flip_ud + U*(m_x + m_y*(1j))*plus_y_flip_du \
        + U*( (n_y_u+n_y_d)/2.0 - (n_u + n_d)/2.0 - m_z_y + m_z )*plus_y_noflip_up) 
    
    plus_y_noflip_down_der = 1/1j * (-t*(n_d-n_y_d)  - 1j*alpha*conj(n_flip) + 1j*alpha*n_y_flip \
        - U*(m_x_y + m_y_y*(1j)) * plus_y_flip_du + U*(m_x + m_y*(-1j))*plus_y_flip_ud \
        + U*( (n_y_u+n_y_d)/2.0 - (n_u + n_d)/2.0 + m_z_y - m_z )*plus_y_noflip_down) 
    
    minus_y_noflip_up_der = 1/1j * (-t*(n_u-n_miny_u)  + 1j*alpha*n_flip - 1j*alpha*conj(n_miny_flip) \
        - U*(m_x_miny + m_y_miny*(-1j)) * minus_y_flip_ud + U*(m_x + m_y*(1j))*minus_y_flip_du \
        + U*( (n_miny_u+n_miny_d)/2.0 - (n_u + n_d)/2.0 - m_z_miny + m_z )*minus_y_noflip_up) 

    minus_y_noflip_down_der = 1/1j * (-t*(n_d-n_miny_d)  + 1j*alpha*conj(n_flip) - 1j*alpha*n_miny_flip \
        - U*(m_x_miny + m_y_miny*(1j)) * minus_y_flip_du + U*(m_x + m_y*(-1j))*minus_y_flip_ud\
        + U*( (n_miny_u+n_miny_d)/2.0 - (n_u + n_d)/2.0 + m_z_miny - m_z )*minus_y_noflip_down) 
    
    plus_x_flip_ud_der = 1/1j * (-t*(n_flip - n_x_flip) + 1j*alpha*(1j)*n_u -1j*alpha*(1j)*n_x_d \
        - U*((m_x_x + m_y_x*(1j)) * plus_x_noflip_up - (m_x + m_y*(1j))*plus_x_noflip_down) \
        +  U*(-(n_u+n_d)/2.0 +(n_x_u+n_x_d)/2.0 + m_z + m_z_x) * plus_x_flip_ud)
        
    plus_x_flip_du_der = 1/1j * (-t*(conj(n_flip) - conj(n_x_flip)) +1j*alpha*(-1j)*n_d -1j*alpha*(-1j)*n_x_u \
        - U*((m_x_x + m_y_x*(-1j)) * plus_x_noflip_down - (m_x + m_y*(-1j))*plus_x_noflip_up) \
        +  U*(-(n_u+n_d)/2.0 +(n_x_u+n_x_d)/2.0 - m_z - m_z_x) * plus_x_flip_du )

    minus_x_flip_ud_der = 1/1j * (-t*(n_flip - n_minx_flip) -1j*alpha*(1j)*n_u +1j*alpha*(1j)*n_minx_d \
        - U*((m_x_minx + m_y_minx*(1j)) * minus_x_noflip_up - (m_x + m_y*(1j))*minus_x_noflip_down) \
        +  U*(-(n_u+n_d)/2.0 +(n_minx_u+n_minx_d)/2.0 + m_z + m_z_minx) * minus_x_flip_ud)

    minus_x_flip_du_der = 1/1j * (-t*(conj(n_flip) - conj(n_minx_flip)) -1j*alpha*(-1j)*n_d+1j*alpha*(-1j)*n_minx_u
        - U*((m_x_minx + m_y_minx*(-1j)) * minus_x_noflip_down - (m_x + m_y*(-1j))*minus_x_noflip_up) \
        +  U*(-(n_u+n_d)/2.0 +(n_minx_u+n_minx_d)/2.0 - m_z - m_z_minx) * minus_x_flip_du)

    plus_y_flip_ud_der = 1/1j * (-t*(n_flip - n_y_flip) -1j*alpha*n_u +1j*alpha*n_y_d \
        - U*((m_x_y + m_y_y*(1j)) * plus_y_noflip_up - (m_x + m_y*(1j))*plus_y_noflip_down) \
        +  U*(-(n_u+n_d)/2.0 + (n_y_u+n_y_d)/2.0 + m_z + m_z_y) * plus_y_flip_ud)

    plus_y_flip_du_der = 1/1j * (-t*(conj(n_flip) - conj(n_y_flip)) -1j*alpha*n_d + 1j*alpha*n_y_u
        - U*((m_x_y + m_y_y*(-1j)) * plus_y_noflip_down - (m_x + m_y*(-1j))*plus_y_noflip_up) \
        +  U*(-(n_u+n_d)/2.0 + (n_y_u+n_y_d)/2.0 - m_z - m_z_y) * plus_y_flip_du)

    minus_y_flip_ud_der = 1/1j * (-t*(n_flip - n_miny_flip) +1j*alpha*n_u - 1j*alpha*n_miny_d \
        - U*((m_x_miny + m_y_miny*(1j)) * minus_y_noflip_up - (m_x + m_y*(1j))*minus_y_noflip_down) \
        +  U*(-(n_u+n_d)/2.0 + (n_miny_u+n_miny_d)/2.0 + m_z + m_z_miny) * minus_y_flip_ud)

    minus_y_flip_du_der =1/1j * (-t*(conj(n_flip) - conj(n_miny_flip)) + 1j*alpha*n_d -1j*alpha*n_miny_u \
        - U*((m_x_miny + m_y_miny*(-1j)) * minus_y_noflip_down - (m_x + m_y*(-1j))*minus_y_noflip_up) \
        +  U*(-(n_u+n_d)/2.0 + (n_miny_u+n_miny_d)/2.0 - m_z - m_z_miny) * minus_y_flip_du)

    return n_u_der, n_d_der, n_flip_der, plus_x_noflip_up_der, plus_x_noflip_down_der, minus_x_noflip_up_der, minus_x_noflip_down_der, \
        plus_y_noflip_up_der, plus_y_noflip_down_der, minus_y_noflip_up_der, minus_y_noflip_down_der, plus_x_flip_ud_der, plus_x_flip_du_der, \
        minus_x_flip_ud_der, minus_x_flip_du_der, plus_y_flip_ud_der, plus_y_flip_du_der, minus_y_flip_ud_der, minus_y_flip_du_der