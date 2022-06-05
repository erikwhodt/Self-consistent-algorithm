import numpy as np

def fermi(beta, mu, E):
    return 1/(np.exp(beta*(E-mu))+1)

def h_ij(t, alpha, direction, sign='upper'):
    if direction == 'y': 
        return np.array([[-t, -1j*alpha],[-1j*alpha, -t]])
    elif direction == 'x': 
        if sign == 'upper':
            return np.array([[-t, -alpha],[alpha, -t]])
        elif sign == 'lower': 
            return np.array([[-t, alpha],[-alpha, -t]])


## The restricted matrix restricts the magnetic texture to a specific Q-vector, effectively forcing the magnetization. The unrestricted matrix takes in the spin-expectation 
## values from one iteration with the restricted, effectively using it as a seed, but without the Q-restriction, letting the pattern evolve freely.
def h_ii_restricted(i, xi, h, psi, Q, angle, N, direction, mu): 
    coordinate = np.array([i//N, (i%N)])
    if direction == 'x':
        return np.array([[-xi-h*np.cos(psi)/2-mu, -h/2 * np.sin(psi)*np.exp(-1j*(angle+np.dot(Q, coordinate)))],[-h/2 * np.sin(psi)*np.exp(1j*(angle+np.dot(Q, coordinate))), -xi + h*np.cos(psi)/2-mu]],dtype=complex)
    elif direction == 'z': 
        return np.array([[-xi-h/2 * np.sin(psi)*np.cos(np.dot(Q, coordinate))-mu, -h/2 *  (np.sin(psi)*np.sin(np.dot(Q,coordinate))-1j*np.cos(psi))],[-h/2 *  (np.sin(psi)*np.sin(np.dot(Q,coordinate))+1j*np.cos(psi)),-xi+h/2 * np.sin(psi)*np.cos(np.dot(Q, coordinate))-mu]],dtype=complex)
    elif direction == 'y':
        return np.array([[-xi-h/2 * np.sin(psi)*np.sin(np.dot(Q, coordinate))-mu, -h/2 *  (np.cos(psi)-1j*np.sin(psi)*np.cos(np.dot(Q,coordinate)))],[ -h/2 *  (np.cos(psi)+1j*np.sin(psi)*np.cos(np.dot(Q,coordinate))),-xi+h/2 * np.sin(psi)*np.sin(np.dot(Q, coordinate))-mu]],dtype=complex)

def h_ii_unrestricted(xi, h_x, h_y, h_z, mu):
    return np.array([[-xi-h_z-mu, -h_x+1j*h_y],[-h_x-1j*h_y, -xi + h_z - mu]],dtype=complex)

def matrix(n_array, m_array, m_x_array, m_y_array, m_z_array, psi, Q, N, U, mu, angle, t, alpha, direction, PM_flag, seed_flag): 
    mat = np.zeros((2*N**2,2*N**2), dtype=complex)

    if PM_flag: 
        m_array = np.full(N**2, 0)
    for i in range(N**2): 
        for j in range(N**2):            
            if i == j: 
                if seed_flag:
                    mat[i*2:i*2+2, i*2:i*2+2] = h_ii_restricted(i,-U*n_array[i]/2, 2*U*m_array[i],psi,Q, angle, N, direction, mu)
                else: 
                    mat[i*2:i*2+2, i*2:i*2+2] = h_ii_unrestricted(-U*n_array[i]/2, U*m_x_array[i], U*m_y_array[i], U*m_z_array[i], mu)
            elif j == i+1 and i%N != (N-1): 
                mat[j*2:j*2+2, i*2:i*2+2] = np.conj(h_ij(t, alpha, 'y'))
            elif j == i-1 and i%N != 0: 
                mat[j*2:j*2+2, i*2:i*2+2] = h_ij(t, alpha, 'y')
            elif j == i + N: 
                mat[j*2:j*2+2, i*2:i*2+2] = h_ij(t, alpha, 'x', 'lower')
            elif j == i - N: 
                mat[j*2:j*2+2, i*2:i*2+2] = h_ij(t, alpha, 'x', 'upper')
    return mat

# def free_energy(n_array, U, h_array, eig_val, N):
#     energy = 0
#     for i in range(N**2):
#         energy += -U*n_array[i]**2 / 4 + U*(h_array[i]/(2*U))**2
#     for i in range(2*N**2): 
#         energy += -eig_val[i]
#     return energy

# def evolution_function(
#     t, alpha, U, n_u, n_d, n_x_u, n_minx_u, n_y_u, n_miny_u, n_x_d, n_minx_d, n_y_d, n_miny_d, n_x_flip, n_minx_flip, n_y_flip, n_miny_flip, 
#     n_flip, m_x, m_y,  m_z, m_x_x, m_x_minx, m_x_y, m_x_miny, m_y_x, m_y_minx, m_y_y, m_y_miny, m_z_x, m_z_minx, m_z_y, m_z_miny ,plus_x_noflip_up, plus_x_noflip_down, minus_x_noflip_up, minus_x_noflip_down, plus_y_noflip_up, plus_y_noflip_down, 
#     minus_y_noflip_up, minus_y_noflip_down, plus_x_flip_ud, plus_x_flip_du, minus_x_flip_ud, minus_x_flip_du,  plus_y_flip_ud, plus_y_flip_du, 
#     minus_y_flip_ud, minus_y_flip_du):

#     n_u_der = 1/1j * (-t*(plus_x_noflip_up + minus_x_noflip_up + plus_y_noflip_up + minus_y_noflip_up \
#         - np.conj(plus_x_noflip_up) - np.conj(minus_x_noflip_up) - np.conj(plus_y_noflip_up) - np.conj(minus_y_noflip_up)) \
#         - 1j*alpha*(-1j)*plus_x_flip_ud + 1j*alpha*(-1j)*minus_x_flip_ud +1j*alpha*plus_y_flip_ud - 1j*alpha*minus_y_flip_ud \
#         + 1j*alpha*(1j)*np.conj(plus_x_flip_ud) -1j*alpha*(1j)*np.conj(minus_x_flip_ud)-1j*alpha*np.conj(plus_y_flip_ud) \
#         + 1j*alpha*np.conj(minus_y_flip_ud) \
#         - U*(m_x * (n_flip-np.conj(n_flip))+ m_y * ((-1j)*n_flip - (1j) * np.conj(n_flip)) ))

#     n_d_der = 1/1j * (-t*(plus_x_noflip_down + minus_x_noflip_down + plus_y_noflip_down + minus_y_noflip_down \
#         - np.conj(plus_x_noflip_down) - np.conj(minus_x_noflip_down) - np.conj(plus_y_noflip_down) - np.conj(minus_y_noflip_down)) \
#         - 1j*alpha*(1j)*plus_x_flip_du + 1j*alpha*(1j)*minus_x_flip_du +1j*alpha*plus_y_flip_du - 1j*alpha*minus_y_flip_du \
#         + 1j*alpha*(-1j)*np.conj(plus_x_flip_du) -1j*alpha*(-1j)*np.conj(minus_x_flip_du)-1j*alpha*np.conj(plus_y_flip_du) \
#         + 1j*alpha*np.conj(minus_y_flip_du) \
#         - U*(m_x * (np.conj(n_flip) - n_flip)+ m_y * ((1j)*np.conj(n_flip) - (-1j) * n_flip)))

#     n_flip_der = 1/1j * (-t*(plus_x_flip_ud + minus_x_flip_ud + plus_y_flip_ud + minus_y_flip_ud  
#         - np.conj(plus_x_flip_du) - np.conj(minus_x_flip_du) - np.conj(plus_y_flip_du) - np.conj(minus_y_flip_du)) \
#         - 1j*alpha*(1j)*plus_x_noflip_up +1j*alpha*(1j)*minus_x_noflip_up +1j*alpha*plus_y_noflip_up -1j*alpha*minus_y_noflip_up +1j*alpha*(1j)*np.conj(plus_x_noflip_down) \
#         - 1j*alpha*(1j)*np.conj(minus_x_noflip_down)-1j*alpha*np.conj(plus_y_noflip_down)+1j*alpha*np.conj(minus_y_noflip_down) \
#         + 2*U*m_z*n_flip \
#         - U*(m_x + m_y * (1j)) * (n_u - n_d))

#     plus_x_noflip_up_der = 1/1j * (-t*(n_u-n_x_u)  + 1j*alpha*(-1j)*n_flip + 1j*alpha*(1j)*np.conj(n_x_flip) \
#         - U*(m_x_x + m_y_x*(-1j)) * plus_x_flip_ud + U*(m_x + m_y*(1j))*plus_x_flip_du \
#         + U*( (n_x_u+n_x_d)/2 - (n_u + n_d)/2 - m_z_x + m_z )*plus_x_noflip_up) 
    
#     plus_x_noflip_down_der = 1/1j * (-t*(n_d-n_x_d)  + 1j*alpha*(1j)*np.conj(n_flip)+ 1j*alpha*(-1j)*n_x_flip \
#         - U*(m_x_x + m_y_x*(1j)) * plus_x_flip_du + U*(m_x + m_y*(-1j))*plus_x_flip_ud\
#         + U*( (n_x_u+n_x_d)/2 - (n_u + n_d)/2 + m_z_x - m_z )*plus_x_noflip_down) 

#     minus_x_noflip_up_der = 1/1j * (-t*(n_u-n_minx_u)  - 1j*alpha*(-1j)*n_flip - 1j*alpha*(1j)*np.conj(n_minx_flip) \
#         - U*(m_x_minx + m_y_minx*(-1j)) * minus_x_flip_ud + U*(m_x + m_y*(1j))*minus_x_flip_du \
#         + U*( (n_minx_u+n_minx_d)/2 - (n_u + n_d)/2 - m_z_minx + m_z )*minus_x_noflip_up) 
    
#     minus_x_noflip_down_der = 1/1j * (-t*(n_d-n_minx_d)  - 1j*alpha*(1j)*np.conj(n_flip) - 1j*alpha*(-1j)*n_minx_flip \
#         - U*(m_x_minx + m_y_minx*(1j)) * minus_x_flip_du + U*(m_x + m_y*(-1j))*minus_x_flip_ud \
#         + U*( (n_minx_u+n_minx_d)/2 - (n_u + n_d)/2 + m_z_minx - m_z )*minus_x_noflip_down) 

#     plus_y_noflip_up_der = 1/1j * (-t*(n_u-n_y_u)  - 1j*alpha*n_flip - 1j*alpha*np.conj(n_y_flip) \
#         - U*(m_x_y + m_y_y*(-1j)) * plus_y_flip_ud + U*(m_x + m_y*(1j))*plus_y_flip_du \
#         + U*( (n_y_u+n_y_d)/2 - (n_u + n_d)/2 - m_z_y + m_z )*plus_y_noflip_up) 
    
#     plus_y_noflip_down_der = 1/1j * (-t*(n_d-n_y_d)  - 1j*alpha*np.conj(n_flip) - 1j*alpha*n_y_flip \
#         - U*(m_x_y + m_y_y*(1j)) * plus_y_flip_du + U*(m_x + m_y*(-1j))*plus_y_flip_ud \
#         + U*( (n_y_u+n_y_d)/2 - (n_u + n_d)/2 + m_z_y - m_z )*plus_y_noflip_down) 
    
#     minus_y_noflip_up_der = 1/1j * (-t*(n_u-n_miny_u)  + 1j*alpha*n_flip + 1j*alpha*np.conj(n_miny_flip) \
#         - U*(m_x_miny + m_y_miny*(-1j)) * minus_y_flip_ud + U*(m_x + m_y*(1j))*minus_y_flip_du \
#         + U*( (n_miny_u+n_miny_d)/2 - (n_u + n_d)/2 - m_z_miny + m_z )*minus_y_noflip_up) 

#     minus_y_noflip_down_der = 1/1j * (-t*(n_d-n_miny_d)  + 1j*alpha*np.conj(n_flip) + 1j*alpha*n_miny_flip \
#         - U*(m_x_miny + m_y_miny*(1j)) * minus_y_flip_du + U*(m_x + m_y*(-1j))*minus_y_flip_ud\
#         + U*( (n_miny_u+n_miny_d)/2 - (n_u + n_d)/2 + m_z_miny - m_z )*minus_y_noflip_down) 
    
#     plus_x_flip_ud_der = 1/1j * (-t*(n_flip - n_x_flip) + 1j*alpha*(1j)*n_u +1j*alpha*(1j)*n_x_d \
#         - U*((m_x_x + m_y_x*(1j)) * plus_x_noflip_up - (m_x + m_y*(1j))*plus_x_noflip_down) \
#         +  U*(-(n_u+n_d)/2 +(n_x_u+n_x_d)/2 + m_z + m_z_x) * plus_x_flip_ud)
        
#     plus_x_flip_du_der = 1/1j * (-t*(np.conj(n_flip) - np.conj(n_x_flip)) +1j*alpha*(-1j)*n_d +1j*alpha*(-1j)*n_x_u \
#         - U*((m_x_x + m_y_x*(-1j)) * plus_x_noflip_down - (m_x + m_y*(-1j))*plus_x_noflip_up) \
#         +  U*(-(n_u+n_d)/2 +(n_x_u+n_x_d)/2 - m_z - m_z_x) * plus_x_flip_du )

#     minus_x_flip_ud_der = 1/1j * (-t*(n_flip - n_minx_flip) -1j*alpha*(1j)*n_u -1j*alpha*(1j)*n_minx_d \
#         - U*((m_x_minx + m_y_minx*(1j)) * minus_x_noflip_up - (m_x + m_y*(1j))*minus_x_noflip_down) \
#         +  U*(-(n_u+n_d)/2 +(n_minx_u+n_minx_d)/2 + m_z + m_z_minx) * minus_x_flip_ud)

#     minus_x_flip_du_der = 1/1j * (-t*(np.conj(n_flip) - np.conj(n_minx_flip)) -1j*alpha*(-1j)*n_d-1j*alpha*(-1j)*n_minx_u
#         - U*((m_x_minx + m_y_minx*(-1j)) * minus_x_noflip_down - (m_x + m_y*(-1j))*minus_x_noflip_up) \
#         +  U*(-(n_u+n_d)/2 +(n_minx_u+n_minx_d)/2 - m_z - m_z_minx) * minus_x_flip_du)

#     plus_y_flip_ud_der = 1/1j * (-t*(n_flip - n_y_flip) -1j*alpha*n_u -1j*alpha*n_y_d \
#         - U*((m_x_y + m_y_y*(1j)) * plus_y_noflip_up - (m_x + m_y*(1j))*plus_y_noflip_down) \
#         +  U*(-(n_u+n_d)/2 + (n_y_u+n_y_d)/2 + m_z + m_z_y) * plus_y_flip_ud)

#     plus_y_flip_du_der = 1/1j * (-t*(np.conj(n_flip) - np.conj(n_y_flip)) -1j*alpha*n_d - 1j*alpha*n_y_u
#         - U*((m_x_y + m_y_y*(-1j)) * plus_y_noflip_down - (m_x + m_y*(-1j))*plus_y_noflip_up) \
#         +  U*(-(n_u+n_d)/2 + (n_y_u+n_y_d)/2 - m_z - m_z_y) * plus_y_flip_du)

#     minus_y_flip_ud_der = 1/1j * (-t*(n_flip - n_miny_flip) +1j*alpha*n_u + 1j*alpha*n_miny_d \
#         - U*((m_x_miny + m_y_miny*(1j)) * minus_y_noflip_up - (m_x + m_y*(1j))*minus_y_noflip_down) \
#         +  U*(-(n_u+n_d)/2 + (n_miny_u+n_miny_d)/2 + m_z + m_z_miny) * minus_y_flip_ud)

#     minus_y_flip_du_der =1/1j * (-t*(np.conj(n_flip) - np.conj(n_miny_flip)) + 1j*alpha*n_d +1j*alpha*n_miny_u \
#         - U*((m_x_miny + m_y_miny*(-1j)) * minus_y_noflip_down - (m_x + m_y*(-1j))*minus_y_noflip_up) \
#         +  U*(-(n_u+n_d)/2 + (n_miny_u+n_miny_d)/2 - m_z - m_z_miny) * minus_y_flip_du)

#     return n_u_der, n_d_der, n_flip_der, plus_x_noflip_up_der, plus_x_noflip_down_der, minus_x_noflip_up_der, minus_x_noflip_down_der, \
#         plus_y_noflip_up_der, plus_y_noflip_down_der, minus_y_noflip_up_der, minus_y_noflip_down_der, plus_x_flip_ud_der, plus_x_flip_du_der, \
#         minus_x_flip_ud_der, minus_x_flip_du_der, plus_y_flip_ud_der, plus_y_flip_du_der, minus_y_flip_ud_der, minus_y_flip_du_der