import numpy as np

class Measurement: 

    def __init__(self, mu, alpha, polarization, Q, Q_text, n, m, TP, did_it_converge, threshold):
        self.mu = mu 
        self.alpha = alpha 
        self.polarization = polarization 
        self.Q = Q 
        self.Q_text = Q_text
        self.density = 0
        self.density_new = n
        self.magnetization = 0
        self.magnetization_new = m
        self.free_energy = TP 
        self.threshold = threshold
        self.did_it_converge = did_it_converge 

    def output(self):
        return [self.mu, self.alpha, self.polarization, self.Q_text, self.density_new, self.magnetization_new, self.free_energy, self.did_it_converge]

    def __repr__(self): 
        return f"\t## new dens: {self.density_new:.4f}\tnew mag: {self.magnetization_new:.4f}\tnew energy: {self.free_energy:.4f}"

    def has_not_converged(self): 
        if np.abs(self.density - self.density_new)/self.density_new > self.threshold or np.abs(self.magnetization - self.magnetization_new)/self.magnetization_new > self.threshold:
            return True
        else: 
            return False


class Temp: 

    initial_iteration = True

    def __init__(self): 
        self.density = 0
        self.magnetization = 0 
        self.energy = 0

    def set(self, n, m, e): 
        self.density = n
        self.magnetization = m 
        self.energy = e 
        self.initial_iteration = False
        
    def __eq__(self, other):
        if self.density ==other.density and self.magnetization == other.magnetization and self.energy == other.energy and not self.initial_iteration:
            return True 
        else: 
            return False
