import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from scipy.integrate import quad
import math
import time

class hf_solver:
    '''
    This is a high-fidelity solver for time-evolving wave functions, used to generate training data and animations. 

    Methods:
    1. get_ranges = Used the store the spacial and temporal grids in external variables
    2. set_initial_state: Sets the initial state of the wavefunction.
    3. construct_hamiltonian: Generates the hamiltonian matrix. Input tag to choose your wavefunction, current takes: quantum harmonic oscillator ("QHO") or Gross-Pitaevskii ("GP")
    4. run_simulation: Used for generating training data. The collect_data attribute is used to determine the length of time data will be collected for. This is due to exponentially increasing error accumulation.
    5. run_animation: Plays an animation of the wave propagation.
    6. graph_energy: Generates an energy/time graph after a complete simulation. Works with both run_simulation and run_animation.

    Questions can be sent to kozeraja@msu.edu
    '''

    def __init__(self, h_bar, mass, ang_freq, dx, x_limit, dt, t_final):
        self.h_bar = h_bar
        self.mass = mass
        self.ang_freq = ang_freq
        self.dx = dx
        self.x_limit = x_limit
        self.dt = dt
        self.t_final = t_final

        self.Nx = int((2*self.x_limit)/self.dx)
        self.x_range = np.linspace(-self.x_limit, self.x_limit, self.Nx)

        self.Nt = int((self.t_final/self.dt))
        self.t_range = np.linspace(0, self.t_final, self.Nt)

    def get_ranges(self):
        return self.x_range, self.t_range
    
    def set_initial_state(self, x0=0, sigma=1/(2)**(1/2), p=0):
        # x0 is the inital position
        # x is current position
        # sigma controls the width of the wave
        # p is the momentum
        self.k = p / self.h_bar
        normalization = 1 / np.sqrt(sigma * np.sqrt(np.pi))
        gaussian_term = np.exp(1j * self.k * self.x_range - (self.x_range - x0)**2 / (4 * sigma**2))
        self.initial_state = normalization * gaussian_term
        
    def construct_hamiltonian(self, type):
        self.type = type

        hamiltonian = np.zeros((self.Nx,self.Nx))

        P = 0.5 * self.mass * self.ang_freq**2 * self.x_range**2
        self.potential = np.diag(P)

        kinetic = -(self.h_bar**2 / (2 * self.mass)) * (
        -(5/2) * np.eye(self.Nx) +
        (4/3) * np.eye(self.Nx, k=1) +
        (4/3) * np.eye(self.Nx, k=-1) +
        -(1/12) * np.eye(self.Nx, k=2) +
        -(1/12) * np.eye(self.Nx, k=-2)
        ) / self.dx**2
        
        if self.type == "QHO":
            hamiltonian = kinetic + self.potential
        elif self.type == "GP":
            hamiltonian = kinetic + self.potential + np.diag(np.real(np.conj(self.initial_state)*self.initial_state/(np.dot(self.initial_state,np.conj(self.initial_state))*self.dx)))

        self.hamiltonian = hamiltonian


    def run_simulation(self, collect_data):
        self.t_i = 0
        count = 0
        total_energies = []

        time_start = time.time()

        eigenvalues, eigenvectors = sp.linalg.eigh(self.hamiltonian)
        all_states = np.zeros((collect_data, *np.shape(self.hamiltonian)), dtype=complex)

        for t in self.t_range:
            total_energy = 0

            if t == 0:
                final_state = self.initial_state.copy()
                final_state_norm = final_state/np.sqrt(np.dot(final_state,np.conj(final_state))*self.dx)
                def energy(x):
                    psi_x = final_state_norm
                    return np.real(np.conj(psi_x) @ self.hamiltonian @ psi_x)
                if count < collect_data:
                    all_states[count, :, :] = final_state_norm[:]
                total_energy, _ = quad(energy, -self.x_limit, self.x_limit)
                total_energies.append(total_energy)
    
            else:

                self.initial_state = final_state_norm[:].copy()
                self.construct_hamiltonian(self.type)
                teo = sp.linalg.expm((-1j)*self.hamiltonian*(self.dt))
                new_state = np.dot(teo, final_state_norm)
                final_state_norm = final_state_norm + new_state
                final_state_norm = final_state_norm/np.sqrt(np.dot(final_state_norm,np.conj(final_state_norm))*self.dx)
                if count < collect_data:
                    all_states[count, :, :] = final_state_norm[:]
                total_energy, _ = quad(energy, -self.x_limit, self.x_limit)
                total_energies.append(total_energy)

            self.t_i += self.dt
            count += 1

        time_finish = time.time()
        time_diff = time_finish - time_start
    
        self.total_energies = total_energies

        return final_state_norm, all_states, total_energies, time_diff
    
    def run_animation(self):
        total_energies = []

        time_start = time.time()

        eigenvalues, eigenvectors = sp.linalg.eigh(self.hamiltonian)

        fig, ax = plt.subplots(figsize=(10, 5))

        for t in self.t_range:
        

            if t == 0:
                total_energy = 0

                final_state = self.initial_state.copy()
                final_state_norm = final_state/np.sqrt(np.dot(final_state,np.conj(final_state))*self.dx)
                def energy(x):
                    psi_x = final_state_norm
                    return np.real(np.conj(psi_x) @ self.hamiltonian @ psi_x)
                total_energy, _ = quad(energy, -self.x_limit, self.x_limit)
                total_energies.append(total_energy)

                plt.plot(self.x_range, (np.real(final_state_norm[:])))
                plt.plot(self.x_range, (np.imag(final_state_norm[:])))
    
                plt.xlim([-self.x_limit, self.x_limit])
                plt.ylim([-1,1])

                plt.xlabel("Position")
                plt.ylabel("Wavefunction")
                plt.legend(["Real","Imaginary"])
                clear_output(wait=True)  
                display(fig) 
                fig.clear()
    
            else:
                self.initial_state = final_state_norm[:].copy()
                self.construct_hamiltonian(self.type)
                teo = sp.linalg.expm((-1j)*self.hamiltonian*(self.dt))
                new_state = np.dot(teo, final_state_norm)
                final_state_norm = final_state_norm + new_state
                final_state_norm = final_state_norm/np.sqrt(np.dot(final_state_norm,np.conj(final_state_norm))*self.dx)
                total_energy, _ = quad(energy, -self.x_limit, self.x_limit)
                total_energies.append(total_energy)

                plt.plot(self.x_range, (np.real(final_state_norm[:])))
                plt.plot(self.x_range, (np.imag(final_state_norm[:])))
    
                plt.xlim([-self.x_limit,self.x_limit])
                plt.ylim([-1,1])

                plt.xlabel("Position")
                plt.ylabel("Wavefunction")
                plt.legend(["Real","Imaginary"])
                clear_output(wait=True)  
                display(fig) 
                fig.clear()


            time_finish = time.time()
        time_diff = time_finish - time_start
        print("Time taken:", time_diff, "sec")

        self.total_energies = total_energies

    def graph_energy(self):
        print("Length of self.t:", len(self.t_range))
        print("Length of self.total_energies:", len(self.total_energies))
        plt.plot(self.t_range, self.total_energies)
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title("Ground State Energy")