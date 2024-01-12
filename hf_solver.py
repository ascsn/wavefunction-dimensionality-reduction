import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from scipy.integrate import quad
import math
import time

class hf_solver:
    '''
    This is a high-fidelity solver for time-evolving wave functions, used to generate training data and animations. The quantum harmonic oscillator is currently hardcoded in. Further work is required to 
    prep the class for non-linear wave functions.

    Methods:
    1. set_potential: Initializes the spacial grid and sets the potential of the wavefunction. Ideally the potential function would be a variable that could be set upon initialization. Further work required.
    2. set_domain: Used set the domain so it does not have to be hardcoded in. Currently non-functional, further work required.
    3. set_initial_state: Sets the initial state of the wavefunction.
    4. construct_hamiltonian: Generates the hamiltonian matrix.
    5. time_evolution: Initializes the timeframe/timesteps, preforms a taylor series expansion on the time evolution operator.
    6. run_simulation: Used for generating training data. The collect_data attribute is used to determine the length of time data will be collected for. This is due to exponentially increasing error accumulation.
    7. run_animation: Plays an animation of the wave propagation.
    8. graph_energy: Generates an energy/time graph after a complete simulation. Works with both run_simulation and run_animation.

    Questions can be sent to kozeraja@msu.edu
    '''

    def __init__(self, h_bar, mass):
        self.h_bar = h_bar
        self.mass = mass

    def set_potential(self, Nx, x_min, x_max, q):
        self.Nx = Nx
        self.x_min = x_min
        self.x_max = x_max
        self.q = q

        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.dx = self.x[1] - self.x[0]

        P = 0.5 * self.mass * self.q**2 * self.x**2
        
        self.potential = np.diag(P)

    def set_domain(self):
        ary = self.x.copy()
        return ary

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state

    def construct_hamiltonian(self):
        hamiltonian = np.zeros((self.Nx,self.Nx))

        finite_diff = -(5/2) * np.eye(self.Nx)
        finite_diff += (4/3) * np.eye(self.Nx, k=1)
        finite_diff += (4/3) * np.eye(self.Nx, k=-1)
        finite_diff += -(1/12) * np.eye(self.Nx, k=2)
        finite_diff += -(1/12) * np.eye(self.Nx, k=-2)

        kinetic = -self.h_bar**2 / (2 * self.mass) * (finite_diff/self.dx**2)

        hamiltonian = kinetic + self.potential
        self.hamiltonian = hamiltonian

    
    def time_evolution(self, terms, Nt, t_final):
        self.Nt = Nt
        self.t_i = 0
        self.t_final = t_final

        self.t = np.linspace(self.t_i, t_final, Nt)
        self.dt = self.t[1] - self.t[0]

        n = self.hamiltonian.shape[0]
        identity = np.eye(n, dtype=np.complex128)
        result = identity.copy()
        matrix_power = self.hamiltonian.copy()

        # Expand time evolution operator e^(-itH)
        for i in range(1, terms + 1):
            term = (matrix_power**i) * (self.dt**i) * (-1j**i) / math.factorial(i)
            result += term

        self.time_evolution_operator = result

    def run_simulation(self, collect_data):
        self.t_i = 0
        count = 0
        total_energies = []

        time_start = time.time()

        eigenvalues, eigenvectors = sp.linalg.eigh(self.hamiltonian)
        all_states = np.zeros((collect_data, *np.shape(self.hamiltonian)), dtype=complex)

        while self.t_i <= self.t_final:
            total_energy = 0

            if self.t_i == 0:
                final_state = self.initial_state.copy()
                final_state_norm = final_state/np.sqrt(np.dot(final_state,np.conj(final_state))*self.dx)
                def energy(x):
                    psi_x = final_state_norm
                    return np.real(np.conj(psi_x) @ self.hamiltonian @ psi_x)
                if count < collect_data:
                    all_states[count, :, :] = final_state_norm[:]
                total_energy, _ = quad(energy, self.x_min, self.x_max)
                total_energies.append(total_energy)
    
            else:

                new_state = np.dot(self.time_evolution_operator, final_state_norm)
                final_state_norm = final_state_norm + new_state
                final_state_norm = final_state_norm/np.sqrt(np.dot(final_state_norm,np.conj(final_state_norm))*self.dx)
                if count < collect_data:
                    all_states[count, :, :] = final_state_norm[:]
                total_energy, _ = quad(energy, self.x_min, self.x_max)
                total_energies.append(total_energy)

            self.t_i += self.dt
            count += 1

        time_finish = time.time()
        time_diff = time_finish - time_start
    
        self.total_energies = total_energies

        return final_state_norm, all_states, total_energies, time_diff
    
    def run_animation(self):
        self.t_i = 0
        total_energies = []

        time_start = time.time()

        eigenvalues, eigenvectors = sp.linalg.eigh(self.hamiltonian)

        fig, ax = plt.subplots(figsize=(10, 5))

        while self.t_i <= self.t_final:
        

            if self.t_i == 0:
                total_energy = 0

                final_state = self.initial_state.copy()
                final_state_norm = final_state/np.sqrt(np.dot(final_state,np.conj(final_state))*self.dx)
                def energy(x):
                    psi_x = final_state_norm
                    return np.real(np.conj(psi_x) @ self.hamiltonian @ psi_x)
                total_energy, _ = quad(energy, self.x_min, self.x_max)
                total_energies.append(total_energy)

                plt.plot(self.x, (np.real(final_state_norm[:])))
                plt.plot(self.x, (np.imag(final_state_norm[:])))
    
                plt.xlim([self.x_min, self.x_max])
                plt.ylim([-1,1])

                plt.xlabel("Position")
                plt.ylabel("Wavefunction")
                plt.legend(["Real","Imaginary"])
                clear_output(wait=True)  
                display(fig) 
                fig.clear()
    
            else:

                new_state = np.dot(self.time_evolution_operator, final_state_norm)
                final_state_norm = final_state_norm + new_state
                final_state_norm = final_state_norm/np.sqrt(np.dot(final_state_norm,np.conj(final_state_norm))*self.dx)
                total_energy, _ = quad(energy, self.x_min, self.x_max)
                total_energies.append(total_energy)

                plt.plot(self.x, (np.real(final_state_norm[:])))
                plt.plot(self.x, (np.imag(final_state_norm[:])))
    
                plt.xlim([self.x_min,self.x_max])
                plt.ylim([-1,1])

                plt.xlabel("Position")
                plt.ylabel("Wavefunction")
                plt.legend(["Real","Imaginary"])
                clear_output(wait=True)  
                display(fig) 
                fig.clear()

            self.t_i += self.dt

            time_finish = time.time()
        time_diff = time_finish - time_start
        print("Time taken:", time_diff, "sec")

        self.total_energies = total_energies

    def graph_energy(self):
        plt.plot(self.t[:self.Nt-1], self.total_energies)
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title("Ground State Energy")