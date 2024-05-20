import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from scipy.integrate import quad
import math
import time
import os
import imageio


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

    def __init__(self, h_bar, mass, ang_freq, q, dx, x_limit, dt, t_final):
        self.h_bar = h_bar
        self.mass = mass
        self.ang_freq = ang_freq
        self.q = q
        self.dx = dx
        self.x_limit = x_limit
        self.dt = dt
        self.t_final = t_final
        self.frames = []

        self.Nx = int((2*self.x_limit)/self.dx)
        self.x_range = np.linspace(-self.x_limit, self.x_limit, self.Nx)

        self.Nt = int((self.t_final/self.dt))
        self.t_range = np.linspace(0, self.t_final, self.Nt)

    def get_ranges(self):
        return self.x_range, self.t_range
    
    def get_all_states(self):
        return self.all_states
    
    def get_run_time(self):
        return self.time_diff
    
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

        self.kinetic = -(self.h_bar**2 / (2 * self.mass)) * (
        -(5/2) * np.eye(self.Nx) +
        (4/3) * np.eye(self.Nx, k=1) +
        (4/3) * np.eye(self.Nx, k=-1) +
        -(1/12) * np.eye(self.Nx, k=2) +
        -(1/12) * np.eye(self.Nx, k=-2)
        ) / self.dx**2
        
        if self.type == "QHO":
            hamiltonian = self.kinetic + self.potential
        elif self.type == "GP":
            hamiltonian = self.kinetic + self.potential + (self.q * np.diag(np.real(np.conj(self.initial_state)*self.initial_state/(np.dot(self.initial_state,np.conj(self.initial_state))*self.dx))))

        self.hamiltonian = hamiltonian
    
    def run_solver(self, gif = False, animation = True, collect_data = False):
        total_energies = np.ones(self.Nt)
        total_kinetics = np.ones(self.Nt)
        total_potentials = np.ones(self.Nt)
        particle_num = np.ones(self.Nt)
        all_states = np.zeros((self.Nt, *np.shape(self.hamiltonian)), dtype=complex)
        step = 0

        time_start = time.time()

        eigenvalues, eigenvectors = sp.linalg.eigh(self.hamiltonian)

        if animation == True:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            pass

        for t in self.t_range:
        

            if t == 0:
                final_state = self.initial_state.copy()

                final_state_norm = final_state/np.sqrt(np.dot(final_state,np.conj(final_state))*self.dx)

                total_energies[step] = np.dot(np.dot(np.conj(final_state_norm[:]),self.hamiltonian), final_state_norm[:])
                total_kinetics[step] = np.dot(np.dot(np.conj(final_state_norm[:]),self.kinetic), final_state_norm[:])
                total_potentials[step] = np.dot(np.dot(np.conj(final_state_norm[:]),self.potential), final_state_norm[:])
                particle_num[step] = sp.integrate.trapezoid(np.abs(final_state_norm[:])**2, self.x_range)

                if collect_data == True:
                    all_states[step, :, :] = final_state_norm[:]
                else:
                    pass
    
            else:
                self.initial_state = final_state_norm[:].copy()
                self.construct_hamiltonian(self.type)
                teo_pred = sp.linalg.expm((-1j)*self.hamiltonian*(self.dt/2))
                self.initial_state = np.dot(teo_pred, final_state_norm)
                self.construct_hamiltonian(self.type)
                teo_final = sp.linalg.expm((-1j)*self.hamiltonian*(self.dt))
                new_state = np.dot(teo_final, final_state_norm)

                final_state_norm = final_state_norm + 0.5*(new_state+self.initial_state)
                final_state_norm = final_state_norm/np.sqrt(np.dot(final_state_norm,np.conj(final_state_norm))*self.dx)

                total_energies[step] = np.dot(np.dot(np.conj(final_state_norm[:]),self.hamiltonian), final_state_norm[:])
                total_kinetics[step] = np.dot(np.dot(np.conj(final_state_norm[:]),self.kinetic), final_state_norm[:])
                total_potentials[step] = np.dot(np.dot(np.conj(final_state_norm[:]),self.potential), final_state_norm[:])
                particle_num[step] = sp.integrate.trapezoid(np.abs(final_state_norm[:])**2, self.x_range)

                if collect_data == True:
                    all_states[step, :, :] = final_state_norm[:]
                else:
                    pass

            if animation == True:
                plt.plot(self.x_range, (np.real(final_state_norm[:])))
                plt.plot(self.x_range, (np.imag(final_state_norm[:])))
                plt.plot(self.x_range, np.abs(final_state_norm[:])**2)
    
                plt.xlim([-self.x_limit,self.x_limit])
                plt.ylim([-1,1])
                plt.xlabel("Position")
                plt.ylabel("Wavefunction")
                plt.legend(["Real","Imaginary", "Density"])

                frame_filename = f"frame_{t}.png"
                fig.savefig(frame_filename)
                self.frames.append(frame_filename)

                clear_output(wait=True)  
                display(fig) 
                fig.clear()
            else:
                pass

            step += 1

        time_finish = time.time()
        time_diff = time_finish - time_start
        print("Time taken:", time_diff, "sec")

        if gif == True and animation == True:
            images = []
            for frame in self.frames:
                images.append(imageio.imread(frame))
                os.remove(frame)  # Remove the temporary image file after adding it to the GIF
            imageio.mimsave('animation.mp4', images, fps=20)  # Adjust duration as needed
        else:
            for frame in self.frames:
                os.remove(frame)

        self.total_energies = total_energies
        self.total_kinetics = total_kinetics
        self.total_potentials = total_potentials
        self.particle_num = particle_num
        self.all_states = all_states
        self.time_diff = time_diff
        

    def graph_energy(self):
        plt.plot(self.t_range, self.total_energies)
        plt.plot(self.t_range, self.total_kinetics)
        plt.plot(self.t_range, self.total_potentials)
        plt.legend(["Total","Kinetic","Potential"])
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title("Ground State Energy")
        plt.savefig('Energy.png')

    def graph_particle_num(self):
        plt.plot(self.t_range, self.particle_num)
        plt.xlabel("Time")
        plt.ylabel("Particle Number")
        plt.title("Particle Number")
        plt.savefig('Part_num.png')