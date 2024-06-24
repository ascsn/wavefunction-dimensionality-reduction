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
    This is a high-fidelity solver for time-evolving wave functions. 

    Methods:
    4. set_initial_state: Sets the initial state of the wavefunction
    5. construct_hamiltonian: Generates the hamiltonian matrix. Input tag to choose your wavefunction, currently takes: quantum harmonic oscillator ("QHO") or Gross-Pitaevskii ("GP")
    6. run_solver: Runs the time-stepper
    7. animation: Plays an animation in the iPython viewer, downloads an MP4
    8. graph_energy: Generates an energy/time graph
    9. graph_particle_num: Generates a particle number/time graph

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
        self.t_range = np.linspace(0, self.t_final, self.Nt + 1)
    
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
    
    def run_solver(self, snapshot = False, snapshot_freq = 1):
        total_energies = np.ones(self.Nt + 1)
        total_kinetics = np.ones(self.Nt + 1)
        total_potentials = np.ones(self.Nt + 1)
        particle_num = np.ones(self.Nt + 1)
        all_states = np.zeros((self.Nt + 1, *np.shape(self.hamiltonian)), dtype=complex)
        snapshots = []
        step = 0

        time_start = time.time()

        eigenvalues, eigenvectors = sp.linalg.eigh(self.hamiltonian)

        for t in self.t_range:
        

            if t == 0:
                final_state = self.initial_state.copy()
                #The line below evolves the ground-state eigenvector of the equation. Set index to higher integers for excited states.
                #final_state  = eigenvectors[:,0].T.copy()

                final_state_norm = final_state/np.sqrt(np.dot(final_state,np.conj(final_state))*self.dx)

                total_energies[step] = np.dot(np.dot(np.conj(final_state_norm[:]),self.hamiltonian), final_state_norm[:])
                total_kinetics[step] = np.dot(np.dot(np.conj(final_state_norm[:]),self.kinetic), final_state_norm[:])
                total_potentials[step] = np.dot(np.dot(np.conj(final_state_norm[:]),self.potential), final_state_norm[:])
                particle_num[step] = sp.integrate.trapezoid(np.abs(final_state_norm[:])**2, self.x_range)

                all_states[step, :] = final_state_norm[:]
    
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

                all_states[step, :] = final_state_norm[:]

            if step % snapshot_freq == 0 and snapshot == True:
                snapshots.append(final_state_norm.copy())

            step += 1

        time_finish = time.time()
        time_diff = time_finish - time_start
        print("Time taken:", time_diff, "sec")

        self.total_energies = total_energies
        self.total_kinetics = total_kinetics
        self.total_potentials = total_potentials
        self.particle_num = particle_num
        self.all_states = all_states
        self.snapshots = snapshots
        self.time_diff = time_diff
        
    def animation(self, show = False, save = False, fps = 30):
        if show or save:
            fig, ax = plt.subplots(figsize=(10, 5))
            images = []
            for i in range(len(self.t_range)):
                ax.plot(self.x_range, np.real(self.all_states[i, 0, :]), label="Real")
                ax.plot(self.x_range, np.imag(self.all_states[i, 0, :]), label="Imaginary")
                ax.plot(self.x_range, np.abs(self.all_states[i, 0, :])**2, label="Density")
    
                ax.set_xlim([-self.x_limit, self.x_limit])
                ax.set_ylim([-1, 1])
                ax.set_xlabel("Position")
                ax.set_ylabel("Wavefunction")
                ax.legend()
    
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image)
    
                if show:
                    clear_output(wait=True)
                    display(fig)
                ax.clear()
    
            plt.close(fig)
            
            if save:
                imageio.mimsave('animation.mp4', images, fps=fps)  # Adjust fps as needed

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

    def get_ranges(self):
        return self.x_range, self.t_range
    
    def get_hamiltonian(self):
        return self.hamiltonian

    def get_kinetics(self):
        return self.total_kinetics
    
    def get_potentials(self):
        return self.total_potentials

    def get_energies(self):
        return self.total_energies
    
    def get_all_states(self):
        return self.all_states
    
    def get_snapshots(self):
        return self.snapshots
    
    def get_run_time(self):
        return self.time_diff