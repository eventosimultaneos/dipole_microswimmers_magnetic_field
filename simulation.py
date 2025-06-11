# Simulation of Janus particles with dipole interactions in a magnetic field using VPython
import csv
import os
import time
import numpy as np
from vpython import *

# PARAMETERS

# Number of particles
N = 100

# Number of steps
steps = 100_000

# Time step size in seconds (When there are interactions: dt=1e-4)
dt = 1e-5

# Number of times the experiment is repeated (for a fit between theory and simulation use iterations > 350)
iterations = 50
batch_size = 100

# Particle diameter (m)
sigma = 1e-6    

# Solvent temperature (K)
T = 300         

# Solvent viscosity (Pa s)
eta = 1e-3      

# Magnetic permeability of the medium (H/m)
mu = 1.257e-6                        

# Magnitude of the particle's dipole moment (A m^2)
mm = 1.15e-16 * 2 * 10
#mm = 0

# Values of the external magnetic field (T)
B_values = [0.0001]

# Particle velocity (m/s)
v = 10 * sigma

# Maximum random position (for isotropic phase particles)
max_pos = 100 * sigma                     

# Separation between particles (in sigma) for initial grid configuration
separation_values = [1]                       

# Initial grid configuration
initial_grid_config = True
n_rows = 10
n_cols = 10


if initial_grid_config:
    if n_rows * n_cols != N:
        raise ValueError("The number of rows and columns must be equal to the number of particles")

# Initial angle (None for random)
initial_angle = None

# Brownian motion: 0 to deactivate, 1 to activate
brownian_motion = True                             

# Number of the steps to skip in the MSD calculation
skip_fraction = 0

B = 0.01
separation = 2
scale_factor = 1

if __name__ == '__main__':
    # Directories and filenames
    general_name = f'{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'
    x_dir_name = f'sqrt10xdipole_10xvel\\x_{general_name}'
    y_dir_name = f'sqrt10xdipole_10xvel\\y_{general_name}'
    phi_dir_name = f'sqrt10xdipole_10xvel\\phi_{general_name}'
    
    x_filename = os.path.join(x_dir_name, f'x_1it_{B*1000:.1f}mT.csv')
    y_filename = os.path.join(y_dir_name, f'y_1it_{B*1000:.1f}mT.csv')
    phi_filename = os.path.join(phi_dir_name, f'phi_1it_{B*1000:.1f}mT.csv')

    # Read data from CSV files
    def read_csv(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            print(f"Reading {filename}...")
            # Read data and convert to floats
            data = np.array([row[1:] for row in reader], dtype=float)  # Read data as floats
            print(f"Data shape: {data.shape}")
        return data
    
    x_data = read_csv(x_filename) / sigma
    y_data = read_csv(y_filename) / sigma
    phi_data = read_csv(phi_filename)

    # Create the scene in VPython
    scene = canvas(title=f"Dipole Janus Particles\n d₀ = {separation}σ, B = {B*1000:.1f} mT, v = {v/sigma} σ/s",
                   width=600, height=500, center=vector(0, 40, 0), background=color.black)

    # Create particles in VPython
    particles = []
    arrows = []

    for i in range(N):
        particle = sphere(pos=vector(x_data[0, i], y_data[0, i], 0),
                          radius=0.5, color=color.blue, opacity=0.7)
        arrow_dir = vector(np.cos(phi_data[0, i]), np.sin(phi_data[0, i]), 0) * scale_factor
        arrow_obj = arrow(pos=particle.pos, axis=arrow_dir, shaftwidth=0.1, color=color.red)
        particles.append(particle)
        arrows.append(arrow_obj)

    # Animation of the simulation
    for t in range(1, steps):
        rate(2000)  # Control the speed of the animation
        for i in range(N):
            # Update position
            new_pos = vector(x_data[t, i], y_data[t, i], 0)
            particles[i].pos = new_pos
            # Update orientation
            new_dir = vector(np.cos(phi_data[t, i]), np.sin(phi_data[t, i]), 0) * scale_factor
            arrows[i].pos = new_pos
            arrows[i].axis = new_dir
    # Pause for a moment before changing to the next separation
    time.sleep(1)
    
print("Simulation finished")
