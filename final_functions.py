import os
import csv

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from statsmodels.tsa.stattools import acf, pacf
from scipy.spatial import cKDTree

from numba import njit, prange

from sklearn.neighbors import KernelDensity

from tqdm import tqdm


def generate_initial_conditions(
        N, steps, iterations, max_pos, separation, 
        initial_grid_config, n_rows, n_cols, initial_angle, 
        sigma, show_data=False, plot_configuration=False
    ):
    """
    Generates initial conditions for the particles and saves them to a CSV file.

    Parameters:
        N (int): Number of particles.
        steps (int): Number of time steps.
        iterations (int): Number of repetitions.
        max_pos (float): Maximum position for the particles.
        separation (float): Separation between particles in a grid configuration.
        initial_grid_config (bool): Whether to use a grid configuration for the initial positions.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        initial_angle (float): Initial angle for the particles.
        sigma (float): Characteristic size of the particles.
        show_data (bool): Whether to show the data in DataFrames. Default is False.
        plot_configuration (bool): Whether to plot the initial configuration. Default is False.
    Returns:
        str: Name of the CSV file with the initial conditions
    """
    # Define arrays for positions and angles
    x = np.zeros(N)
    y = np.zeros(N)
    phi = np.zeros(N)

    # Initial position configuration
    if initial_grid_config:
        if n_rows * n_cols != N:
            raise ValueError("The product of n_rows and n_cols must be equal to N")
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                x[idx] = j * separation * sigma
                y[idx] = i * separation * sigma
    else:
        x = max_pos * (2 * np.random.rand(N) - 1)
        y = max_pos * (2 * np.random.rand(N) - 1)

    # Assign initial angles
    if initial_angle is None:
        np.random.seed(42)  # Set a fixed seed for reproducibility
        phi = 2 * np.pi * np.random.rand(N)  # Random angles
    else:
        phi = np.full(N, initial_angle)

    # Create a CSV file name
    filename = f'initial_conditions_{N}part_{steps // 1000}ksteps_{iterations}its_{separation}sep.csv'

    # Save the initial conditions to a CSV file
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['particle', 'x', 'y', 'phi'])  # Headers
        for i in range(N):
            writer.writerow([i + 1, x[i], y[i], phi[i]])

    print(f"Initial conditions saved in {filename}")

    # If show_data is True, show the data in DataFrames
    if show_data:
        print("\nInitial conditions:")
        data = pd.read_csv(filename)
        print(data)
    
    if plot_configuration:
        # Create a plot with the initial configuration
        plt.figure(figsize=(8, 8))
        plt.scatter(x/sigma, y/sigma, c='blue', s=100)
        #Use quiver to plot the orientation of the particles
        for i in range(N):
            plt.quiver(x[i]/sigma, y[i]/sigma, np.cos(phi[i]), np.sin(phi[i]), 
                       angles='xy', scale_units='xy', scale=2, color='black',
                       width=0.005)
        plt.xlim(-1, max_pos + 1)
        plt.ylim(-1, max_pos + 1)
        plt.xlabel('x (σ)')
        plt.ylabel('y (σ)')
        plt.title('Initial Configuration')
        plt.grid(True)
        plt.show()
    return filename


def plot_initial_configuration(N, steps, iterations, separation, sigma, 
                                B, arrow_scale=None):
    """
    Plots the initial configuration of particles.

    Parameters:
        N: Number of particles.
        steps: Number of time steps.
        iterations: Number of repetitions.
        separation: Separation between particles in a grid configuration.
        dt: Time step.
        sigma: Characteristic size of the particles.
        v: Particle velocity.
        B: Magnetic field.
        mm: Magnetic moment.
        T: Temperature.
        brownian_motion (bool): Whether to include Brownian motion.
        arrow_scale: Scale of the arrows (optional).
    """
     
    # Load initial configuration data from CSV files
    x_data = np.zeros((N, iterations))
    y_data = np.zeros((N, iterations))
    phi_data = np.zeros((N, iterations))

    general_name = f'{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'

    x_dir_name = f'x_{general_name}'  # x directory name
    y_dir_name = f'y_{general_name}'  # y directory name
    phi_dir_name = f'phi_{general_name}'  # phi directory name

    for i in range(iterations):
        x_filename = os.path.join(x_dir_name, f'x_{i+1}it_{B*1000:.1f}mT.csv')
        y_filename = os.path.join(y_dir_name, f'y_{i+1}it_{B*1000:.1f}mT.csv')
        phi_filename = os.path.join(phi_dir_name, f'phi_{i+1}it_{B*1000:.1f}mT.csv')

        # Read CSV files
        with open(x_filename, 'r', encoding='utf-8') as x_file, \
             open(y_filename, 'r', encoding='utf-8') as y_file, \
             open(phi_filename, 'r', encoding='utf-8') as phi_file:
            
            x_reader = csv.reader(x_file)
            y_reader = csv.reader(y_file)
            phi_reader = csv.reader(phi_file)

            # Skip the first row (headers)
            next(x_reader)
            next(y_reader)
            next(phi_reader)

            # Read the first row (initial configuration)
            x_row = next(x_reader)
            y_row = next(y_reader)
            phi_row = next(phi_reader)

            x_data[:, i] = np.array(x_row[1:], dtype=float)
            y_data[:, i] = np.array(y_row[1:], dtype=float)
            phi_data[:, i] = np.array(phi_row[1:], dtype=float)

    # Use the first iteration for plotting
    x_plot = x_data[:, 0] / sigma
    y_plot = y_data[:, 0] / sigma
    phi_plot = phi_data[:, 0]

    # Arrow components (direction based on angles)
    ux_arrows = np.cos(phi_plot)
    uy_arrows = np.sin(phi_plot)

    # Define limits dynamically with margins
    margin = 1.0  # Adjust as needed
    x_min, x_max = np.min(x_plot) - margin, np.max(x_plot) + margin
    y_min, y_max = np.min(y_plot) - margin, np.max(y_plot) + margin

    # Create plot
    plt.figure(figsize=(10, 8))

    # Add arrows with quiver
    arrow_scale = arrow_scale if arrow_scale is not None else 1.0

    # Generate colors for quiver using viridis
    norm = Normalize(vmin=0, vmax=N - 1)
    colors = get_cmap('viridis')(norm(range(N)))

    for i in range(N):
        plt.quiver(
            x_plot[i:i+1], y_plot[i:i+1],  # Arrow positions
            ux_arrows[i:i+1], uy_arrows[i:i+1],  # Arrow directions
            angles='xy', scale_units='xy', scale=arrow_scale, width=0.005, alpha=1,
            color=colors[i]
        )

    # Desired radius for points
    radius = 0.5  # Graphic units

    # Adjust marker scale
    for i in range(N):
        circle = plt.Circle((x_plot[i], y_plot[i]), radius=radius, color='blue', alpha=0.3, label='Initial positions' if i == 0 else "")
        plt.gca().add_patch(circle)

    # Adjust limits and axes
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis('equal')

    # Create title with simulation information
    plt.title(
        f'd₀ = {separation}σ',
        fontsize=25
    )

    plt.xlabel('x (σ)', fontsize=20)
    plt.ylabel('y (σ)', fontsize=20)
    plt.tick_params(axis='x', labelsize=18)  # Change x-axis label size
    plt.tick_params(axis='y', labelsize=18)  # Change y-axis label size

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return

def solve_equations(
        N, steps, iterations, separation, sigma, 
        B, mm, v, dt, mu, eta, T, 
        brownian_motion, initial_conditions_file, batch_size
    ):
    """
    Solves the Langevin equations for the particles given the initial conditions.
    
    Optimizations include:
    - Using Numba for critical computation loops
    - Vectorized operations where possible
    - Reduced memory allocation
    - Optimized force calculations
    - Better batch processing
    
    Parameters:
        N (int): Number of particles.
        steps (int): Number of time steps.
        iterations (int): Number of repetitions.
        separation (float): Separation between particles in a grid configuration.
        sigma (float): Characteristic size of the particles.
        B (float): Magnetic field.
        mm (float): Magnetic moment.
        v (float): Particle velocity.
        dt (float): Time step.
        mu (float): Viscosity of the fluid.
        eta (float): Dynamic viscosity of the fluid.
        T (float): Temperature of the fluid.
        brownian_motion (bool): Whether to include Brownian motion.
        initial_conditions_file (str): Name of the CSV file with the initial conditions.
        batch_size (int): Number of steps to save in each batch.
    """
    # Validate inputs
    if batch_size > steps:
        raise ValueError("batch_size cannot be greater than the number of steps (steps).")

    # Load initial conditions
    initial_data = pd.read_csv(initial_conditions_file)
    x = np.zeros((steps + 1, N))
    y = np.zeros((steps + 1, N))
    phi = np.zeros((steps + 1, N))
    
    x[0] = initial_data.iloc[:, 1].values
    y[0] = initial_data.iloc[:, 2].values
    phi[0] = initial_data.iloc[:, 3].values

    # Constants
    k = 1.38e-23  # Boltzmann constant
    K0 = k * T
    Dt = K0 / (3 * np.pi * eta * sigma)  # Translational diffusion coefficient
    Dr = K0 / (np.pi * eta * sigma**3)  # Rotational diffusion coefficient
    gammar = K0 / Dr  # Rotational friction coefficient
    gammat = K0 / Dt  # Translational friction coefficient
    KF = -(3 * mu * mm**2) / (4 * np.pi)  # Force coefficient
    KTAU = (mu * mm**2) / (4 * np.pi)  # Torque coefficient
    rc = 2 ** (1/6) * sigma  # Cut-off radius
    epsilon = (mu * mm**2 / (4 * np.pi * sigma**3)) * 1  # Potential well depth
    
    # Precompute square roots for Brownian motion
    sqrt_2_Dt_dt = np.sqrt(2 * Dt * dt)
    sqrt_2_Dr_dt = np.sqrt(2 * Dr * dt)
    brown = 1 if brownian_motion else 0
    
    # Create output directories once
    general_name = f'{N}part_{steps//1000}ksteps_{iterations}its_{separation}sep'
    x_dir_name = f'x_{general_name}'
    y_dir_name = f'y_{general_name}'
    phi_dir_name = f'phi_{general_name}'
    
    os.makedirs(x_dir_name, exist_ok=True)
    os.makedirs(y_dir_name, exist_ok=True)
    os.makedirs(phi_dir_name, exist_ok=True)

    for i in range(iterations):
        # Initialize arrays for current iteration
        current_x = x[0].copy()
        current_y = y[0].copy()
        current_phi = phi[0].copy()
        
        # Prepare output files
        B_mT = B * 1000
        x_filename = os.path.join(x_dir_name, f'x_{i+1}it_{B_mT:.1f}mT.csv')
        y_filename = os.path.join(y_dir_name, f'y_{i+1}it_{B_mT:.1f}mT.csv')
        phi_filename = os.path.join(phi_dir_name, f'phi_{i+1}it_{B_mT:.1f}mT.csv')
        
        # Initialize DataFrames
        columns = ['time'] + [f'particle_{j+1}' for j in range(N)]
        x_df = pd.DataFrame(columns=columns)
        y_df = pd.DataFrame(columns=columns)
        phi_df = pd.DataFrame(columns=columns)
        
        # Write initial conditions
        x_df.loc[0] = [0] + list(current_x)
        y_df.loc[0] = [0] + list(current_y)
        phi_df.loc[0] = [0] + list(current_phi)
        
        # Batch processing
        total_batches = (steps + batch_size - 1) // batch_size
        current_batch = 0
        
        for n in range(1, steps + 1):
            # Calculate forces and torques
            F12x, F12y, tau12z = calculate_forces_torques(
                current_x, current_y, current_phi, 
                N, rc, sigma, epsilon, KF, KTAU
            )
            
            # Update positions and orientations
            noise_x = sqrt_2_Dt_dt * brown * np.random.randn(N)
            noise_y = sqrt_2_Dt_dt * brown * np.random.randn(N)
            noise_phi = sqrt_2_Dr_dt * brown * np.random.randn(N)
            
            current_x = (
                current_x 
                + v * np.cos(current_phi) * dt 
                + F12x.sum(axis=1) * dt / gammat 
                + noise_x
            )
            current_y = (
                current_y 
                + v * np.sin(current_phi) * dt 
                + F12y.sum(axis=1) * dt / gammat 
                + noise_y
            )
            current_phi = (
                current_phi 
                + (mm * B * np.cos(current_phi) + tau12z.sum(axis=1)) / gammar * dt 
                + noise_phi
            )
            
            # Save to DataFrames
            time = n * dt
            x_df.loc[n] = [time] + list(current_x)
            y_df.loc[n] = [time] + list(current_y)
            phi_df.loc[n] = [time] + list(current_phi)
            
            # Batch processing
            if n % batch_size == 0 or n == steps:
                # Write to CSV
                write_mode = 'w' if current_batch == 0 else 'a'
                header = current_batch == 0
                
                x_df.to_csv(x_filename, mode=write_mode, header=header, index=False)
                y_df.to_csv(y_filename, mode=write_mode, header=header, index=False)
                phi_df.to_csv(phi_filename, mode=write_mode, header=header, index=False)
                
                # Reset DataFrames
                x_df = pd.DataFrame(columns=columns)
                y_df = pd.DataFrame(columns=columns)
                phi_df = pd.DataFrame(columns=columns)
                
                current_batch += 1
                print(f"Iteration {i+1}/{iterations}: Batch {current_batch}/{total_batches}", end="\r")
        
        print(f"Iteration {i+1}/{iterations}: Completed {' ' * 20}")
    
    print(f'Finished B={B*1000} mT. {iterations} iterations completed.')
    return

@njit
def calculate_forces_torques(x, y, phi, N, rc, sigma, epsilon, KF, KTAU):
    """Calculate forces and torques between all particles using Numba for speed."""
    F12x = np.zeros((N, N))
    F12y = np.zeros((N, N))
    tau12z = np.zeros((N, N))
    
    for p in range(N):
        for q in range(p + 1, N):
            r12x = x[q] - x[p]
            r12y = y[q] - y[p]
            r12_sq = r12x**2 + r12y**2
            r12 = np.sqrt(r12_sq)
            
            cos_phi_p = np.cos(phi[p])
            sin_phi_p = np.sin(phi[p])
            cos_phi_q = np.cos(phi[q])
            sin_phi_q = np.sin(phi[q])
            
            u12 = np.cos(phi[p] - phi[q])
            u1r = (r12x * cos_phi_p + r12y * sin_phi_p) / r12
            u2r = (r12x * cos_phi_q + r12y * sin_phi_q) / r12
            
            if r12 <= rc:
                # Lennard-Jones potential
                sigma_r = sigma / r12
                sigma_r6 = sigma_r**6
                factor = (24 * epsilon / r12_sq) * (2 * sigma_r6**2 - sigma_r6)
                F12x[p, q] = -factor * r12x
                F12y[p, q] = -factor * r12y
            else:
                # Dipole-dipole interaction
                r12_4 = r12_sq**2
                term = (u12 - 5 * u1r * u2r) / r12
                F12x[p, q] = (KF / r12_4) * (term * r12x + u2r * cos_phi_p + u1r * cos_phi_q)
                F12y[p, q] = (KF / r12_4) * (term * r12y + u2r * sin_phi_p + u1r * sin_phi_q)
            
            # Torque calculation
            tau12z[p, q] = (KTAU / (r12**3)) * (
                sin_phi_p * (cos_phi_q - 3 * u2r * r12x / r12) - 
                cos_phi_p * (sin_phi_q - 3 * u2r * r12y / r12)
            )
            
            # Newton's third law
            F12x[q, p] = -F12x[p, q]
            F12y[q, p] = -F12y[p, q]
            tau12z[q, p] = -tau12z[p, q]
    
    return F12x, F12y, tau12z


def plot_trajectories(N, steps, iterations, separation, dt, sigma, 
                      v, B, mm, T, brownian_motion, arrow_scale=None):
    """
    Plots the trajectories and orientations of particles from generated CSV files.

    Parameters:
        N: Number of particles.
        steps: Number of time steps.
        iterations: Number of repetitions.
        separation: Separation between particles in a grid configuration.
        dt: Time step.
        sigma: Characteristic size of the particles.
        v: Particle velocity.
        B: Magnetic field.
        mm: Magnetic moment.
        T: Temperature.
        brownian_motion (bool): Whether to include Brownian motion.
        arrow_scale: Scale of the arrows (optional).
    """

    general_name = f'{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'

    x_dir_name = f'x_{general_name}'  # x directory name
    y_dir_name = f'y_{general_name}'  # y directory name
    phi_dir_name = f'phi_{general_name}'  # phi directory name

    iterations = 1  # For testing purposes, set iterations to 1 
    # Load trajectory data from CSV files
    x_data = np.zeros((steps + 1, N, iterations))
    y_data = np.zeros((steps + 1, N, iterations))
    phi_data = np.zeros((steps + 1, N, iterations))

    # For testing purposes, set iterations to 1
    for i in [0]:  # instead of range(iterations)
        x_filename = os.path.join(x_dir_name,
                                  f'x_{i+1}it_{B*1000:.1f}mT.csv')
        y_filename = os.path.join(y_dir_name,
                                  f'y_{i+1}it_{B*1000:.1f}mT.csv')
        phi_filename = os.path.join(phi_dir_name,
                                    f'phi_{i+1}it_{B*1000:.1f}mT.csv')

        # Read CSV files
        with open(x_filename, 'r', encoding='utf-8') as x_file, \
             open(y_filename, 'r', encoding='utf-8') as y_file, \
             open(phi_filename, 'r', encoding='utf-8') as phi_file:
            
            x_reader = csv.reader(x_file)
            y_reader = csv.reader(y_file)
            phi_reader = csv.reader(phi_file)

            # Skip the first row (headers)
            next(x_reader)
            next(y_reader)
            next(phi_reader)

            for n, (x_row, y_row, phi_row) in enumerate(zip(x_reader, y_reader, phi_reader)):
                x_data[n, :, i] = np.array(x_row[1:], dtype=float)
                y_data[n, :, i] = np.array(y_row[1:], dtype=float)
                phi_data[n, :, i] = np.array(phi_row[1:], dtype=float)

    # Calculate average trajectories
    x_plot = np.mean(x_data / sigma , axis=2)  # Average over iterations
    y_plot = np.mean(y_data / sigma, axis=2)

    # Calculate average orientation (direction of arrows)
    phi_plot = np.mean(phi_data, axis=2)  # Average angles

    # Sampling: Select 8% of the points along the trajectories
    num_points = len(x_plot)
    step = max(1, num_points // 4)  # Ensure not to divide by 0
    indices = np.arange(0, num_points, step)

    # Arrow coordinates
    x_arrows = x_plot[indices, :]
    y_arrows = y_plot[indices, :]

    # Arrow components (direction based on angles)
    ux_arrows = np.cos(phi_plot[indices, :])
    uy_arrows = np.sin(phi_plot[indices, :])

    # Define limits dynamically with margins
    margin = 10000 * steps * v / sigma * dt
    x_min, x_max = np.min(x_plot) - margin, np.max(x_plot) + margin
    y_min, y_max = np.min(y_plot) - margin, np.max(y_plot) + margin

    # Create plot
    plt.figure(figsize=(10, 10))

    # Plot average trajectory
    plt.plot(x_plot, y_plot, linewidth=1, alpha=0.1)

    # Add arrows with quiver
    arrow_scale = arrow_scale if arrow_scale is not None else (y_max - y_min) / (x_max - x_min) * 5

    # Generate colors for quiver using viridis
    norm = Normalize(vmin=0, vmax=N - 1)
    colors = get_cmap('viridis')(norm(range(N)))
    colors[0] = colors[-1] = (0, 0, 0, 1)  # Set first and last group to black

    for i in range(N):
        plt.quiver(
            x_arrows[:, i], y_arrows[:, i],  # Arrow positions
            ux_arrows[:, i], uy_arrows[:, i],  # Arrow directions
            angles='xy', scale_units='xy', scale=arrow_scale, width=0.003, alpha=0.9,
            color=colors[i]
        )

    # Desired radius for points
    radius = 0.5  # Graphic units

    # Adjust marker scale
    for i in range(len(x_plot[0, :])):
        circle = plt.Circle((x_plot[0, i], y_plot[0, i]), radius=radius, color='blue', 
                            alpha=0.3, label='Initial positions' if i == 0 else "")
        plt.gca().add_patch(circle)

    for i in range(len(x_plot[steps, :])):
        circle = plt.Circle((x_plot[steps, i], y_plot[steps, i]), radius=radius, color='blue', 
                            alpha=0.7, label='Final positions' if i == 0 else "")
        plt.gca().add_patch(circle)

    # Adjust limits and axes
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axis('equal')

    # Create title with simulation information
    plt.title(
        f'd₀ = {separation}σ, B = {B*1000:.1f} mT',
        fontsize=25
    )

    plt.xlabel('x (σ)', fontsize=20)
    plt.ylabel('y (σ)', fontsize=20)
    plt.tick_params(axis='x', labelsize=18)  # Change x-axis label size
    plt.tick_params(axis='y', labelsize=18)  # Change y-axis label size

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=15, loc='upper right')
    

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return


def gather_displacement_data(B_values, steps, iterations, separation_values, sigma, N):
    """
    Gathers displacement data for multiple separations and magnetic field values.

    Parameters:
        B_values: List of B values (magnetic field).
        steps: Number of time steps in the simulation.
        iterations: Number of repetitions of the simulation.
        separation_values: List of separations between particles in a grid configuration.
        sigma: Characteristic size of the particles (for normalization).
        N: Number of particles.

    Returns:
        A pandas DataFrame containing the aggregated data.
    """
    data = []

    for separation in separation_values:
        general_name = f'{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'
        y_dir_name = f'y_{general_name}'

        for B in B_values:
            y_initial = np.zeros((iterations, N))
            y_final = np.zeros((iterations, N))

            for i in range(iterations):
                y_filename = os.path.join(y_dir_name, f'y_{i+1}it_{B*1000:.1f}mT.csv')

                # Use pandas to read only the first and last rows of the CSV file
                y_data_first = pd.read_csv(y_filename, nrows=1)  # Read the first row
                y_data_last = pd.read_csv(y_filename, skiprows=range(1, steps - 1))  # Read the last row

                # Extract the first and last rows
                y_initial_row = y_data_first.iloc[0, 1:].values  # Skip the first column (assumed to be an index or time)
                y_last_row = y_data_last.iloc[0, 1:].values

                del y_data_first, y_data_last

                y_initial[i, :] = y_initial_row.astype(float)
                y_final[i, :] = y_last_row.astype(float)

            # Calculate displacements
            y_displacements = (y_final - y_initial) / sigma
            mean_y_displacement = np.mean(y_displacements)
            std_y_displacement = np.std(y_displacements)

            # Append data to the list
            data.append({
                'B (mT)': B * 1000,
                'mean y displacement (sigma)': mean_y_displacement,
                'std y displacement (sigma)': std_y_displacement,
                'separation (sigma)': separation
            })

    # Convert the list of dictionaries to a DataFrame
    displacement_data = pd.DataFrame(data)
    displacement_data.to_csv('aggregated_displacements.csv', index=False)
    print('Saved aggregated data to aggregated_displacements.csv')

    return displacement_data


def plot_displacement_data(v, sigma, dipole_moment, figsize=(7,6)):
    """
    Plots the displacement data.

    Parameters:
        v: Particle velocity.
        sigma: Characteristic size of the particles (for normalization).
        dipole_moment: Dipole moment of the particles.
    """
    displacement_data = pd.read_csv('aggregated_displacements.csv')

    # Multiply std y displacement by 2 for better visibility
    displacement_data['std y displacement (sigma)'] *= 2

    plt.figure(figsize=figsize)
    fontsize = 18
    elinewidth = 3
    marker_size = 12

    # Convert 'B (mT)' to a categorical variable (string)
    displacement_data['B (mT)'] = displacement_data['B (mT)'].astype(str)

    for separation, group in displacement_data.groupby('separation (sigma)'):
        plt.errorbar(group['B (mT)'], group['mean y displacement (sigma)'], 
                     yerr=group['std y displacement (sigma)'], fmt='o', capsize=7,
                     markersize=marker_size, elinewidth=elinewidth, label=f'd₀ = {separation}σ')

    plt.xlabel('B (mT)', fontsize=fontsize+4)
    plt.ylabel('$\\langle \\Delta y\\rangle$', fontsize=fontsize+4)
    if dipole_moment == '1':
        #plt.title(f'v = {v/sigma:.0f}σ/s, m = {dipole_moment} m₀', fontsize=fontsize)
        plt.title(f'm = {dipole_moment} m₀', fontsize=fontsize+4)
    elif dipole_moment == 'sqrt10':
        #plt.title(f'v = {v/sigma:.0f}σ/s, m = $\\sqrt{{10}}$ m₀', fontsize=fontsize)
        plt.title('m = $\\sqrt{{10}}$ m₀', fontsize=fontsize+4)
    plt.ylim(-60, 110)
    plt.grid()
    plt.legend(fontsize=fontsize+4)
    plt.xticks(fontsize=fontsize+4, rotation=45)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=fontsize+4)
    plt.show()


def gather_final_angles(B_values, steps, iterations, separation_values, N):
    """
    Gathers final angle data for multiple separations and magnetic field values.

    Parameters:
        B_values: List of B values (magnetic field).
        steps: Number of time steps in the simulation.
        iterations: Number of repetitions of the simulation.
        separation_values: List of separations between particles in a grid configuration.
        N: Number of particles.

    Returns:
        A pandas DataFrame containing the aggregated angle data.
    """
    data = []

    for separation in separation_values:
        general_name = f'{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'
        phi_dir_name = f'phi_{general_name}'

        for B in B_values:
            phi_final = np.zeros((iterations, N))

            for i in range(iterations):
                phi_filename = os.path.join(phi_dir_name, f'phi_{i+1}it_{B*1000:.1f}mT.csv')
                
                phi_data = pd.read_csv(phi_filename, skiprows=range(1, steps - 1))  # Read last row
                phi_last_row = phi_data.iloc[0, 1:].values  # Skip first column
                
                phi_final[i, :] = np.mod(phi_last_row.astype(float), 2 * np.pi)

            # Calculate statistics
            mean_phi = np.mean(phi_final)
            std_phi = np.std(phi_final)

            data.append({
                'B (mT)': B * 1000,
                'mean phi': mean_phi,
                'std phi': std_phi,
                'separation (sigma)': separation
            })

    # Convert to DataFrame
    angle_data = pd.DataFrame(data)
    angle_data.to_csv('aggregated_final_angles.csv', index=False)
    print('Saved aggregated data to aggregated_final_angles.csv')

    return angle_data


def plot_final_angles(dipole_moment, figsize=(6,5)):
    """
    Plots the final angle data.

    Parameters:
        dipole_moment: Dipole moment of the particles.
    """
    # Load the final angle data
    angle_data = pd.read_csv('aggregated_final_angles.csv')

    plt.figure(figsize=figsize)
    fontsize = 18
    elinewidth = 3
    marker_size = 12

    # Convert 'B (mT)' to a categorical variable (string)
    angle_data['B (mT)'] = angle_data['B (mT)'].astype(str)
    # Multiply std phi by 2 for better visibility
    angle_data['std phi'] *= 2

    for separation, group in angle_data.groupby('separation (sigma)'):
        plt.errorbar(group['B (mT)'], group['mean phi'], 
                     yerr=group['std phi'], fmt='o', capsize=7,
                     markersize=marker_size, elinewidth=elinewidth, label=f'd₀ = {separation}σ', alpha=0.8)
    
    # Horizontal line at pi/2
    plt.axhline(y=np.pi/2, color='k', linestyle='dashed')

    # Horizontal line at pi
    plt.axhline(y=np.pi, color='r', linestyle='dashed')

    plt.xlabel('B (mT)', fontsize=fontsize+4)
    plt.ylabel('$\\langle\\phi_P\\rangle$', fontsize=fontsize+4)
    plt.grid()
    if dipole_moment == '1':
        #plt.title(f'v = {v/sigma:.0f}σ/s, m = {dipole_moment} m₀', fontsize=fontsize)
        plt.title(f'm = 1 m₀', fontsize=fontsize+4)
    elif dipole_moment == 'sqrt10':
        #plt.title(f'v = {v/sigma:.0f}σ/s, m = $\\sqrt{{10}}$ m₀', fontsize=fontsize)
        plt.title(f'm = $\\sqrt{{10}}$ m₀', fontsize=fontsize+4)
    #plt.legend(fontsize=fontsize+4)
    plt.xticks(fontsize=fontsize+4, rotation=45)
    # y ticks in radians
    plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2], ['0', '$π/2$', '$π$', '$3π/2$'], fontsize=fontsize+4)
    plt.legend(fontsize=fontsize+4)
    plt.show()

@njit(parallel=True)
def calculate_msd_numba(pos_x, pos_y, steps, steps_to_skip):
    """
    Numba-optimized core MSD calculation function.
    """
    n_points = steps // steps_to_skip
    avg_msd = np.zeros(n_points)
    
    for n in prange(n_points):
        selected_steps = np.arange(0, steps, steps_to_skip)
        valid_range = np.arange(0, len(selected_steps) - n)
        valid_range = valid_range[valid_range >= 0]  # Only positive indices
        
        if len(valid_range) == 0:
            continue
            
        # Vectorized displacement calculation
        delta_x2 = (pos_x[:, :, selected_steps[valid_range + n]] - pos_x[:, :, selected_steps[valid_range]]) ** 2
        delta_y2 = (pos_y[:, :, selected_steps[valid_range + n]] - pos_y[:, :, selected_steps[valid_range]]) ** 2
        
        # Sum all displacements
        avg_msd[n] = (np.sum(delta_x2) + np.sum(delta_y2)) / (pos_x.shape[0] * len(valid_range) * pos_x.shape[1])
    
    return avg_msd

def calculate_MSD(N, steps, iterations, separation, 
                 sigma, dt, B_values, v, eta, T, 
                 type_msd, skip_fraction=0):
    """
    Optimized MSD calculation using Numba for core computations.
    
    Key optimizations:
    - Numba-accelerated MSD calculation
    - Parallel processing of time lags
    - Reduced memory allocation
    - Vectorized operations where possible
    """
    # Create directory to save the results
    results_dir = f'msd_{N}part_{steps//1000}ksteps_{iterations}its_{separation}sep'
    os.makedirs(results_dir, exist_ok=True)

    # Determine how many steps to skip
    steps_to_skip = max(1, int(steps * skip_fraction))
    time_values = np.arange(0, steps, steps_to_skip) * dt
    msd_data = {"time": time_values}

    # Calculate diffusion coefficients
    k = 1.38e-23  # Boltzmann constant
    K0 = k * T
    Dt = K0 / (3 * np.pi * eta * sigma)
    Dr = K0 / (np.pi * eta * sigma**3)

    # Calculate theoretical MSD
    if type_msd == 'tra':
        if v is None or Dr is None or Dt is None:
            raise ValueError("For translational MSD, 'v', 'Dr', and 'Dt' must be provided.")
        msd_data[f'{type_msd}_msd_smoluchowski'] = (
            (4 * (v ** 2 / (2 * Dr) + Dt) * time_values +
            ((2 * v ** 2) / (Dr ** 2)) * (np.exp(-Dr * time_values) - 1))) / (sigma ** 2)
    elif type_msd == 'rot':
        if Dr is None:
            raise ValueError("For rotational MSD, 'Dr' must be provided.")
        msd_data[f'{type_msd}_msd_smoluchowski'] = 2 * (1 - np.exp(-Dr * time_values))

    general_name = f'{N}part_{steps//1000}ksteps_{iterations}its_{separation}sep'
    
    for B in B_values:
        print(f"Processing B = {B*1000:.1f} mT...")
        
        # Pre-allocate arrays
        if type_msd == 'tra':
            pos_x = np.zeros((iterations, N, steps))
            pos_y = np.zeros((iterations, N, steps))
            
            # Load data in bulk
            for i in range(iterations):
                x_filename = os.path.join(f'x_{general_name}', f'x_{i+1}it_{B*1000:.1f}mT.csv')
                y_filename = os.path.join(f'y_{general_name}', f'y_{i+1}it_{B*1000:.1f}mT.csv')
                
                x_data = pd.read_csv(x_filename, header=0).iloc[:, 1:].values.T / sigma
                y_data = pd.read_csv(y_filename, header=0).iloc[:, 1:].values.T / sigma
                
                pos_x[i, :, :x_data.shape[1]] = x_data[:, :steps]
                pos_y[i, :, :y_data.shape[1]] = y_data[:, :steps]
                
        elif type_msd == 'rot':
            pos_phi = np.zeros((iterations, N, steps))
            
            for i in range(iterations):
                phi_filename = os.path.join(f'phi_{general_name}', f'phi_{i+1}it_{B*1000:.1f}mT.csv')
                phi_data = pd.read_csv(phi_filename, header=0).iloc[:, 1:].values.T
                pos_phi[i, :, :phi_data.shape[1]] = phi_data[:, :steps]
            
            # Convert to coordinates
            pos_x = np.cos(pos_phi)
            pos_y = np.sin(pos_phi)

        # Calculate MSD using Numba-optimized function
        avg_msd = calculate_msd_numba(pos_x, pos_y, steps, steps_to_skip)
        msd_data[f'{type_msd}_msd_{B*1000:.1f}mT'] = avg_msd

    # Save results
    msd_filename = os.path.join(results_dir, f'{type_msd}_msd_all_B.csv')
    pd.DataFrame(msd_data).to_csv(msd_filename, index=False)
    print(f"MSD data saved to {msd_filename}")

    return


def plot_MSD(N, steps, iterations, separation, B_values, sigma, v, T, eta,
             type_msd, loglog=True, start=0, end=None, figsize=(7,7), dipole_moment=None, label=True, legend=True):
    """
    Plots the Mean Squared Displacement (MSD) for a system of micro-swimmers with a dynamic header.

    Args:
        N (int): Number of particles.
        steps (int): Number of steps in the simulation.
        iterations (int): Number of simulation repetitions.
        B_values (list): List of magnetic field values.
        sigma (float): Diameter of the particles (in meters).
        v (float): Velocity of the particles (in meters/second).
        dt (float): Time step size (in seconds).
        T (float): Temperature in Kelvin.
        mm (float): Magnetic moment of the particles (in Am²).
        eta (float): Solvent viscosity (in Pa·s).
        type_msd (str): Type of MSD ('tra' for translational, 'rot' for rotational).
        loglog (bool): If True, uses a log-log scale for the plot.
        start (int): Initial step for plotting.
        end (int): Final step for plotting. If None, all steps are plotted.
        brown (bool): Indicates whether to include the Brownian motion term (True) or not (False).
        figsize (tuple): Size of the figure.
        dipole_moment (str): Title for dipole moment of the particles.
        label (bool): Indicates whether to include labels in the plot.
    """
    plt.figure(figsize=figsize)
    fontsize = 22-.5
    linewidth = 4

    # Directory where the results are stored
    results_dir = f'msd_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'

    # Read the CSV file containing MSD data
    msd_filename = os.path.join(results_dir, f'{type_msd}_msd_all_B.csv')
    df = pd.read_csv(msd_filename)

    # Use the 'viridis' colormap
    colors = get_cmap('viridis')(np.linspace(0, 1, len(B_values)))

    # Calculate diffusion coefficients
    k = 1.38e-23  # Boltzmann constant (in J/K)
    K0 = k * T  # Thermal energy
    Dt = K0 / (3 * np.pi * eta * sigma)  # Translational diffusion coefficient
    Dr = K0 / (np.pi * eta * sigma**3)  # Rotational diffusion coefficient

    # Calculate theoretical MSD based on the Smoluchowski model
    if type_msd == 'tra':
        df[f'{type_msd}_msd_smoluchowski'] = (
            (4 * (v ** 2 / (2 * Dr) + Dt) * df['time'] + 
            ((2 * v ** 2) / (Dr ** 2)) * (np.exp(- Dr * df['time']) - 1)) / (sigma ** 2)
        )
    elif type_msd == 'rot':
        df[f'{type_msd}_msd_smoluchowski'] = 2 * (1 - np.exp(- Dr * df['time']))

    # Plot the theoretical MSD
    if end is None:
        end = len(df) - 1
    plt.plot(df["time"][start:end+1] * Dr, df[f"{type_msd}_msd_smoluchowski"][start:end+1], 
             label='No interactions', color='red', linestyle='-', linewidth=linewidth)
    
    # Plot each MSD curve
    for B, color in zip(B_values, colors):
        if end is None:
            end = len(df) - 1
        plt.plot(df["time"][start:end+1] * Dr, df[f'{type_msd}_msd_{B/1e-3:.1f}mT'][start:end+1], 
                 label=f'{B/1e-3:.1f}', color=color, linewidth=linewidth+2, linestyle='dotted')

    if label:
        # Create title with simulation information
        plt.title(label=f'd₀ = {separation}σ', fontsize=fontsize)
        plt.xlabel('$\\tau$', fontsize=fontsize)
        plt.ylabel(f'm = {dipole_moment}m$_0$\nWₜ / σ²' if type_msd == 'tra' else f'm = {dipole_moment}m$_0$\nWᵣ', fontsize=fontsize)
        if legend:
            if type_msd == 'tra':
                loc = 'upper left'
            elif type_msd == 'rot':
                loc = 'upper left'
            plt.legend(fontsize=fontsize-7, title='B (mT)', title_fontsize=fontsize-7, loc=loc)

    plt.tick_params(axis='x', labelsize=fontsize)  # Change x-axis label size
    plt.tick_params(axis='y', labelsize=fontsize)  # Change y-axis label size

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)

    # Use log-log scale if specified
    if loglog:
        plt.xscale('log')
        plt.yscale('log')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def generate_correlation_data(B_values, N, steps, iterations, separation_values, lags, mode='acf'):
    """
    Genera los datos de correlación (ACF o PACF) para diferentes valores de separación y los guarda en archivos CSV.

    Parameters:
        B_values: Lista de valores de B (campo magnético).
        N: Número de partículas.
        steps: Número de pasos de tiempo.
        iterations: Número de repeticiones.
        separation_values: Lista de separaciones entre partículas.
        lags: Número de lags para la correlación.
        mode: Tipo de correlación ('acf' o 'pacf').
    """
    if mode not in ['acf', 'pacf']:
        raise ValueError("El parámetro 'mode' debe ser 'acf' o 'pacf'.")

    for separation in separation_values:
        for B in B_values:
            # Inicializar arrays para almacenar los datos
            x_data = np.zeros((steps + 1, N, iterations))
            y_data = np.zeros((steps + 1, N, iterations))
            phi_data = np.zeros((steps + 1, N, iterations))

            general_name = f'{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'
            x_dir_name = f'x_{general_name}'
            y_dir_name = f'y_{general_name}'
            phi_dir_name = f'phi_{general_name}'

            for i in range(iterations):
                x_filename = os.path.join(x_dir_name, f'x_{i+1}it_{B*1000:.1f}mT.csv')
                y_filename = os.path.join(y_dir_name, f'y_{i+1}it_{B*1000:.1f}mT.csv')
                phi_filename = os.path.join(phi_dir_name, f'phi_{i+1}it_{B*1000:.1f}mT.csv')

                if not (os.path.exists(x_filename) and os.path.exists(y_filename) and os.path.exists(phi_filename)):
                    raise FileNotFoundError(f"Archivos CSV no encontrados para B = {B} T y separación = {separation}")

                with open(x_filename, 'r') as x_file, open(y_filename, 'r') as y_file, open(phi_filename, 'r') as phi_file:
                    x_reader = csv.reader(x_file)
                    y_reader = csv.reader(y_file)
                    phi_reader = csv.reader(phi_file)

                    next(x_reader)
                    next(y_reader)
                    next(phi_reader)

                    for n, (x_row, y_row, phi_row) in enumerate(zip(x_reader, y_reader, phi_reader)):
                        x_data[n, :, i] = np.array(x_row[1:], dtype=float)
                        y_data[n, :, i] = np.array(y_row[1:], dtype=float)
                        phi_data[n, :, i] = np.array(phi_row[1:], dtype=float)

            x_mean = np.mean(x_data, axis=(1, 2))
            y_mean = np.mean(y_data, axis=(1, 2))
            phi_mean = np.mean(phi_data, axis=(1, 2))

            if mode == 'acf':
                x_corr = acf(x_mean, nlags=lags)
                y_corr = acf(y_mean, nlags=lags)
                phi_corr = acf(phi_mean, nlags=lags)
            else:
                x_corr = pacf(x_mean, nlags=lags)
                y_corr = pacf(y_mean, nlags=lags)
                phi_corr = pacf(phi_mean, nlags=lags)

            # Guardar los datos de correlación en archivos CSV
            output_dir = f'correlation_data_sep_{separation}'
            os.makedirs(output_dir, exist_ok=True)
            np.savetxt(os.path.join(output_dir, f'x_corr_B{B}.csv'), x_corr, delimiter=',')
            np.savetxt(os.path.join(output_dir, f'y_corr_B{B}.csv'), y_corr, delimiter=',')
            np.savetxt(os.path.join(output_dir, f'phi_corr_B{B}.csv'), phi_corr, delimiter=',')


def plot_correlation_comparison(B_values, separation_values, lags, mode='acf'):
    """
    Reads the generated correlation data and creates comparative plots for different separations.

    Parameters:
        B_values: List of B values (magnetic field).
        separation_values: List of separations between particles.
        lags: Number of lags for the correlation.
        mode: Type of correlation ('acf' or 'pacf').
    """
    fontsize = 20
    colors = get_cmap('viridis')(np.linspace(0, 1, len(B_values)))

    for coord, coord_label in zip(['x', 'y', 'phi'], ['x', 'y', 'φ']):
        fig, axes = plt.subplots(1, len(separation_values), figsize=(10, 5), sharey=True)

        for col, separation in enumerate(separation_values):
            for B, color in zip(B_values, colors):
                input_dir = f'correlation_data_sep_{separation}'
                corr_data = np.loadtxt(os.path.join(input_dir, f'{coord}_corr_B{B}.csv'), delimiter=',')

                axes[col].plot(corr_data, label=f'B = {B*1000:.1f} mT', color=color)

            axes[col].set_title(f'd₀ = {separation}σ', fontsize=fontsize)
            axes[col].set_xlabel('Lag', fontsize=fontsize-2)
            axes[col].tick_params(axis='both', labelsize=fontsize-8)
            
            if col == 0:
                axes[col].set_ylabel(coord_label, fontsize=fontsize)

            if col == len(separation_values) - 1:  # Add legend only in the last column
                axes[col].legend(title='Magnetic Field', fontsize=fontsize-9, title_fontsize=fontsize-9)

            axes[col].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


import os
import csv
import numpy as np
import pandas as pd

def calculate_structure_factor(N, steps, iterations, separation, sigma, B_values, 
                               n_step, v, num_wavevector_steps=250, batch_size=100, chunk_size=50):
    """
    Calculates the structure factor for a system of particles under different magnetic fields,
    processing wavevectors in chunks to reduce memory usage.

    Args:
        N (int): Number of particles.
        steps (int): Number of time steps in the simulation.
        iterations (int): Number of repetitions of the simulation.
        separation (float): Separation between particles in the initial configuration.
        sigma (float): Diameter of the particles.
        B_values (numpy.ndarray): Array of magnetic field values.
        n_step (int): Specific time step to calculate the structure factor.
        v (float): Velocity of the particles.
        num_wavevector_steps (int): Number of wavevector steps to scan. Default is 250.
        batch_size (int): Number of iterations to process in each batch. Default is 100.
        chunk_size (int): Number of wavevectors to process in each chunk. Default is 50.

    Returns:
        None
    """
    # Create directory to save results
    results_dir = f'struct_factor_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'
    os.makedirs(results_dir, exist_ok=True)

    # Define angles and wavevectors to scan
    num_angle_steps = num_wavevector_steps
    num_wave_steps = num_wavevector_steps
    angle = np.linspace(0, 2 * np.pi, num_angle_steps, endpoint=False)
    wave_magnitude = np.linspace(2 * np.pi / num_wave_steps, 2 * np.pi * 2, num_wave_steps + 1)
    
    # Create arrays for x and y components of wavevectors
    qx = wave_magnitude[:, None] * np.cos(angle)
    qy = wave_magnitude[:, None] * np.sin(angle)
    
    # Initialize dictionary to store structure factors
    structure_factors = {"q_sigma": wave_magnitude}

    # Calculate structure factor for each magnetic field value
    for B in B_values:
        Sqprom_total = np.zeros((num_wave_steps + 1, num_angle_steps))
        num_batches = (iterations + batch_size - 1) // batch_size  # Round up
        
        # Process iterations in batches to reduce memory usage
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, iterations)
            batch_iterations = batch_end - batch_start
            
            # Initialize arrays to store particle positions
            pos_x = np.zeros((batch_iterations, N))
            pos_y = np.zeros((batch_iterations, N))

            # Load particle positions from CSV files
            for i, iteration in enumerate(range(batch_start, batch_end)):
                x_filename = os.path.join(f'x_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep',
                                          f'x_{iteration+1}it_{B*1000:.1f}mT.csv')
                y_filename = os.path.join(f'y_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep',
                                          f'y_{iteration+1}it_{B*1000:.1f}mT.csv')
                
                with open(x_filename, 'r') as x_file, open(y_filename, 'r') as y_file:
                    x_reader = csv.reader(x_file)
                    y_reader = csv.reader(y_file)
                    next(x_reader)  # Skip headers
                    next(y_reader)
                    x_row = list(x_reader)[n_step]
                    y_row = list(y_reader)[n_step]
                    
                    # Normalize positions by particle diameter
                    pos_x[i, :] = np.array(x_row[1:], dtype=float) / sigma
                    pos_y[i, :] = np.array(y_row[1:], dtype=float) / sigma

            # Initialize array to store batch structure factor
            Sqprom_batch = np.zeros((num_wave_steps + 1, num_angle_steps))
            
            # Process wavevectors in chunks
            for chunk_start in range(0, num_wave_steps + 1, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_wave_steps + 1)
                qx_chunk = qx[chunk_start:chunk_end, :]
                qy_chunk = qy[chunk_start:chunk_end, :]
                
                # Initialize chunk structure factor
                Sqprom_chunk = np.zeros((chunk_end - chunk_start, num_angle_steps))
                
                # Calculate structure factor for the chunk
                for i in range(batch_iterations):
                    delta_x = pos_x[i, :, None] - pos_x[i, None, :]
                    delta_y = pos_y[i, :, None] - pos_y[i, None, :]
                    np.fill_diagonal(delta_x, 0)  # Avoid self-interaction
                    np.fill_diagonal(delta_y, 0)
                    
                    # Calculate cosine term for the chunk
                    cos_term = np.cos(
                        qx_chunk[None, None, :, :] * delta_x[:, :, None, None] +
                        qy_chunk[None, None, :, :] * delta_y[:, :, None, None]
                    )
                    
                    # Sum over all particles and normalize
                    Sqprom_chunk += np.sum(cos_term, axis=(0, 1)) / (N * (N - 1))
                
                # Accumulate results for the chunk
                Sqprom_batch[chunk_start:chunk_end, :] += Sqprom_chunk / batch_iterations
            
            # Accumulate results for the batch
            Sqprom_total += Sqprom_batch / num_batches

        # Average over all batches
        Sqprom_avg = np.mean(Sqprom_total, axis=1)  # Average over angles
        structure_factors[f'Sq_avg_{B/1e-3:.1f}mT'] = Sqprom_avg
    
    # Save structure factors to a CSV file
    struct_factor_filename = os.path.join(results_dir, f'struct_factor_all_B_{n_step}step.csv')
    df = pd.DataFrame(structure_factors)
    df.to_csv(struct_factor_filename, index=False)
    
    print(f"All structure factors saved to {struct_factor_filename}")


def plot_structure_factors(N, steps, iterations, separation, B_values, n_step, 
                           eta, sigma, v, dt, T, mm, start=0, end=None, figsize=(7,7), dipole_moment=None, label=True):
    """
    Plots the structure factor for a system of microswimmers with a dynamic header.

    Args:
        N (int): Number of particles.
        steps (int): Number of steps in the simulation.
        iterations (int): Number of simulation repetitions.
        B_values (list): List of magnetic field values.
        eta (float): Viscosity of the fluid (in Pa.s).
        sigma (float): Diameter of the particles (in meters).
        v (float): Velocity of the particles (in meters/second).
        dt (float): Time step size (in seconds).
        T (float): Temperature in Kelvin.
        mm (float): Magnetic moment of the particles (in Am²).
        start (int): Initial step for plotting.
        end (int): Final step for plotting. If None, all steps are plotted.
        figsize (tuple): Size of the figure.
        dipole_moment (str): Title for dipole moment of the particles.
        label (bool): Indicates whether to include labels in the plot.
    """
    plt.figure(figsize=figsize)
    fontsize = 22+8
    linewidth = 4+4
    labelfont=fontsize - 4

    # Calculate diffusion coefficients
    k = 1.38e-23  # Boltzmann constant (in J/K)
    K0 = k * T  # Thermal energy
    Dt = K0 / (3 * np.pi * eta * sigma)  # Translational diffusion coefficient
    Dr = K0 / (np.pi * eta * sigma**3)  # Rotational diffusion coefficient

    # Directory where the results were saved
    results_dir = f'struct_factor_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'

    # Read the CSV file with all structure factors
    struct_factor_filename = os.path.join(results_dir, f'struct_factor_all_B_{n_step}step.csv')
    df = pd.read_csv(struct_factor_filename)

    # Use viridis colormap
    colors = get_cmap('viridis')(np.linspace(0, 1, len(B_values)))

    # Plot each structure factor
    for B, color in zip(B_values, colors):            
        if end is None:
            end = len(df["q_sigma"])
        plt.plot(df["q_sigma"][start:end], 
                 df[f'Sq_avg_{B/1e-3:.1f}mT'][start:end],
                 label=f'{B/1e-3:.1f}', color=color, linewidth=linewidth+2, linestyle=(0, (1, 1)))

    if label:
        plt.legend(fontsize=fontsize-4, title='B (mT)', loc='upper right', title_fontsize=fontsize-4)
    
    plt.title(label=f'$\\tau$ = {n_step*dt*Dr:.3f}', fontsize=fontsize)
    plt.xlabel('$q\\sigma$', fontsize=fontsize)
    plt.ylabel(f'm = {dipole_moment}m$_0$\nS', fontsize=fontsize)

    plt.ylim(0, 6.4)  # Adjust y-axis limits
    plt.tick_params(axis='x', labelsize=fontsize)  # Change x-axis label size
    plt.tick_params(axis='y', labelsize=fontsize)  # Change y-axis label size
    plt.xticks([3,6,9,12])

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def load_phi_data(N, steps, iterations, B_values, separation, n_step):
    """
    Loads the phi data for a specific step n from all iterations, separated by B field value.
    
    Parameters:
        N (int): Number of particles.
        steps (int): Number of simulation steps.
        iterations (int): Number of iterations.
        B_values (list): List of B field values.
        separation (float): Initial particle separation.
        n_step (int): Step number for which the data will be loaded.
    
    Returns:
        dict: A dictionary where the keys are the B values and the values are arrays of phi.
    """
    phi_data = {B: [] for B in B_values}  # Dictionary to store phi data by B

    for B in B_values:
        for iteration in range(iterations):
            # Construct the filename
            phi_filename = os.path.join(f'phi_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep', 
                                       f'phi_{iteration+1}it_{B*1000:.1f}mT.csv')
            
            # Load the data from the file
            if os.path.exists(phi_filename):
                data = pd.read_csv(phi_filename)
                phi_data[B].append(data.iloc[n_step, 1:].values)  # Ignore the time column

    # Concatenate all phi data for each B
    for B in B_values:
        if phi_data[B]:  # Check if there is data
            phi_data[B] = np.concatenate(phi_data[B], axis=0)
        else:
            raise FileNotFoundError(f"No files found for B = {B}")

    # Normalize phi_data to be in the range [0, 2pi]
    phi_data = {B: (phi_data[B] + 2 * np.pi) % (2 * np.pi) for B in B_values}

    return phi_data

def calculate_kde_phi_data(N, steps, iterations, separation, B_values, n_step, overwrite=False):
    """
    Calculate the kernel density estimation (KDE) for the angle phi at a specific step n
    for different values of B and save the results to CSV files.

    Parameters:
        N (int): Number of particles.
        steps (int): Number of simulation steps.
        iterations (int): Number of iterations.
        separation (float): Initial particle separation.
        B_values (list): List of B field values.
        n_step (int): Step number for which the data will be calculated.
        overwrite (bool): If True, overwrite existing files.

    """
    # Create directory to save results
    save_dir = f'kde_phi_data_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'
    os.makedirs(save_dir, exist_ok=True)

    phi_data = load_phi_data(N, steps, iterations, B_values, separation, n_step)

    for B in B_values:
        filename = os.path.join(save_dir, f'kde_phi_B{B*1000:.1f}mT_step{n_step}.csv')
        if not overwrite and os.path.exists(filename):
            continue  # Ya existe, no recalcular

        # Calcular KDE con seaborn
        kde = sns.kdeplot(phi_data[B], bw_method='scott', clip=(0, 2 * np.pi), gridsize=1000)
        phi, density = kde.get_lines()[0].get_data()
        plt.cla()  # Limpia el plot de seaborn

        if B == 0:
            uniform_value = 1 / (2 * np.pi)  # Valor uniforme para B=0
            density_norm = np.full(len(phi), uniform_value)  # Uniform distribution for B=0
        else:
            # Calcular la normal ajustada
            mean = np.mean(phi_data[B])
            std = np.std(phi_data[B])
            import scipy.stats as stats
            density_norm = stats.norm.pdf(phi, loc=mean, scale=std)  # Normal distribution

        kde_df = pd.DataFrame({'phi': phi, 'density': density, 'density_norm': density_norm})
        kde_df.to_csv(filename, index=False)

        print(f"KDE for B = {B*1000:.1f}mT saved to {filename}")
        plt.close()  # Cierra la figura de seaborn para liberar memoria

def plot_kde_phi_data(N, steps, iterations, dt, separation, B_values, eta, sigma, T, dipole_moment, n_step, 
                      figsize=(7,6), ylimsup=1, normal_dist=False, label=True):
    """
    Plot the kernel density estimation (KDE) of the angle phi for different values of B at a specific time step.
    
    Parameters:
        N (int): Number of particles.
        steps (int): Number of simulation steps.
        iterations (int): Number of iterations.
        dt (float): Time step size.
        separation (float): Separation between particles.
        B_values (list): List of magnetic field values.
        eta (float): Viscosity of the fluid.
        sigma (float): Diameter of the particles.
        T (float): Temperature in Kelvin.
        dipole_moment (str): Title for dipole moment of the particles.
        n_step (int): Step number for which the data will be plotted.
        figsize (tuple): Size of the figure.
        ylimsup (float): Upper limit for y-axis.
        normal_dist (bool): If True, plot the normal distribution.
        label (bool): If True, display the legend.
        
    """
    plt.figure(figsize=figsize)
    fontsize = 22+8
    linewidth = 4+4
    labelfont=fontsize - 4

    # Calculate diffusion coefficients
    k = 1.38e-23  # Boltzmann constant (in J/K)
    K0 = k * T  # Thermal energy
    Dt = K0 / (3 * np.pi * eta * sigma)  # Translational diffusion coefficient
    Dr = K0 / (np.pi * eta * sigma**3)  # Rotational diffusion coefficient

    # Load KDE data
    save_dir = f'kde_phi_data_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'
    os.makedirs(save_dir, exist_ok=True)

    # Use viridis colormap
    colors = get_cmap('viridis')(np.linspace(0, 1, len(B_values)))

    for B, color in zip(B_values, colors):
        filename = os.path.join(save_dir, f'kde_phi_B{B*1000:.1f}mT_step{n_step}.csv')
        if not os.path.exists(filename):
            print(f"File {filename} not found. Please run calculate_kde_phi_data first.")
            continue

        kde_df = pd.read_csv(filename)
        
        # Plot KDE
        plt.plot(kde_df['phi'], kde_df['density'], label=f'{B*1000:.1f}', linewidth=linewidth, color=color, linestyle=(0, (1, 1)))
        
        if normal_dist:
            # Plot normal distribution
            plt.plot(kde_df['phi'], kde_df['density_norm'], linestyle='-', linewidth=linewidth-2, alpha=0.6, color=color)

    # Configure x-axis ticks in multiples of pi/2
    plt.xticks(
        np.linspace(0, 2 * np.pi, 5),
        [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    )
    
    # Labels and title
    plt.xlabel('$\\phi$', fontsize=fontsize, labelpad=15)
    plt.ylabel('Density', fontsize=fontsize)
    plt.title(label=f'm = {dipole_moment}m$_0$, $\\tau$ = {n_step*dt/Dr:.3f}', fontsize=fontsize)
    
    # Set x-axis limits to [0, 2π]
    plt.xlim(0, 2 * np.pi)
    plt.ylim(0,ylimsup)
    plt.tick_params(axis='x', labelsize=fontsize)  # Change x-axis label size
    plt.tick_params(axis='y', labelsize=fontsize)  # Change y-axis label size
    
    # Legend and grid
    if label:
        plt.legend(fontsize=fontsize-4, title='B (mT)', loc='upper right', title_fontsize=fontsize-4)
    
    plt.title(label=f'$\\tau$ = {n_step*dt*Dr:.3f}', fontsize=fontsize)
    plt.xlabel('$\\phi$', fontsize=fontsize)
    plt.ylabel(f'm = {dipole_moment}m$_0$\nDensity', fontsize=fontsize)    

    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show plot
    plt.show()


def calculate_global_hexagonal_order(N, steps, iterations, separation, sigma, B_values, 
                                   n_step, v, rc=1.5, batch_size=100):
    """
    Calculates the global hexagonal bond order parameter (ψ6) and its standard deviation
    for a system of particles under different magnetic fields.

    Args:
        N (int): Number of particles.
        steps (int): Number of time steps in the simulation.
        iterations (int): Number of repetitions of the simulation.
        separation (float): Separation between particles in the initial configuration.
        sigma (float): Diameter of the particles.
        B_values (numpy.ndarray): Array of magnetic field values.
        n_step (int): Specific time step to calculate ψ6.
        v (float): Velocity of the particles.
        rc (float): Cutoff distance for nearest neighbors (in units of sigma). Default is 1.5.
        batch_size (int): Number of iterations to process in each batch. Default is 100.

    Returns:
        None (saves results to CSV file)
    """
    # Create directory to save results
    results_dir = f'hex_order_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep'
    os.makedirs(results_dir, exist_ok=True)

    # Initialize lists to store results
    B_list = []
    psi6_avg_list = []
    psi6_std_list = []

    # Calculate ψ6 for each magnetic field value
    for B in B_values:
        psi6_values = []  # Store all ψ6 values for this B to calculate std
        num_batches = (iterations + batch_size - 1) // batch_size  # Round up
        
        # Process iterations in batches to reduce memory usage
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, iterations)
            batch_iterations = batch_end - batch_start
            
            # Initialize arrays to store particle positions
            pos_x = np.zeros((batch_iterations, N))
            pos_y = np.zeros((batch_iterations, N))

            # Load particle positions from CSV files
            for i, iteration in enumerate(range(batch_start, batch_end)):
                x_filename = os.path.join(f'x_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep',
                                        f'x_{iteration+1}it_{B*1000:.1f}mT.csv')
                y_filename = os.path.join(f'y_{N}part_{steps/1000:.0f}ksteps_{iterations}its_{separation}sep',
                                        f'y_{iteration+1}it_{B*1000:.1f}mT.csv')
                
                with open(x_filename, 'r') as x_file, open(y_filename, 'r') as y_file:
                    x_reader = csv.reader(x_file)
                    y_reader = csv.reader(y_file)
                    next(x_reader)  # Skip headers
                    next(y_reader)
                    x_row = list(x_reader)[n_step]
                    y_row = list(y_reader)[n_step]
                    
                    # Normalize positions by particle diameter
                    pos_x[i, :] = np.array(x_row[1:], dtype=float) / sigma
                    pos_y[i, :] = np.array(y_row[1:], dtype=float) / sigma

            # Calculate ψ6 for each configuration in the batch
            for i in range(batch_iterations):
                # Combine positions into a single array
                positions = np.column_stack((pos_x[i], pos_y[i]))
                
                # Use KDTree to find nearest neighbors efficiently
                tree = cKDTree(positions)
                neighbors = tree.query_ball_tree(tree, rc)
                
                psi6_i = 0.0
                
                for j in range(N):
                    # Get neighbors (excluding self)
                    neighbor_indices = [n for n in neighbors[j] if n != j]
                    Nb = len(neighbor_indices)
                    
                    if Nb == 0:
                        continue  # skip if no neighbors
                    
                    # Calculate angles to all neighbors
                    dx = pos_x[i, neighbor_indices] - pos_x[i, j]
                    dy = pos_y[i, neighbor_indices] - pos_y[i, j]
                    theta_ij = np.arctan2(dy, dx)
                    
                    # Calculate local ψ6 contribution
                    psi6_i += np.abs(np.sum(np.exp(6j * theta_ij))) / Nb
                
                # Normalize by number of particles and store
                psi6_values.append(psi6_i / N)

        # Calculate average and standard deviation
        psi6_avg = np.mean(psi6_values)
        psi6_std = np.std(psi6_values, ddof=1)  # Sample standard deviation
        
        B_list.append(B * 1000)  # Convert to mT
        psi6_avg_list.append(psi6_avg)
        psi6_std_list.append(psi6_std)
    
    # Create DataFrame from lists
    psi6_results = {
        "B_mT": B_list,
        "psi6_mean": psi6_avg_list,
        "psi6_std": psi6_std_list,
        "relative_std": [std/mean if mean !=0 else 0 for mean, std in zip(psi6_avg_list, psi6_std_list)]
    }
    df = pd.DataFrame(psi6_results)
    
    # Save results to a CSV file
    psi6_filename = os.path.join(results_dir, f'global_hex_order_{n_step}step.csv')
    df.to_csv(psi6_filename, index=False)
    
    print(f"Global hexagonal order parameters saved to {psi6_filename}")
    print("\nResults summary:")
    print(df)


def calculate_local_phi6(N, steps, iterations, separation, sigma, B_values, 
                        n_step, rc=1.5, batch_size=100):
    """
    Calculates the local hexagonal bond order parameter (φ₆) for a system of particles
    under different magnetic fields. The results are saved in a CSV file.
    
    Args:
        N (int): Number of particles.
        steps (int): Number of time steps in the simulation.
        iterations (int): Number of repetitions of the simulation.
        separation (float): Separation between particles in the initial configuration.
        sigma (float): Diameter of the particles.
        B_values (numpy.ndarray): Array of magnetic field values.
        n_step (int): Specific time step to calculate φ₆.
        rc (float): Cutoff distance for nearest neighbors (in units of sigma). Default is 1.5.
        batch_size (int): Number of iterations to process in each batch. Default is 100.
        
    Returns:
        None
    """
    
    # Create directory to save results
    results_dir = f'local_phi6_{N}part_{int(steps/1000)}ksteps_{iterations}its_{separation}sep'
    os.makedirs(results_dir, exist_ok=True)
    
    # Dictionary to store results
    phi6_results = {"B_mT": [], "iteration": [], "particle": [], "phi6": []}
    
    # Process in batches to reduce memory
    num_batches = (iterations + batch_size - 1) // batch_size
    
    for B in B_values:
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, iterations)
            
            # Load positions for this batch
            pos_data = []
            for iteration in range(batch_start, batch_end):
                x_file = f'x_{N}part_{int(steps/1000)}ksteps_{iterations}its_{separation}sep/x_{iteration+1}it_{B*1000:.1f}mT.csv'
                y_file = f'y_{N}part_{int(steps/1000)}ksteps_{iterations}its_{separation}sep/y_{iteration+1}it_{B*1000:.1f}mT.csv'
                
                with open(x_file, 'r') as xf, open(y_file, 'r') as yf:
                    x_reader = list(csv.reader(xf))[n_step+1]  # Skip header
                    y_reader = list(csv.reader(yf))[n_step+1]
                    
                    x = np.array(x_reader[1:], dtype=float)/sigma
                    y = np.array(y_reader[1:], dtype=float)/sigma
                    pos_data.append(np.column_stack((x, y)))
            
            # Calculate φ₆ for each particle in each iteration
            for i, positions in enumerate(pos_data):
                tree = cKDTree(positions)
                neighbors = [tree.query_ball_point(p, rc) for p in positions]
                
                for p in range(N):
                    nb = [n for n in neighbors[p] if n != p]  # Exclude the particle itself
                    if len(nb) >= 1:  # Minimum 1 neighbor for meaningful calculation
                        vectors = positions[nb] - positions[p]
                        theta = np.arctan2(vectors[:,1], vectors[:,0])
                        phi6 = np.abs(np.mean(np.exp(6j * theta)))
                    else:
                        phi6 = 0.0
                    
                    phi6_results["B_mT"].append(B*1000)  # Convert to mT
                    phi6_results["iteration"].append(batch_start + i + 1)
                    phi6_results["particle"].append(p)
                    phi6_results["phi6"].append(phi6)
        
        print(f"B = {B*1000:.1f} mT processed")
                    
    
    # Save results
    df = pd.DataFrame(phi6_results)
    output_file = os.path.join(results_dir, f'phi6_B_all_step{n_step}.csv')
    df.to_csv(output_file, index=False)
    print(f"Datos de φ₆ guardados en: {output_file}")


def plot_phi6_snapshot(N, steps, iterations, iteration, separation, sigma, B, n_step, rc=1.5, dipole_moment='1'):
    """
    Plots the configuration of particles at a given time step, colored by local hexagonal order (phi6),
    using the results from calculate_phi6_data function.

    Parameters:
        N (int): Number of particles.
        steps (int): Total simulation steps.
        iterations (int): Total iterations (repetitions) of the simulation.
        iteration (int): Iteration number to analyze.
        separation (float): Initial particle separation.
        sigma (float): Particle diameter.
        B (float): Magnetic field (in T).
        n_step (int): Time step to analyze.
        rc (float): Cutoff radius for neighbors (in units of sigma). Default is 1.5.
        dipole_moment (str): Dipole moment value for the plot title.
    """

    general_name = f'{N}part_{int(steps/1000)}ksteps_{iterations}its_{separation}sep'
    x_dir = f'x_{general_name}'
    y_dir = f'y_{general_name}'
    local_phi6_dir = f'local_phi6_{general_name}'

    # File paths
    x_filename = os.path.join(x_dir, f'x_{iteration}it_{B*1000:.1f}mT.csv')
    y_filename = os.path.join(y_dir, f'y_{iteration}it_{B*1000:.1f}mT.csv')
    phi6_results_file = os.path.join(local_phi6_dir, f'phi6_B_all_step{n_step}.csv')

    fontsize = 22

    # Load position and orientation data
    with open(x_filename, 'r', encoding='utf-8') as x_file, open(y_filename, 'r', encoding='utf-8') as y_file:
        x_reader = list(csv.reader(x_file))
        y_reader = list(csv.reader(y_file))
        
        # Get current positions and orientations
        x_row = x_reader[n_step+1]  # +1 to skip header
        y_row = y_reader[n_step+1]

        del x_reader, y_reader  # Free memory

        x = np.array(x_row[1:], dtype=float) / sigma
        y = np.array(y_row[1:], dtype=float) / sigma

    # Load phi6 data from the consolidated results file
    phi6_data = pd.read_csv(phi6_results_file)
    current_phi6 = phi6_data[(phi6_data['B_mT'] == B*1000) & 
                            (phi6_data['iteration'] == iteration)]
    
    # Ensure we have phi6 values for all particles
    if len(current_phi6) != N:
        raise ValueError(f"Expected {N} phi6 values, found {len(current_phi6)}")
    
    phi6_local = current_phi6['phi6'].values

    # Create figure
    plt.figure(figsize=(10, 8))
    cmap = get_cmap('viridis')
    norm = Normalize(vmin=0.0, vmax=1.0)  # phi6 between 0 and 1
    
    # Plot particles colored by phi6
    sc = plt.scatter(x, y, c=phi6_local, cmap=cmap, norm=norm, s=32, alpha=1)
    
    # Add arrows with quiver
    colors = plt.cm.viridis(norm(phi6_local))  # Get colors matching phi6 values
    
    # Add circular markers
    radius = 0.5  # Graphic units
    for i in range(N):
        circle = plt.Circle((x[i], y[i]), radius=radius, color=colors[i], 
                           alpha=1, label='Current positions' if i == 0 else "")
        plt.gca().add_patch(circle)

    # Plot adjustments
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f'd₀ = {separation}σ, m = {dipole_moment}m$_0$\n B = {B*1000:.1f} mT', fontsize=fontsize)
    plt.xlabel('x (σ)', fontsize=fontsize)
    plt.ylabel(f'y (σ)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # Add colorbar
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\Psi_6$', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    
    plt.tight_layout()
    plt.show()


def calculate_cluster_size_distribution(N, steps, iterations, separation, sigma, B_values, 
                                     n_step, rc=1.5, bandwidth=0.5, min_cluster_size=2):
    """
    Calculate and save cluster size distributions for different magnetic fields.
    
    Args:
        N (int): Number of particles
        steps (int): Total simulation steps
        iterations (int): Number of simulation repetitions
        separation (float): Initial separation between particles
        sigma (float): Particle diameter
        B_values (array-like): Array of magnetic field values (in Tesla)
        n_step (int): Time step to analyze
        rc (float): Cutoff radius for neighbor consideration (in units of sigma)
        bandwidth (float): Bandwidth for KDE smoothing
        min_cluster_size (int): Minimum size to consider a cluster
        
    Returns:
        None (saves results to CSV files)
    """
    # Create results directory
    results_dir = f'cluster_distribution_{N}part_{int(steps/1000)}ksteps_{iterations}its_{separation}sep'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a summary DataFrame that will be saved
    summary_data = []
    
    for B in tqdm(B_values, desc="Processing magnetic fields"):
        B_mT = B * 1000  # Convert to mT
        all_cluster_sizes = []
        
        for iteration in range(1, iterations + 1):
            # Load particle positions
            x_file = f'x_{N}part_{int(steps/1000)}ksteps_{iterations}its_{separation}sep/x_{iteration}it_{B_mT:.1f}mT.csv'
            y_file = f'y_{N}part_{int(steps/1000)}ksteps_{iterations}its_{separation}sep/y_{iteration}it_{B_mT:.1f}mT.csv'
            
            try:
                # Read position data and convert to numeric values
                x = pd.read_csv(x_file, header=None).apply(pd.to_numeric, errors='coerce').iloc[n_step+1, 1:].values / sigma
                y = pd.read_csv(y_file, header=None).apply(pd.to_numeric, errors='coerce').iloc[n_step+1, 1:].values / sigma
                positions = np.column_stack((x, y))
                
                # Find clusters
                tree = cKDTree(positions)
                neighbors = tree.query_ball_tree(tree, rc)
                
                # Identify clusters
                visited = np.zeros(N, dtype=bool)
                cluster_sizes = []
                
                for i in range(N):
                    if not visited[i]:
                        cluster = []
                        queue = [i]
                        visited[i] = True
                        
                        while queue:
                            particle = queue.pop()
                            cluster.append(particle)
                            
                            for neighbor in neighbors[particle]:
                                if not visited[neighbor]:
                                    visited[neighbor] = True
                                    queue.append(neighbor)
                        
                        if len(cluster) >= min_cluster_size:
                            cluster_sizes.append(len(cluster))
                
                all_cluster_sizes.extend(cluster_sizes)
            
            except Exception as e:
                print(f"Error processing B={B_mT}mT, iteration={iteration}: {str(e)}")
                continue
        
        # Convert to numpy array
        cluster_sizes_array = np.array(all_cluster_sizes)
        
        # Save raw cluster sizes to CSV
        cluster_filename = os.path.join(results_dir, f'cluster_sizes_{B_mT:.1f}mT_{n_step}step.csv')
        np.savetxt(cluster_filename, cluster_sizes_array, fmt='%d', delimiter=',')
        
        # Calculate and save histogram
        bins = np.arange(0.5, N+0.5, 1)
        hist, bin_edges = np.histogram(cluster_sizes_array, bins=bins)
        hist_data = np.column_stack((bin_edges[:-1], hist))
        hist_filename = os.path.join(results_dir, f'cluster_histogram_{B_mT:.1f}mT_{n_step}step.csv')
        np.savetxt(hist_filename, hist_data, fmt=['%.1f', '%d'], delimiter=',', 
                  header='cluster_size,count')
        
        # Calculate KDE if we have clusters
        if len(cluster_sizes_array) > 0:
            x_grid = np.linspace(0, N, 1000)
            data = cluster_sizes_array.reshape(-1, 1)
            
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(data)
            log_dens = kde.score_samples(x_grid.reshape(-1, 1))
            density = np.exp(log_dens)
        else:
            x_grid = np.linspace(0, N, 1000)
            density = np.zeros_like(x_grid)
        
        # Save KDE to CSV
        kde_data = np.column_stack((x_grid, density))
        kde_filename = os.path.join(results_dir, f'cluster_kde_{B_mT:.1f}mT_{n_step}step.csv')
        np.savetxt(kde_filename, kde_data, fmt=['%.4f', '%.6f'], delimiter=',', 
                  header='cluster_size,density')
        
        # Add to summary data
        if len(cluster_sizes_array) > 0:
            avg_size = np.mean(cluster_sizes_array)
            max_size = np.max(cluster_sizes_array)
            num_clusters = len(cluster_sizes_array)
        else:
            avg_size = 0
            max_size = 0
            num_clusters = 0
            
        summary_data.append({
            'B_mT': B_mT,
            'avg_cluster_size': avg_size,
            'max_cluster_size': max_size,
            'total_clusters': num_clusters
        })
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_filename = os.path.join(results_dir, f'cluster_summary_{n_step}step.csv')
    summary_df.to_csv(summary_filename, index=False)
    
    # Save parameters
    params = {
        'N': N,
        'steps': steps,
        'iterations': iterations,
        'separation': separation,
        'sigma': sigma,
        'n_step': n_step,
        'rc': rc,
        'bandwidth': bandwidth,
        'min_cluster_size': min_cluster_size
    }
    params_df = pd.DataFrame([params])
    params_filename = os.path.join(results_dir, f'simulation_parameters_{n_step}step.csv')
    params_df.to_csv(params_filename, index=False)


def plot_cluster_size_distribution(N, steps, iterations, separation, B_values, dt, Dr, n_step, 
                                 figsize=(10, 6), ylim=None, legend=True):
    """
    Plot cluster size distributions from saved CSV files.
    
    Args:
        N (int): Number of particles
        steps (int): Total simulation steps
        iterations (int): Number of simulation repetitions
        separation (float): Initial separation between particles
        B_values (array-like): Array of magnetic field values (in Tesla)
        dt (float): Time step size
        Dr (float): Rotational diffusion coefficient
        n_step (int): Time step to analyze
        figsize (tuple): Figure size
        ylim (tuple): Y-axis limits
        legend (bool): Whether to show the legend
    """
    plt.figure(figsize=figsize)
    fontsize = 22+8
    linewidth = 4+4
    labelfont=fontsize - 4
    
    
    # Directory where results are saved
    results_dir = f'cluster_distribution_{N}part_{int(steps/1000)}ksteps_{iterations}its_{separation}sep'
    
    # Create colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(B_values)))
    
    for B, color in zip(B_values, colors):
        B_mT = B * 1000
        kde_filename = os.path.join(results_dir, f'cluster_kde_{B_mT:.1f}mT_{n_step}step.csv')
        
        # Load KDE data
        kde_data = np.loadtxt(kde_filename, delimiter=',', skiprows=1)
        x_grid = kde_data[:, 0]
        density = kde_data[:, 1]

        # Plot KDE
        plt.plot(x_grid, density, label=f'{B_mT:.1f}',
                 color=color, linewidth=linewidth, linestyle=(0, (1, 1)) )
    
    plt.title(f'd₀ = {separation}σ', fontsize=fontsize)
    
    plt.xlabel('Cluster size (# of particles)', fontsize=fontsize)
    plt.ylabel('m = $\sqrt{10}m_0$\nDensity', fontsize=fontsize)
    
    plt.tick_params(axis='x', labelsize=fontsize)  # Change x-axis label size
    plt.tick_params(axis='y', labelsize=fontsize)  # Change y-axis label size

    if legend:    
        plt.legend(fontsize=fontsize - 4, title='B (mT)',
               title_fontsize=fontsize - 4)
    
    plt.grid(True, alpha=0.3)
    
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.tight_layout()
    plt.show()