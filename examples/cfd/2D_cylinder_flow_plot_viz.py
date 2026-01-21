"""
Interactive visualization for 2D cylinder flow simulation.
Optimized for smooth, real-time slider interaction.

Usage: python plot_flow.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

# Load the saved data
print("Loading simulation data...")
try:
    saved_steps = np.load('saved_steps.npy')
    saved_rho = np.load('saved_rho.npy')
    saved_u = np.load('saved_u.npy')
    saved_vorticity = np.load('saved_vorticity.npy')
    print(f"Loaded {len(saved_steps)} frames")
except FileNotFoundError:
    print("Error: Data files not found. Please run 2D_cylinder_flow.py first.")
    exit(1)

# Simulation parameters
nx, ny = 800, 200
cylinder_radius = 10

# Pre-compute velocity magnitude for all frames
print("Pre-computing velocity magnitudes...")
saved_u_mag = np.sqrt(saved_u[:, 0]**2 + saved_u[:, 1]**2)

# Pre-compute global min/max for consistent colormaps (except vorticity)
u_mag_max = np.max(saved_u_mag)
rho_min, rho_max = np.min(saved_rho), np.max(saved_rho)
ux_min, ux_max = np.min(saved_u[:, 0]), np.max(saved_u[:, 0])

print("Setting up visualization...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax_slider = fig.add_subplot(gs[2, :])

# Initialize with first frame
frame_idx = 0

# Velocity magnitude
im1 = ax1.imshow(
    saved_u_mag[frame_idx].T,
    origin='lower',
    cmap='jet',
    extent=[0, nx, 0, ny],
    vmin=0,
    vmax=u_mag_max,
    aspect='equal'
)
circle1 = Circle((ny//2, ny//2), cylinder_radius, color='black', zorder=10)
ax1.add_patch(circle1)
ax1.set_title('Velocity Magnitude')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
cbar1 = plt.colorbar(im1, ax=ax1, label='|u|')

# Vorticity – no fixed vmin/vmax here
im2 = ax2.imshow(
    saved_vorticity[frame_idx].T,
    origin='lower',
    cmap='RdBu_r',
    extent=[0, nx, 0, ny],
    aspect='equal'
)
circle2 = Circle((ny//2, ny//2), cylinder_radius, color='black', zorder=10)
ax2.add_patch(circle2)
ax2.set_title('Vorticity (ω = ∂v/∂x - ∂u/∂y)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
cbar2 = plt.colorbar(im2, ax=ax2, label='ω')

# Density
im3 = ax3.imshow(
    saved_rho[frame_idx].T,
    origin='lower',
    cmap='viridis',
    extent=[0, nx, 0, ny],
    vmin=rho_min,
    vmax=rho_max,
    aspect='equal'
)
circle3 = Circle((ny//2, ny//2), cylinder_radius, color='black', zorder=10)
ax3.add_patch(circle3)
ax3.set_title('Density (ρ)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
cbar3 = plt.colorbar(im3, ax=ax3, label='ρ')

# X-velocity
im4 = ax4.imshow(
    saved_u[frame_idx, 0].T,
    origin='lower',
    cmap='coolwarm',
    extent=[0, nx, 0, ny],
    vmin=ux_min,
    vmax=ux_max,
    aspect='equal'
)
circle4 = Circle((ny//2, ny//2), cylinder_radius, color='black', zorder=10)
ax4.add_patch(circle4)
ax4.set_title('X-Velocity Component (u_x)')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
cbar4 = plt.colorbar(im4, ax=ax4, label='u_x')

title = fig.suptitle(
    f'2D Cylinder Flow - Step {saved_steps[frame_idx]}',
    fontsize=16,
    fontweight='bold'
)

# Create slider
slider = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=0,
    valmax=len(saved_steps) - 1,
    valinit=0,
    valstep=1
)

def update(val):
    """Update plots with dynamic vorticity scaling."""
    frame_idx = int(slider.val)

    # Velocity magnitude, density, x-velocity
    im1.set_data(saved_u_mag[frame_idx].T)
    im3.set_data(saved_rho[frame_idx].T)
    im4.set_data(saved_u[frame_idx, 0].T)

    # Vorticity with per-frame scaling (use percentiles to ignore outliers)
    vort = saved_vorticity[frame_idx]
    vmin = np.percentile(vort, 5)
    vmax = np.percentile(vort, 95)
    im2.set_data(vort.T)
    im2.set_clim(vmin, vmax)   # update color limits
    cbar2.update_normal(im2)   # keep colorbar in sync

    # Update title
    title.set_text(f'2D Cylinder Flow - Step {saved_steps[frame_idx]}')

    fig.canvas.draw_idle()

slider.on_changed(update)

print("\nVisualization ready!")
print("Use the slider to scroll through timesteps.")
print("Close the window to exit.")

plt.show()
