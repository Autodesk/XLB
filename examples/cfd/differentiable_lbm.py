"""
Differentiable LBM Example with Configurable Target Shapes

This example demonstrates gradient-based optimization of initial conditions
to achieve various target density patterns using the Lattice Boltzmann Method.

Available target shapes:
- 'n_letter': Letter N pattern
- 'circle': Circular pattern
- 'cross': Cross/plus pattern
- 'checkerboard': Checkerboard pattern

The optimization finds initial conditions (distribution function f) that,
after simulation, produce a density field matching the target pattern.

Key concepts:
- LBM density stays ~1.0 (physics constraint), so we normalize to [0,1] for loss
- JAX backend is used for automatic differentiation through the stepper
- Simple gradient descent with tuned learning rate

References:
- Warp example: warp/examples/optim/example_fluid_checkpoint.py
- XLB OOC example: examples/out_of_core/autodiff_lbm.py
"""

import argparse
import os
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
from jax import value_and_grad

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization disabled")

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.macroscopic import Macroscopic
import xlb.velocity_set

# For loading XLB logo
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Available target shapes
AVAILABLE_SHAPES = ['n_letter', 'circle', 'cross', 'checkerboard']


class DifferentiableLBM:
    """
    Differentiable LBM with configurable target shapes.

    Optimizes initial conditions to achieve a target density pattern.
    """

    def __init__(
        self,
        grid_shape=(128, 128),
        Re=100.0,
        sim_steps=50,
        target_shape='n_letter',
        learning_rate=1.0,
        target_coverage=0.5,  # Fraction of grid covered by target pattern
        target_image_path=None,  # Path to custom target image (e.g., XLB logo)
    ):
        self.grid_shape = grid_shape
        self.Re = Re
        self.sim_steps = sim_steps
        self.target_shape = target_shape
        self.learning_rate = learning_rate
        self.target_coverage = target_coverage
        self.output_dir = None  # Set by run_optimization if saving
        self.target_image_path = target_image_path

        # LBM parameters
        self.rho_background = 1.0
        self.rho_variation = 0.1  # Density varies from 0.9 to 1.1

        # Compute omega from Reynolds number
        # Re = u * L / nu, nu = (1/omega - 0.5) / 3
        L = grid_shape[0]
        u_ref = 0.1
        nu = u_ref * L / Re
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.omega = np.clip(self.omega, 0.5, 1.99)

        # Use JAX backend - required for autodiff through the stepper
        # Note: XLB's Warp stepper kernel doesn't have adjoint implementations,
        # so gradients are zero when using wp.Tape (verified by test_stepper_autodiff.py).
        # JAX uses source transformation which works through the stepper.
        self.compute_backend = ComputeBackend.JAX
        self.precision_policy = PrecisionPolicy.FP32FP32

        # Initialize velocity set
        self.velocity_set = xlb.velocity_set.D2Q9(
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )

        # Store lattice weights and velocities for equilibrium
        self.w = jnp.array(self.velocity_set.w, dtype=jnp.float32)
        self.c = jnp.array(self.velocity_set.c, dtype=jnp.int32)

        # Initialize XLB
        xlb.init(
            velocity_set=self.velocity_set,
            default_backend=self.compute_backend,
            default_precision_policy=self.precision_policy,
        )

        # Create grid and stepper (periodic boundaries)
        self.grid = grid_factory(grid_shape, compute_backend=self.compute_backend)
        self.stepper = IncompressibleNavierStokesStepper(
            grid=self.grid,
            boundary_conditions=[],  # Periodic
            collision_type="BGK",
        )

        # Prepare fields
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields()

        # Create macroscopic operator
        self.macroscopic = Macroscopic(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.compute_backend,
        )

        # Initialize with uniform density (far from any target)
        self._initialize_uniform()

        # Create target pattern
        self._create_target()

        print("DifferentiableLBM initialized:")
        print(f"  Grid: {grid_shape}")
        print(f"  Re: {Re}, omega: {self.omega:.4f}")
        print(f"  Sim steps: {sim_steps}")
        print(f"  Target shape: {target_shape}")
        print(f"  Learning rate: {learning_rate}")

    def _initialize_uniform(self):
        """Initialize with uniform density (normalized = 0)."""
        nx, ny = self.grid_shape
        rho = np.full((nx, ny), self.rho_background - self.rho_variation, dtype=np.float32)
        self.initial_density_normalized = self._normalize_density(rho)
        self.f_0 = self._equilibrium(jnp.array(rho), jnp.zeros((2, nx, ny)))
        self.f_1 = self.f_0.copy()

    def _normalize_density(self, rho):
        """Normalize density to [0, 1] range."""
        rho_min = self.rho_background - self.rho_variation
        rho_max = self.rho_background + self.rho_variation
        return (rho - rho_min) / (rho_max - rho_min)

    def _equilibrium(self, rho, u):
        """Compute equilibrium distribution."""
        cs2 = 1.0 / 3.0
        cu = self.c[0, :, None, None] * u[0] + self.c[1, :, None, None] * u[1]
        u_sq = u[0]**2 + u[1]**2
        f_eq = self.w[:, None, None] * rho * (
            1.0 + cu / cs2 + cu**2 / (2.0 * cs2**2) - u_sq / (2.0 * cs2)
        )
        return f_eq

    def _create_target(self):
        """Create target pattern based on selected shape."""
        shape_creators = {
            'n_letter': self._create_n_pattern,
            'circle': self._create_circle_pattern,
            'cross': self._create_cross_pattern,
            'checkerboard': self._create_checkerboard_pattern,
        }

        # If custom image path provided, use it
        if self.target_image_path:
            target = self._load_target_image(self.target_image_path)
        elif self.target_shape not in shape_creators:
            raise ValueError(f"Unknown shape: {self.target_shape}. Available: {AVAILABLE_SHAPES}")
        else:
            target = shape_creators[self.target_shape]()
        self.target_normalized = jnp.array(target)

        coverage = float(jnp.mean(target))
        initial_loss = float(jnp.mean((self.initial_density_normalized - target) ** 2))
        print(f"  Target coverage: {coverage*100:.1f}%")
        print(f"  Expected initial loss: {initial_loss:.6f}")

    def _create_n_pattern(self):
        """Create letter N pattern."""
        nx, ny = self.grid_shape
        target = np.zeros((nx, ny), dtype=np.float32)

        margin = nx // 10
        bar_width = int(nx * self.target_coverage / 3)  # Adjust for coverage

        # Left vertical bar
        target[margin:margin+bar_width, margin:ny-margin] = 1.0
        # Right vertical bar
        target[nx-margin-bar_width:nx-margin, margin:ny-margin] = 1.0
        # Diagonal
        for i in range(nx):
            j_center = int(margin + (ny - 2*margin) * (i - margin) / (nx - 2*margin))
            j_start = max(margin, j_center - bar_width//2)
            j_end = min(ny - margin, j_center + bar_width//2)
            if margin <= i < nx - margin:
                target[i, j_start:j_end] = 1.0

        return target

    def _create_circle_pattern(self):
        """Create circular pattern."""
        nx, ny = self.grid_shape
        target = np.zeros((nx, ny), dtype=np.float32)

        cx, cy = nx // 2, ny // 2
        # Radius based on coverage: pi*r^2 / (nx*ny) = coverage
        radius = np.sqrt(self.target_coverage * nx * ny / np.pi)

        for i in range(nx):
            for j in range(ny):
                if (i - cx)**2 + (j - cy)**2 < radius**2:
                    target[i, j] = 1.0

        return target

    def _create_cross_pattern(self):
        """Create cross/plus pattern."""
        nx, ny = self.grid_shape
        target = np.zeros((nx, ny), dtype=np.float32)

        # Width based on coverage: 2*w*L - w^2 = coverage * L^2
        # Simplified: w = coverage * L / 2
        width = int(self.target_coverage * nx / 2)

        cx, cy = nx // 2, ny // 2
        margin = nx // 10

        # Horizontal bar
        target[cx-width//2:cx+width//2, margin:ny-margin] = 1.0
        # Vertical bar
        target[margin:nx-margin, cy-width//2:cy+width//2] = 1.0

        return target

    def _create_checkerboard_pattern(self):
        """Create checkerboard pattern."""
        nx, ny = self.grid_shape
        target = np.zeros((nx, ny), dtype=np.float32)

        # Number of squares based on coverage (checkerboard is always ~50%)
        num_squares = 4  # 4x4 checkerboard
        sq_size_x = nx // num_squares
        sq_size_y = ny // num_squares

        for i in range(num_squares):
            for j in range(num_squares):
                if (i + j) % 2 == 0:
                    x_start = i * sq_size_x
                    x_end = (i + 1) * sq_size_x
                    y_start = j * sq_size_y
                    y_end = (j + 1) * sq_size_y
                    target[x_start:x_end, y_start:y_end] = 1.0

        return target

    def _load_target_image(self, image_path):
        """Load target pattern from image file."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required to load images. Install with: pip install Pillow")

        nx, ny = self.grid_shape
        try:
            img = Image.open(image_path)
            img_resized = img.resize((ny, nx))  # PIL uses (width, height)
            # Convert to grayscale and normalize to [0, 1]
            img_gray = img_resized.convert('L')
            target = np.array(img_gray, dtype=np.float32) / 255.0
            target = np.flipud(target)  # Flip vertically to match plot orientation
            target = target.T  # Transpose to (nx, ny)
            print(f"  Loaded target image: {image_path}")
            return target
        except Exception as e:
            raise ValueError(f"Could not load image {image_path}: {e}")

    def compute_loss(self, f):
        """Compute MSE loss on normalized density."""
        rho, _ = self.macroscopic(f)
        rho_norm = self._normalize_density(rho[0])
        rho_norm = jnp.clip(rho_norm, 0.0, 1.0)
        loss = jnp.mean((rho_norm - self.target_normalized) ** 2)
        return loss

    def forward(self, f_init):
        """Run simulation forward."""
        f_curr = f_init
        f_next = jnp.zeros_like(f_init)

        for step in range(self.sim_steps):
            _, f_next = self.stepper(
                f_curr, f_next, self.bc_mask, self.missing_mask, self.omega, step
            )
            f_curr, f_next = f_next, f_curr

        return f_curr

    def loss_fn(self, f_init):
        """Loss function for optimization."""
        f_final = self.forward(f_init)
        return self.compute_loss(f_final)

    def optimize_step(self):
        """Perform one gradient descent step."""
        loss_val, grad_f = value_and_grad(self.loss_fn)(self.f_0)

        # Gradient descent update
        self.f_0 = self.f_0 - self.learning_rate * grad_f

        # Clamp f to physical range
        f_min = 0.01 * self.w[:, None, None]
        f_max = 10.0 * self.w[:, None, None]
        self.f_0 = jnp.clip(self.f_0, f_min, f_max)

        self.f_1 = self.f_0.copy()

        return float(loss_val)

    def get_initial_density(self):
        """Get normalized initial density (from current f_0)."""
        rho, _ = self.macroscopic(self.f_0)
        rho_norm = self._normalize_density(rho[0])
        return np.array(jnp.clip(rho_norm, 0.0, 1.0))

    def get_final_density(self):
        """Get normalized final density (after simulation)."""
        f_final = self.forward(self.f_0)
        rho, _ = self.macroscopic(f_final)
        rho_norm = self._normalize_density(rho[0])
        return np.array(jnp.clip(rho_norm, 0.0, 1.0))

    def save_iteration_plot(self, iteration, loss):
        """Save plot showing initial, final, and target density for this iteration."""
        if not MATPLOTLIB_AVAILABLE or self.output_dir is None:
            return

        initial = self.get_initial_density()
        final = self.get_final_density()
        target = np.array(self.target_normalized)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Initial density (what we're optimizing)
        im0 = axes[0].imshow(initial.T, origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title('Initial Condition\n(optimized f_0)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        # Final density (after simulation)
        im1 = axes[1].imshow(final.T, origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f'Final Density\n(after {self.sim_steps} steps)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        # Target density
        im2 = axes[2].imshow(target.T, origin='lower', cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title(f'Target\n({self.target_shape})')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

        # Difference (final - target)
        diff = np.abs(final - target)
        im3 = axes[3].imshow(diff.T, origin='lower', cmap='Reds', vmin=0, vmax=0.5)
        axes[3].set_title(f'|Final - Target|\nMSE={loss:.4f}')
        axes[3].set_xlabel('x')
        axes[3].set_ylabel('y')
        plt.colorbar(im3, ax=axes[3], shrink=0.8)

        plt.suptitle(f'Iteration {iteration:05d} - Loss: {loss:.6f}', fontsize=14)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f'iteration_{iteration:05d}.png')
        plt.savefig(filepath, dpi=100)
        plt.close(fig)

    def save_convergence_plot(self, losses):
        """Save convergence plot showing loss over iterations."""
        if not MATPLOTLIB_AVAILABLE or self.output_dir is None:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(losses, 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title(f'Optimization Convergence - {self.target_shape}', fontsize=14)
        ax.grid(True, alpha=0.3)
        # Use linear scale with regular numbers (not scientific notation)
        ax.ticklabel_format(style='plain', axis='y')

        # Add annotations
        ax.axhline(y=losses[-1], color='r', linestyle='--', alpha=0.5,
                   label=f'Final: {losses[-1]:.4f}')
        ax.legend()

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'convergence.png')
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"  Saved convergence plot: {filepath}")

    def run_optimization(self, num_iterations=100, verbose=True, save_plots=False,
                         save_every=10, output_dir=None):
        """Run optimization loop with optional visualization.

        Parameters
        ----------
        num_iterations : int
            Number of optimization iterations
        verbose : bool
            Print loss each iteration
        save_plots : bool
            Save density plots to disk
        save_every : int
            Save plot every N iterations (also saves first and last)
        output_dir : str
            Directory to save plots (default: output_diff_lbm_<timestamp>)
        """
        losses = []

        # Setup output directory if saving
        if save_plots and MATPLOTLIB_AVAILABLE:
            if output_dir is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = f'output_diff_lbm_{self.target_shape}_{timestamp}'
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"  Saving plots to: {self.output_dir}")

        for i in range(num_iterations):
            loss = self.optimize_step()
            losses.append(loss)

            if verbose:
                print(f"Iteration {i:05d} loss: {loss:.6f}")

            # Save plots at specified intervals
            if save_plots and MATPLOTLIB_AVAILABLE:
                if i == 0 or i == num_iterations - 1 or (i + 1) % save_every == 0:
                    self.save_iteration_plot(i, loss)

        # Save convergence plot
        if save_plots and MATPLOTLIB_AVAILABLE:
            self.save_convergence_plot(losses)

        return losses


def main():
    parser = argparse.ArgumentParser(
        description="Differentiable LBM with configurable target shapes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape", type=str, default="n_letter",
        choices=AVAILABLE_SHAPES,
        help="Target shape to optimize towards",
    )
    parser.add_argument(
        "--grid-size", type=int, default=128,
        help="Grid size (NxN)",
    )
    parser.add_argument(
        "--sim-steps", type=int, default=50,
        help="Number of simulation steps per forward pass",
    )
    parser.add_argument(
        "--iterations", type=int, default=150,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1.0,
        help="Learning rate for gradient descent",
    )
    parser.add_argument(
        "--Re", type=float, default=100.0,
        help="Reynolds number",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.5,
        help="Target pattern coverage (0-1)",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save density plots to disk",
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save plot every N iterations",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: auto-generated)",
    )
    parser.add_argument(
        "--target-image", type=str, default=None,
        help="Path to custom target image (overrides --shape)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Differentiable LBM - Configurable Target Shapes")
    print("=" * 70)
    print()

    sim = DifferentiableLBM(
        grid_shape=(args.grid_size, args.grid_size),
        Re=args.Re,
        sim_steps=args.sim_steps,
        target_shape=args.shape,
        learning_rate=args.learning_rate,
        target_coverage=args.coverage,
        target_image_path=args.target_image,
    )

    print()
    losses = sim.run_optimization(
        num_iterations=args.iterations,
        verbose=True,
        save_plots=args.save_plots,
        save_every=args.save_every,
        output_dir=args.output_dir,
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss:   {losses[-1]:.6f}")
    print(f"Improvement:  {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")

    # Check convergence
    if len(losses) >= 10:
        last_10_change = abs(losses[-10] - losses[-1]) / losses[-10] * 100
        print(f"Last 10 iter change: {last_10_change:.2f}%")
        if last_10_change < 1.0:
            print("Status: CONVERGED")
        else:
            print("Status: Still improving (run more iterations)")


if __name__ == "__main__":
    main()
