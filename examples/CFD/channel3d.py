"""
This script performs a 3D simulation of turbulent channel flow using the lattice Boltzmann method (LBM). 
Turbulent channel flow, also known as plane Couette flow, is a fundamental case in the study of wall-bounded turbulent flows.

In this example you'll be introduced to the following concepts:

1. Lattice: A D3Q27 lattice is used, which is a three-dimensional lattice model with 27 discrete velocity directions. This type of lattice allows for a more precise representation of fluid flow in three dimensions.

2. Initial Conditions: The initial conditions for the flow are randomly generated, and the populations are initialized to be the solution of an advection-diffusion equation.

3. Boundary Conditions: Bounce back boundary conditions are applied at the top and bottom walls, simulating a no-slip condition typical for wall-bounded flows.

4. External Force: An external force is applied to drive the flow.

"""

from src.boundary_conditions import *
from jax.config import config
from src.utils import *
import numpy as np
from src.lattice import LatticeD3Q27
from src.models import KBCSim, AdvectionDiffusionBGK
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax

# disable JIt compilation

jax.config.update('jax_enable_x64', True)

def vonKarman_loglaw_wall(yplus):
    vonKarmanConst = 0.41
    cplus = 5.5
    uplus = np.log(yplus)/vonKarmanConst + cplus
    return uplus

def get_dns_data():
    """
    Reference: DNS of Turbulent Channel Flow up to Re_tau=590, 1999,
    Physics of Fluids, vol 11, 943-945.
    https://turbulence.oden.utexas.edu/data/MKM/chan180/profiles/chan180.means
    """
    dns_dic = {
        "y":[0,0.000301,0.0012,0.00271,0.00482,0.00752,0.0108,0.0147,0.0192,0.0243,0.03,0.0362,0.0431,0.0505,0.0585,0.067,0.0761,0.0858,0.096,0.107,0.118,0.13,0.142,0.155,0.169,0.182,0.197,0.212,0.227,0.243,0.259,0.276,0.293,0.31,0.328,0.347,0.366,0.385,0.404,0.424,0.444,0.465,0.486,0.507,0.529,0.55,0.572,0.595,0.617,0.64,0.663,0.686,0.71,0.733,0.757,0.781,0.805,0.829,0.853,0.878,0.902,0.926,0.951,0.975,1],
        "y+":[0,0.053648,0.21456,0.48263,0.85771,1.3396,1.9279,2.6224,3.4226,4.328,5.3381,6.4523,7.67,8.9902,10.412,11.936,13.559,15.281,17.102,19.019,21.033,23.141,25.342,27.635,30.019,32.492,35.053,37.701,40.432,43.247,46.143,49.118,52.171,55.3,58.503,61.778,65.123,68.536,72.016,75.559,79.164,82.828,86.55,90.327,94.157,98.037,101.97,105.94,109.96,114.02,118.12,122.25,126.42,130.62,134.84,139.1,143.37,147.67,151.99,156.32,160.66,165.02,169.38,173.75,178.12],
        "Umean":[0,0.053639,0.21443,0.48197,0.85555,1.3339,1.9148,2.5939,3.3632,4.2095,5.1133,6.0493,6.9892,7.9052,8.7741,9.579,10.311,10.967,11.55,12.066,12.52,12.921,13.276,13.59,13.87,14.121,14.349,14.557,14.75,14.931,15.101,15.264,15.419,15.569,15.714,15.855,15.993,16.128,16.26,16.389,16.515,16.637,16.756,16.872,16.985,17.094,17.2,17.302,17.4,17.494,17.585,17.672,17.756,17.835,17.911,17.981,18.045,18.103,18.154,18.198,18.235,18.264,18.285,18.297,18.301],
        "dUmean/dy":[178,178,178,178,177,176,175,173,169,163,155,144,131,116,101,87.1,73.9,62.2,52.2,43.8,36.9,31.1,26.4,22.6,19.4,16.9,14.9,13.3,12,10.9,10.1,9.38,8.79,8.29,7.86,7.49,7.19,6.91,6.63,6.35,6.07,5.81,5.58,5.36,5.14,4.92,4.68,4.45,4.23,4.04,3.85,3.66,3.48,3.28,3.06,2.81,2.54,2.25,1.96,1.67,1.35,1.02,0.673,0.33,0],
        "Wmean":[0,0.0000707,0.000283,0.000636,0.00113,0.00176,0.00252,0.00339,0.00435,0.00538,0.00643,0.00751,0.00864,0.00986,0.0112,0.0126,0.0141,0.0156,0.017,0.0181,0.0186,0.0184,0.0176,0.0163,0.0149,0.0135,0.0124,0.0116,0.0107,0.00966,0.00843,0.00695,0.00519,0.00329,0.00145,-0.000284,-0.00177,-0.00292,-0.00377,-0.00445,-0.00497,-0.0054,-0.00594,-0.00681,-0.0082,-0.00996,-0.0119,-0.0139,-0.0163,-0.0191,-0.0225,-0.0263,-0.0306,-0.0354,-0.0405,-0.0455,-0.05,-0.0539,-0.0577,-0.0615,-0.0653,-0.0685,-0.071,-0.0724,-0.0729],
        "dWmean/dy":[0.235,0.235,0.235,0.234,0.234,0.232,0.228,0.22,0.208,0.194,0.179,0.168,0.164,0.164,0.166,0.167,0.162,0.148,0.121,0.076,0.0159,-0.0439,-0.087,-0.107,-0.106,-0.0871,-0.0643,-0.0546,-0.061,-0.0707,-0.0818,-0.0958,-0.108,-0.106,-0.0989,-0.0881,-0.0697,-0.0506,-0.0379,-0.0303,-0.0221,-0.0216,-0.0314,-0.0522,-0.0756,-0.0841,-0.0884,-0.0974,-0.114,-0.136,-0.154,-0.172,-0.196,-0.214,-0.215,-0.199,-0.174,-0.156,-0.155,-0.159,-0.147,-0.118,-0.0788,-0.0387,0],
        "Pmean":[6.2170e-13,-7.3193e-10,-1.5832e-07,-3.7598e-06,-3.3837e-05,-1.7683e-04,-6.5008e-04,-1.8650e-03,-4.4488e-03,-9.2047e-03,-1.7023e-02,-2.8777e-02,-4.5228e-02,-6.6952e-02,-9.4281e-02,-1.2724e-01,-1.6551e-01,-2.0842e-01,-2.5498e-01,-3.0396e-01,-3.5398e-01,-4.0362e-01,-4.5163e-01,-4.9698e-01,-5.3880e-01,-5.7639e-01,-6.0919e-01,-6.3686e-01,-6.5930e-01,-6.7652e-01,-6.8867e-01,-6.9613e-01,-6.9928e-01,-6.9854e-01,-6.9444e-01,-6.8744e-01,-6.7802e-01,-6.6675e-01,-6.5429e-01,-6.4131e-01,-6.2817e-01,-6.1487e-01,-6.0122e-01,-5.8703e-01,-5.7221e-01,-5.5678e-01,-5.4090e-01,-5.2493e-01,-5.0917e-01,-4.9371e-01,-4.7867e-01,-4.6421e-01,-4.5050e-01,-4.3759e-01,-4.2550e-01,-4.1436e-01,-4.0444e-01,-3.9595e-01,-3.8900e-01,-3.8360e-01,-3.7966e-01,-3.7702e-01,-3.7542e-01,-3.7460e-01,-3.7436e-01]
        }
    return dns_dic

class TurbulentChannel(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # top and bottom sides of the channel are no-slip and the other directions are periodic
        wall = np.concatenate((self.boundingBoxIndices['bottom'], self.boundingBoxIndices['top']))
        self.BCs.append(Regularized(tuple(wall.T), self.gridInfo, self.precisionPolicy, 'velocity', np.zeros((wall.shape[0], 3))))
        return

    def initialize_macroscopic_fields(self):
        rho = self.precisionPolicy.cast_to_output(1.0)
        u = self.distributed_array_init((self.nx, self.ny, self.nz, self.dim),
                                         self.precisionPolicy.compute_dtype, init_val=1e-2 * np.random.random((self.nx, self.ny, self.nz, self.dim)))
        u = self.precisionPolicy.cast_to_output(u)
        return rho, u

    def initialize_populations(self, rho, u):
        omegaADE = 1.0
        lattice = LatticeD3Q27(precision)

        kwargs = {'lattice': lattice, 'nx': self.nx, 'ny': self.ny, 'nz': self.nz,  'precision': precision, 'omega': omegaADE, 'vel': u}
        ADE = AdvectionDiffusionBGK(**kwargs)
        ADE.initialize_macroscopic_fields = self.initialize_macroscopic_fields
        print("Initializing the distribution functions using the specified macroscopic fields....")
        f = ADE.run(50000)
        return f

    def get_force(self):
        # define the external force
        force = np.zeros((self.nx, self.ny, self.nz, 3))
        force[..., 0] = Re_tau**2 * visc**2 / h**3
        return self.precisionPolicy.cast_to_output(force)

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho"])
        u = np.array(kwargs["u"])
        timestep = kwargs["timestep"]
        u_prev = kwargs['u_prev']

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        err = np.sum(np.abs(u_old - u_new))
        print("error= {:07.6f}".format(err))

        # mean streamwise velocity in wall units u^+(z)
        uplus = np.mean(u[..., 0], axis=(0,1))/u_tau
        uplus_loglaw = vonKarman_loglaw_wall(yplus)
        dns_dic = get_dns_data()
        plt.clf()
        plt.semilogx(yplus, uplus,'r.', yplus, uplus_loglaw, 'k:', dns_dic['y+'], dns_dic['Umean'], 'b-')
        ax = plt.gca()
        ax.set_xlim([0.1, 300])
        ax.set_ylim([0, 20])
        fname = "uplus_" + str(timestep//10000).zfill(5) + '.pdf'
        plt.savefig(fname, format='pdf')
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        save_fields_vtk(timestep, fields)



if __name__ == "__main__":
    precision = "f64/f64"
    lattice = LatticeD3Q27(precision)

    # h: channel half-width
    h = 50

    # Define channel geometry based on h
    nx = 6*h
    ny = 3*h
    nz = 2*h

    # Define flow regime
    Re_tau = 180
    u_tau = 0.001
    DeltaPlus = Re_tau/h    # DeltaPlus = u_tau / nu * Delta where u_tau / nu = Re_tau/h
    visc = u_tau * h / Re_tau
    omega = 1.0 / (3.0 * visc + 0.5)

    # Wall distance in wall units to be used inside output_data
    zz = np.arange(nz)
    zz = np.minimum(zz, zz.max() - zz)
    yplus = zz * u_tau / visc

    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 500000,
        'print_info_rate': 100000
    }
    sim = turbulentChannel(**kwargs)
    sim.run(10000000)
