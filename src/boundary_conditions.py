import jax.numpy as jnp
from jax import jit, device_count
from functools import partial
import numpy as np
class BoundaryCondition(object):
    """
    Base class for boundary conditions in a LBM simulation.

    This class provides a general structure for implementing boundary conditions. It includes methods for preparing the
    boundary attributes and for applying the boundary condition. Specific boundary conditions should be implemented as
    subclasses of this class, with the `apply` method overridden as necessary.

    Attributes
    ----------
    lattice : Lattice
        The lattice used in the simulation.
    nx:
        The number of nodes in the x direction.
    ny:
        The number of nodes in the y direction.
    nz:
        The number of nodes in the z direction.
    dim : int
        The number of dimensions in the simulation (2 or 3).
    precision_policy : PrecisionPolicy
        The precision policy used in the simulation.
    indices : array-like
        The indices of the boundary nodes.
    name : str or None
        The name of the boundary condition. This should be set in subclasses.
    isSolid : bool
        Whether the boundary condition is for a solid boundary. This should be set in subclasses.
    isDynamic : bool
        Whether the boundary condition is dynamic (changes over time). This should be set in subclasses.
    needsExtraConfiguration : bool
        Whether the boundary condition requires extra configuration. This should be set in subclasses.
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. This should be set in subclasses.
    """

    def __init__(self, indices, gridInfo, precision_policy):
        self.lattice = gridInfo["lattice"]
        self.nx = gridInfo["nx"]
        self.ny = gridInfo["ny"]
        self.nz = gridInfo["nz"]
        self.dim = gridInfo["dim"]
        self.precisionPolicy = precision_policy
        self.indices = indices
        self.name = None
        self.isSolid = False
        self.isDynamic = False
        self.needsExtraConfiguration = False
        self.implementationStep = "PostStreaming"

    def create_local_bitmask_and_normal_arrays(self, connectivity_bitmask):

        """
        Creates local bitmask and normal arrays for the boundary condition.

        Parameters
        ----------
        connectivity_bitmask : array-like
            The connectivity bitmask for the lattice.

        Returns
        -------
        None

        Notes
        -----
        This method creates local bitmask and normal arrays for the boundary condition based on the connectivity bitmask.
        If the boundary condition requires extra configuration, the `configure` method is called.
        """

        if self.needsExtraConfiguration:
            boundaryBitmask = self.get_boundary_bitmask(connectivity_bitmask)
            self.configure(boundaryBitmask)
            self.needsExtraConfiguration = False

        boundaryBitmask = self.get_boundary_bitmask(connectivity_bitmask)
        self.normals = self.get_normals(boundaryBitmask)
        self.imissing, self.iknown = self.get_missing_indices(boundaryBitmask)
        self.imissingBitmask, self.iknownBitmask, self.imiddleBitmask = self.get_missing_bitmask(boundaryBitmask)

        return

    def get_boundary_bitmask(self, connectivity_bitmask):  
        """
        Add jax.device_count() to the self.indices in x-direction, and 1 to the self.indices other directions
        This is to make sure the boundary condition is applied to the correct nodes as connectivity_bitmask is
        expanded by (jax.device_count(), 1, 1)

        Parameters
        ----------
        connectivity_bitmask : array-like
            The connectivity bitmask for the lattice.
        
        Returns
        -------
        boundaryBitmask : array-like
        """   
        shifted_indices = np.array(self.indices)
        shifted_indices[0] += device_count()
        shifted_indices[1:] += 1
        # Convert back to tuple
        shifted_indices = tuple(shifted_indices)
        boundaryBitmask = np.array(connectivity_bitmask[shifted_indices])

        return boundaryBitmask

    def configure(self, boundaryBitmask):
        """
        Configures the boundary condition.

        Parameters
        ----------
        boundaryBitmask : array-like
            The connectivity bitmask for the boundary voxels.

        Returns
        -------
        None

        Notes
        -----
        This method should be overridden in subclasses if the boundary condition requires extra configuration.
        """
        return

    @partial(jit, static_argnums=(0, 3), inline=True)
    def prepare_populations(self, fout, fin, implementation_step):
        """
        Prepares the distribution functions for the boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The incoming distribution functions.
        fin : jax.numpy.ndarray
            The outgoing distribution functions.
        implementation_step : str
            The step in the lattice Boltzmann method algorithm at which the preparation is applied.

        Returns
        -------
        jax.numpy.ndarray
            The prepared distribution functions.

        Notes
        -----
        This method should be overridden in subclasses if the boundary condition requires preparation of the distribution functions during post-collision or post-streaming. See ExtrapolationBoundaryCondition for an example.
        """   
        return fout

    def get_normals(self, boundaryBitmask):
        """
        Calculates the normal vectors at the boundary nodes.

        Parameters
        ----------
        boundaryBitmask : array-like
            The boundary bitmask for the lattice.

        Returns
        -------
        array-like
            The normal vectors at the boundary nodes.

        Notes
        -----
        This method calculates the normal vectors by dotting the boundary bitmask with the main lattice directions.
        """
        main_c = self.lattice.c.T[self.lattice.main_indices]
        m = boundaryBitmask[..., self.lattice.main_indices]
        normals = -np.dot(m, main_c)
        return normals

    def get_missing_indices(self, boundaryBitmask):
        """
        Returns two int8 arrays the same shape as boundaryBitmask. The non-zero entries of these arrays indicate missing
        directions that require BCs (imissing) as well as their corresponding opposite directions (iknown).

        Parameters
        ----------
        boundaryBitmask : array-like
            The boundary bitmask for the lattice.

        Returns
        -------
        tuple of array-like
            The missing and known indices for the boundary condition.

        Notes
        -----
        This method calculates the missing and known indices based on the boundary bitmask. The missing indices are the
        non-zero entries of the boundary bitmask, and the known indices are their corresponding opposite directions.
        """

        # Find imissing, iknown 1-to-1 corresponding indices
        # Note: the "zero" index is used as default value here and won't affect BC computations
        nbd = len(self.indices[0])
        imissing = np.vstack([np.arange(self.lattice.q, dtype='uint8')] * nbd)
        iknown = np.vstack([self.lattice.opp_indices] * nbd)
        imissing[~boundaryBitmask] = 0
        iknown[~boundaryBitmask] = 0
        return imissing, iknown

    def get_missing_bitmask(self, boundaryBitmask):
        """
        Returns three boolean arrays the same shape as boundaryBitmask.
        Note: these boundary bitmasks are useful for reduction (eg. summation) operators of selected q-directions.

        Parameters
        ----------
        boundaryBitmask : array-like
            The boundary bitmask for the lattice.

        Returns
        -------
        tuple of array-like
            The missing, known, and middle bitmasks for the boundary condition.

        Notes
        -----
        This method calculates the missing, known, and middle bitmasks based on the boundary bitmask. The missing bitmask
        is the boundary bitmask, the known bitmask is the opposite directions of the missing bitmask, and the middle bitmask
        is the directions that are neither missing nor known.
        """
        # Find Bitmasks for imissing, iknown and imiddle
        imissingBitmask = boundaryBitmask
        iknownBitmask = imissingBitmask[:, self.lattice.opp_indices]
        imiddleBitmask = ~(imissingBitmask | iknownBitmask)
        return imissingBitmask, iknownBitmask, imiddleBitmask

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        None

        Notes
        -----
        This method should be overridden in subclasses to implement the specific boundary condition. The method should
        modify the output distribution functions in place to apply the boundary condition.
        """
        pass

    @partial(jit, static_argnums=(0,))
    def equilibrium(self, rho, u):
        """
        Compute equilibrium distribution function.

        Parameters
        ----------
        rho : jax.numpy.ndarray
            The density at each node in the lattice.
        u : jax.numpy.ndarray
            The velocity at each node in the lattice.

        Returns
        -------
        jax.numpy.ndarray
            The equilibrium distribution function at each node in the lattice.

        Notes
        -----
        This method computes the equilibrium distribution function based on the density and velocity. The computation is
        performed in the compute precision specified by the precision policy. The result is not cast to the output precision as
        this is function is used inside other functions that require the compute precision.
        """
        rho, u = self.precisionPolicy.cast_to_compute((rho, u))
        c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
        cu = 3.0 * jnp.dot(u, c)
        usqr = 1.5 * jnp.sum(u**2, axis=-1, keepdims=True)
        feq = rho * self.lattice.w * (1.0 + 1.0 * cu + 0.5 * cu**2 - usqr)

        return feq

    @partial(jit, static_argnums=(0,))
    def momentum_flux(self, fneq):
        """
        Compute the momentum flux.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            The non-equilibrium distribution function at each node in the lattice.

        Returns
        -------
        jax.numpy.ndarray
            The momentum flux at each node in the lattice.

        Notes
        -----
        This method computes the momentum flux by dotting the non-equilibrium distribution function with the lattice
        direction vectors.
        """
        return jnp.dot(fneq, self.lattice.cc)

    @partial(jit, static_argnums=(0,))
    def momentum_exchange_force(self, f_poststreaming, f_postcollision):
        """
        Using the momentum exchange method to compute the boundary force vector exerted on the solid geometry
        based on [1] as described in [3]. Ref [2] shows how [1] is applicable to curved geometries only by using a
        bounce-back method (e.g. Bouzidi) that accounts for curved boundaries.
        NOTE: this function should be called after BC's are imposed.
        [1] A.J.C. Ladd, Numerical simulations of particular suspensions via a discretized Boltzmann equation.
            Part 2 (numerical results), J. Fluid Mech. 271 (1994) 311-339.
        [2] R. Mei, D. Yu, W. Shyy, L.-S. Luo, Force evaluation in the lattice Boltzmann method involving
            curved geometry, Phys. Rev. E 65 (2002) 041203.
        [3] Caiazzo, A., & Junk, M. (2008). Boundary forces in lattice Boltzmann: Analysis of momentum exchange
            algorithm. Computers & Mathematics with Applications, 55(7), 1415-1423.

        Parameters
        ----------
        f_poststreaming : jax.numpy.ndarray
            The post-streaming distribution function at each node in the lattice.
        f_postcollision : jax.numpy.ndarray
            The post-collision distribution function at each node in the lattice.

        Returns
        -------
        jax.numpy.ndarray
            The force exerted on the solid geometry at each boundary node.

        Notes
        -----
        This method computes the force exerted on the solid geometry at each boundary node using the momentum exchange method. 
        The force is computed based on the post-streaming and post-collision distribution functions. This method
        should be called after the boundary conditions are imposed.
        """
        c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        phi = f_postcollision[self.indices][bindex, self.iknown] + \
              f_poststreaming[self.indices][bindex, self.imissing]
        force = jnp.sum(c[:, self.iknown] * phi, axis=-1).T
        return force

class BounceBack(BoundaryCondition):
    """
    Bounce-back boundary condition for a lattice Boltzmann method simulation.

    This class implements a full-way bounce-back boundary condition, where particles hitting the boundary are reflected
    back in the direction they came from. The boundary condition is applied after the collision step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "BounceBackFullway".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostCollision".
    """
    def __init__(self, indices, gridInfo, precision_policy):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "BounceBackFullway"
        self.implementationStep = "PostCollision"

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the bounce-back boundary condition by reflecting the input distribution functions at the
        boundary nodes in the opposite direction.
        """

        return fin[self.indices][..., self.lattice.opp_indices]

class BounceBackMoving(BoundaryCondition):
    """
    Moving bounce-back boundary condition for a lattice Boltzmann method simulation.

    This class implements a moving bounce-back boundary condition, where particles hitting the boundary are reflected
    back in the direction they came from, with an additional velocity due to the movement of the boundary. The boundary
    condition is applied after the collision step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "BounceBackFullwayMoving".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostCollision".
    isDynamic : bool
        Whether the boundary condition is dynamic (changes over time). For this class, it is True.
    update_function : function
        A function that updates the boundary condition. For this class, it is a function that updates the boundary
        condition based on the current time step. The signature of the function is `update_function(time) -> (indices, vel)`,

    """
    def __init__(self, gridInfo, precision_policy, update_function=None):
        # We get the indices at time zero to pass to the parent class for initialization
        indices, _ = update_function(0)
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "BounceBackFullwayMoving"
        self.implementationStep = "PostCollision"
        self.isDynamic = True
        self.update_function = jit(update_function)

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin, time):
        """
        Applies the moving bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.
        time : int
            The current time step.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        """
        indices, vel = self.update_function(time)
        c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
        cu = 6.0 * self.lattice.w * jnp.dot(vel, c)
        return fout.at[indices].set(fin[indices][..., self.lattice.opp_indices] - cu)


class BounceBackHalfway(BoundaryCondition):
    """
    Halfway bounce-back boundary condition for a lattice Boltzmann method simulation.

    This class implements a halfway bounce-back boundary condition. The boundary condition is applied after
    the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "BounceBackHalfway".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    needsExtraConfiguration : bool
        Whether the boundary condition needs extra configuration before it can be applied. For this class, it is True.
    isSolid : bool
        Whether the boundary condition represents a solid boundary. For this class, it is True.
    vel : array-like
        The prescribed value of velocity vector for the boundary condition. No-slip BC is assumed if vel=None (default).
    """
    def __init__(self, indices, gridInfo, precision_policy, vel=None):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "BounceBackHalfway"
        self.implementationStep = "PostStreaming"
        self.needsExtraConfiguration = True
        self.isSolid = True
        self.vel = vel

    def configure(self, boundaryBitmask):
        """
        Configures the boundary condition.

        Parameters
        ----------
        boundaryBitmask : array-like
            The connectivity bitmask for the boundary voxels.

        Returns
        -------
        None

        Notes
        -----
        This method performs an index shift for the halfway bounce-back boundary condition. It updates the indices of
        the boundary nodes to be the indices of fluid nodes adjacent of the solid nodes.
        """
        # Perform index shift for halfway BB.
        shiftDir = ~boundaryBitmask[:, self.lattice.opp_indices]
        idx = np.array(self.indices).T
        idx_trg = []
        for i in range(self.lattice.q):
            idx_trg.append(idx[shiftDir[:, i], :] + self.lattice.c[:, i])
        indices_new = np.unique(np.vstack(idx_trg), axis=0)
        self.indices = tuple(indices_new.T)
        return

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the halfway bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        """
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fout[self.indices]
        if self.vel is not None:
            c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
            cu = 6.0 * self.lattice.w * jnp.dot(self.vel, c)
            fbd = fbd.at[bindex, self.imissing].set(fin[self.indices][bindex, self.iknown] - cu[bindex, self.iknown])
        else:
            fbd = fbd.at[bindex, self.imissing].set(fin[self.indices][bindex, self.iknown])

        return fbd
    
class EquilibriumBC(BoundaryCondition):
    """
    Equilibrium boundary condition for a lattice Boltzmann method simulation.

    This class implements an equilibrium boundary condition, where the distribution function at the boundary nodes is
    set to the equilibrium distribution function. The boundary condition is applied after the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "EquilibriumBC".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    out : jax.numpy.ndarray
        The equilibrium distribution function at the boundary nodes.
    """

    def __init__(self, indices, gridInfo, precision_policy, rho, u):
        super().__init__(indices, gridInfo, precision_policy)
        self.out = self.precisionPolicy.cast_to_output(self.equilibrium(rho, u))
        self.name = "EquilibriumBC"
        self.implementationStep = "PostStreaming"

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the equilibrium boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the equilibrium boundary condition by setting the output distribution functions at the
        boundary nodes to the equilibrium distribution function.
        """
        return self.out

class DoNothing(BoundaryCondition):
    def __init__(self, indices, gridInfo, precision_policy):
        """
        Do-nothing boundary condition for a lattice Boltzmann method simulation.

        This class implements a do-nothing boundary condition, where no action is taken at the boundary nodes. The boundary
        condition is applied after the streaming step.

        Attributes
        ----------
        name : str
            The name of the boundary condition. For this class, it is "DoNothing".
        implementationStep : str
            The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
            it is "PostStreaming".

        Notes
        -----
        This boundary condition enforces skipping of streaming altogether as it sets post-streaming equal to post-collision
        populations (so no streaming at this BC voxels). The problem with returning post-streaming values or "fout[self.indices]
        is that the information that exit the domain on the opposite side of this boundary, would "re-enter". This is because
        we roll the entire array and so the boundary condition acts like a one-way periodic BC. If EquilibriumBC is used as
        the BC for that opposite boundary, then the rolled-in values are taken from the initial condition at equilibrium.
        Otherwise if ZouHe is used for example the simulation looks like a run-down simulation at low-Re. The opposite boundary
        may be even a wall (consider pipebend example). If we correct imissing directions and assign "fin", this method becomes
        much less stable and also one needs to correctly take care of corner cases.
        """
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "DoNothing"
        self.implementationStep = "PostStreaming"


    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the do-nothing boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the do-nothing boundary condition by simply returning the input distribution functions at the
        boundary nodes.
        """
        return fin[self.indices]


class ZouHe(BoundaryCondition):
    """
    Zou-He boundary condition for a lattice Boltzmann method simulation.

    This class implements the Zou-He boundary condition, which is a non-equilibrium bounce-back boundary condition.
    It can be used to set inflow and outflow boundary conditions with prescribed pressure or velocity.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "ZouHe".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    type : str
        The type of the boundary condition. It can be either 'velocity' for a prescribed velocity boundary condition,
        or 'pressure' for a prescribed pressure boundary condition.
    prescribed : float or array-like
        The prescribed values for the boundary condition. It can be either the prescribed velocities for a 'velocity'
        boundary condition, or the prescribed pressures for a 'pressure' boundary condition.

    References
    ----------
    Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model.
    Physics of Fluids, 9(6), 1591-1598. doi:10.1063/1.869307
    """
    def __init__(self, indices, gridInfo, precision_policy, type, prescribed):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "ZouHe"
        self.implementationStep = "PostStreaming"
        self.type = type
        self.prescribed = prescribed
        self.needsExtraConfiguration = True

    def configure(self, boundaryBitmask):
        """
        Correct boundary indices to ensure that only voxelized surfaces with normal vectors along main cartesian axes
        are assigned this type of BC.
        """
        nv = np.dot(self.lattice.c, ~boundaryBitmask.T)
        corner_voxels = np.count_nonzero(nv, axis=0) > 1
        # removed_voxels = np.array(self.indices)[:, corner_voxels]
        self.indices = tuple(np.array(self.indices)[:, ~corner_voxels])
        self.prescribed = self.prescribed[~corner_voxels]
        return

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_vel(self, fpop, rho):
        """
        Calculate velocity based on the prescribed pressure/density (Zou/He BC)
        """
        unormal = -1. + 1. / rho * (jnp.sum(fpop[self.indices] * self.imiddleBitmask, axis=1) +
                               2. * jnp.sum(fpop[self.indices] * self.iknownBitmask, axis=1))

        # Return the above unormal as a normal vector which sets the tangential velocities to zero
        vel = unormal[:, None] * self.normals
        return vel

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_rho(self, fpop, vel):
        """
        Calculate density based on the prescribed velocity (Zou/He BC)
        """
        unormal = np.sum(self.normals*vel, axis=1)

        rho = (1.0/(1.0 + unormal))[..., None] * (jnp.sum(fpop[self.indices] * self.imiddleBitmask, axis=1, keepdims=True) +
                                  2.*jnp.sum(fpop[self.indices] * self.iknownBitmask, axis=1, keepdims=True))
        return rho

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_equilibrium(self, fpop):
        """
        This is the ZouHe method of calculating the missing macroscopic variables at the boundary.
        """
        if self.type == 'velocity':
            vel = self.prescribed
            rho = self.calculate_rho(fpop, vel)
        elif self.type == 'pressure':
            rho = self.prescribed
            vel = self.calculate_vel(fpop, rho)
        else:
            raise ValueError(f"type = {self.type} not supported! Use \'pressure\' or \'velocity\'.")

        # compute feq at the boundary
        feq = self.equilibrium(rho, vel)
        return feq

    @partial(jit, static_argnums=(0,), inline=True)
    def bounceback_nonequilibrium(self, fpop, feq):
        """
        Calculate unknown populations using bounce-back of non-equilibrium populations
        a la original Zou & He formulation
        """
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fpop[self.indices]
        fknown = fpop[self.indices][bindex, self.iknown] + feq[bindex, self.imissing] - feq[bindex, self.iknown]
        fbd = fbd.at[bindex, self.imissing].set(fknown)
        return fbd

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, _):
        """
        Applies the Zou-He boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        _ : jax.numpy.ndarray
            The input distribution functions. This is not used in this method.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the Zou-He boundary condition by first computing the equilibrium distribution functions based
        on the prescribed values and the type of boundary condition, and then setting the unknown distribution functions
        based on the non-equilibrium bounce-back method. 
        Tangential velocity is not ensured to be zero by adding transverse contributions based on
        Hecth & Harting (2010) (doi:10.1088/1742-5468/2010/01/P01018) as it caused numerical instabilities at higher
        Reynolds numbers. One needs to use "Regularized" BC at higher Reynolds.
        """
        # compute the equilibrium based on prescribed values and the type of BC
        feq = self.calculate_equilibrium(fout)

        # set the unknown f populations based on the non-equilibrium bounce-back method
        fbd = self.bounceback_nonequilibrium(fout, feq)


        return fbd

class Regularized(ZouHe):
    """
    Regularized boundary condition for a lattice Boltzmann method simulation.

    This class implements the regularized boundary condition, which is a non-equilibrium bounce-back boundary condition
    with additional regularization. It can be used to set inflow and outflow boundary conditions with prescribed pressure
    or velocity.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "Regularized".
    Qi : numpy.ndarray
        The Qi tensor, which is used in the regularization of the distribution functions.

    References
    ----------
    Latt, J. (2007). Hydrodynamic limit of lattice Boltzmann equations. PhD thesis, University of Geneva.
    Latt, J., Chopard, B., Malaspinas, O., Deville, M., & Michler, A. (2008). Straight velocity boundaries in the
    lattice Boltzmann method. Physical Review E, 77(5), 056703. doi:10.1103/PhysRevE.77.056703
    """

    def __init__(self, indices, gridInfo, precision_policy, type, prescribed):
        super().__init__(indices, gridInfo, precision_policy, type, prescribed)
        self.name = "Regularized"
        #TODO for Hesam: check to understand why corner cases cause instability here.
        # self.needsExtraConfiguration = False
        self.construct_symmetric_lattice_moment()

    def construct_symmetric_lattice_moment(self):
        """
        Construct the symmetric lattice moment Qi.

        The Qi tensor is used in the regularization of the distribution functions. It is defined as Qi = cc - cs^2*I,
        where cc is the tensor of lattice velocities, cs is the speed of sound, and I is the identity tensor.
        """
        Qi = self.lattice.cc
        if self.dim == 3:
            diagonal = (0, 3, 5)
            offdiagonal = (1, 2, 4)
        elif self.dim == 2:
            diagonal = (0, 2)
            offdiagonal = (1,)
        else:
            raise ValueError(f"dim = {self.dim} not supported")

        # Qi = cc - cs^2*I
        Qi = Qi.at[:, diagonal].set(self.lattice.cc[:, diagonal] - 1./3.)

        # multiply off-diagonal elements by 2 because the Q tensor is symmetric
        Qi = Qi.at[:, offdiagonal].set(self.lattice.cc[:, offdiagonal] * 2.0)

        self.Qi = Qi.T
        return

    @partial(jit, static_argnums=(0,), inline=True)
    def regularize_fpop(self, fpop, feq):
        """
        Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.

        Parameters
        ----------
        fpop : jax.numpy.ndarray
            The distribution functions.
        feq : jax.numpy.ndarray
            The equilibrium distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The regularized distribution functions.
        """

        # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
        f_neq = fpop - feq
        PiNeq = self.momentum_flux(f_neq)
        # PiNeq = self.momentum_flux(fpop) - self.momentum_flux(feq)

        # Compute double dot product Qi:Pi1
        # QiPi1 = np.zeros_like(fpop)
        # Pi1 = PiNeq
        # QiPi1 = jnp.dot(Qi, Pi1)
        QiPi1 = jnp.dot(PiNeq, self.Qi)

        # assign all populations based on eq 45 of Latt et al (2008)
        # fneq ~ f^1
        fpop1 = 9. / 2. * self.lattice.w[None, :] * QiPi1
        fpop_regularized = feq + fpop1

        return fpop_regularized

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, _):
        """
        Applies the regularized boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        _ : jax.numpy.ndarray
            The input distribution functions. This is not used in this method.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the regularized boundary condition by first computing the equilibrium distribution functions based
        on the prescribed values and the type of boundary condition, then setting the unknown distribution functions
        based on the non-equilibrium bounce-back method, and finally regularizing the distribution functions.
        """

        # compute the equilibrium based on prescribed values and the type of BC
        feq = self.calculate_equilibrium(fout)

        # set the unknown f populations based on the non-equilibrium bounce-back method
        fbd = self.bounceback_nonequilibrium(fout, feq)

        # Regularize the boundary fpop
        fbd = self.regularize_fpop(fbd, feq)
        return fbd


class ExtrapolationOutflow(BoundaryCondition):
    """
    Extrapolation outflow boundary condition for a lattice Boltzmann method simulation.

    This class implements the extrapolation outflow boundary condition, which is a type of outflow boundary condition
    that uses extrapolation to avoid strong wave reflections.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "ExtrapolationOutflow".
    sound_speed : float
        The speed of sound in the simulation.

    References
    ----------
    Geier, M., Schönherr, M., Pasquali, A., & Krafczyk, M. (2015). The cumulant lattice Boltzmann equation in three
    dimensions: Theory and validation. Computers & Mathematics with Applications, 70(4), 507–547.
    doi:10.1016/j.camwa.2015.05.001.
    """

    def __init__(self, indices, gridInfo, precision_policy):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "ExtrapolationOutflow"
        self.needsExtraConfiguration = True
        self.sound_speed = 1./jnp.sqrt(3.)

    def configure(self, boundaryBitmask):
        """
        Configure the boundary condition by finding neighbouring voxel indices.

        Parameters
        ----------
        boundaryBitmask : np.ndarray
            The connectivity bitmask for the boundary voxels.
        """        
        shiftDir = ~boundaryBitmask[:, self.lattice.opp_indices]
        idx = np.array(self.indices).T
        idx_trg = []
        for i in range(self.lattice.q):
            idx_trg.append(idx[shiftDir[:, i], :] + self.lattice.c[:, i])
        indices_nbr = np.unique(np.vstack(idx_trg), axis=0)
        self.indices_nbr = tuple(indices_nbr.T)

        return

    @partial(jit, static_argnums=(0, 3), inline=True)
    def prepare_populations(self, fout, fin, implementation_step):
        """
        Prepares the distribution functions for the boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The incoming distribution functions.
        fin : jax.numpy.ndarray
            The outgoing distribution functions.
        implementation_step : str
            The step in the lattice Boltzmann method algorithm at which the preparation is applied.

        Returns
        -------
        jax.numpy.ndarray
            The prepared distribution functions.

        Notes
        -----
        Because this function is called "PostCollision", f_poststreaming refers to previous time step or t-1
        """
        f_postcollision = fout
        f_poststreaming = fin
        if implementation_step == 'PostStreaming':
            return f_postcollision
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fps_bdr = f_poststreaming[self.indices]
        fps_nbr = f_poststreaming[self.indices_nbr]
        fpc_bdr = f_postcollision[self.indices]
        fpop = fps_bdr[bindex, self.imissing]
        fpop_neighbour = fps_nbr[bindex, self.imissing]
        fpop_extrapolated = self.sound_speed * fpop_neighbour + (1. - self.sound_speed) * fpop

        # Use the iknown directions of f_postcollision that leave the domain during streaming to store the BC data
        fpc_bdr = fpc_bdr.at[bindex, self.iknown].set(fpop_extrapolated)
        f_postcollision = f_postcollision.at[self.indices].set(fpc_bdr)
        return f_postcollision

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the extrapolation outflow boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        """
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fout[self.indices]
        fbd = fbd.at[bindex, self.imissing].set(fin[self.indices][bindex, self.iknown])
        return fbd
