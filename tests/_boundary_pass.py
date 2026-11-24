"""Shared helper for the boundary-condition tests: run the boundary pass of the time step (LBMBase.prepare_streaming)."""

import jax
import jax.numpy as jnp

from jax_lab.core.base import LBMBase


def boundary_pass(sim, f_postcollision, f_poststreaming, component=0):
    """
    Apply a simulation's boundary conditions the way its time step does, to given populations of one component.

    Parameters
    ----------
    sim (LBMBase): The simulation (single-phase or Multiphase).

    f_postcollision (jax.numpy.ndarray): Post-collision populations (..., q), before the PostCollision boundary conditions.

    f_poststreaming (jax.numpy.ndarray): The populations they were collided from (the input of the PostCollision boundary
        conditions).

    component (int): The component, for Multiphase.

    Returns
    -------
    f_postcollision (jax.numpy.ndarray): The post-collision populations after the PostCollision boundary conditions.

    f_poststreaming (jax.numpy.ndarray): Their streamed populations after the PostStreaming boundary conditions.
    """
    bc_data = sim.streaming_bc_data[component] if isinstance(sim.streaming_bc_data, list) else sim.streaming_bc_data

    @jax.jit
    def run(f_postcollision, f_poststreaming):
        previous = LBMBase.state_from_populations(sim, f_poststreaming)
        state, postcollision_fields = sim._prepare_streaming(f_postcollision, previous, 0, bc_data)
        return jnp.stack(postcollision_fields, axis=-1), sim.stream(state)

    return run(f_postcollision, f_poststreaming)


def reference_poststreaming(sim, bc, f_postcollision):
    """
    Ground truth for one PostStreaming boundary condition: the streamed populations with the boundary condition's own
    global-index prepare_populations and apply.
    """
    fout = sim.streaming(f_postcollision)
    fout = bc.prepare_populations(fout, f_postcollision, "PostStreaming")
    return fout.at[bc.indices].set(bc.apply(fout, f_postcollision))
