# Multiphase (Volume-of-Fluid)

## Free droplets: momentum conservation and moving interfaces

For free droplets (not touching a wall) that move through the grid, for example settling or colliding droplets, set
`consistent_link_stress=True` and `zero_net_capillary_force=True`:

- **Consistent link stress.** The populations rebuilt at the interface links then carry the same viscous part that is returned to the
  shared interface velocity, so the link momentum is handed back exactly. Without it, a droplet with surface tension moving through the
  grid gains or loses momentum (25 % in 1e4 steps for a 40-cell droplet carried by the gas). The rebuild is unstable when the shear
  relaxation rate of either phase approaches 2 (low viscosity, e.g. high-Reynolds-number impacts), so it is off by default; it was tested
  stable up to s_nu = 1.43.

- **Net capillary force.** The curvature of a sharp volume fraction on a fixed grid has an error that depends on where the droplet sits
  within its cells. Summed over the surface it leaves a net capillary force, which pins slow droplets to the lattice (a droplet carried
  by the gas at 1e-4 oscillates in place) and breaks momentum conservation. Each droplet's net capillary force is removed exactly by a
  smooth (normal-direction) correction of its curvature. Droplets are labeled at initialization by connected liquid and interface cells,
  and the labels follow the interface; merged droplets take the smaller label.
- **Curvature model** (`capillary_curvature_degree`, default 3). The local curvature also carries a fine error pattern tied to the grid.
  On a droplet at rest the liquid balances it once; on a moving droplet it shakes the liquid and grows until the droplet blows up (after
  about 0.7 cells of travel for a 40-cell droplet). Each droplet's curvature is therefore fitted by a low-order polynomial in the
  interface normal: on a near-spherical droplet this keeps the mean curvature and the shape modes up to l = degree and drops the
  grid-locked error. A per-droplet uniform shift, relaxed over time (`capillary_shift_relaxation`, default 0.02), keeps the Laplace pressure
  of the local curvature. `capillary_curvature_degree=0` keeps the local curvature.

Measured for a 40-cell water droplet in air (density ratio 1/829, D3Q19 MRT, 160³):

| Test | Default | `consistent_link_stress=True`, `zero_net_capillary_force=True` (degree 3) |
|---|---|---|
| Droplet carried by the gas at 1e-4 | pinned (0.04 cells in 1e4 steps) | moves with the gas (0.79 of 0.80 cells in 8000 steps), momentum within 1 % |
| Static droplet, Laplace pressure | within 0.2 % | within 0.01 % |
| l = 2 oscillation (Lamb) | period 1.12x, damping 3.3x | period 1.08x, damping 1.23x |

Limits:

- Free droplets only: a droplet on a wall carries a real net capillary force, taken up by the wall at the contact line.
- `consistent_link_stress` needs s_nu well below 2 in both phases (see above).
- The curvature fit assumes a roughly convex droplet. Once two droplets merge, the fit of the merged label is no longer meaningful:
  stop shortly after contact, or continue with `capillary_curvature_degree=0`.
- The per-droplet work runs on the interface band; it costs about 7 % of the throughput at 256³ and 15 % at 160³.

::: jax_lab.core.vof.multiphase.VoF

::: jax_lab.core.vof.multiphase.VoFBGK

::: jax_lab.core.vof.multiphase.VoFMRT

::: jax_lab.core.vof.boundary_conditions.VoFOpenBoundary
