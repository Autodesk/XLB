# Introduction to XLB

Welcome to XLB! This page will introduce you to the core concepts of the library, the numerical method it's based on, and the general philosophy behind its design.

## What is XLB?

XLB is a modern, high-performance library for computational fluid dynamics (CFD) based on the **Lattice Boltzmann Method (LBM)**. It is written in Python and designed with two primary goals in mind:

1.  **Ease of Use:** XLB provides a high-level, object-oriented API that makes setting up complex fluid simulations intuitive and straightforward. The goal is to let you focus on the physics of your problem, not on boilerplate code.
2.  **Flexibility and Performance:** While being easy to use, XLB is built for serious research. It is designed to be easily extendable with new physical models, and it leverages modern numerical backends (like JAX or Numba) to achieve performance comparable to compiled languages, all from within the comfort of Python.

It is an ideal tool for students learning CFD, researchers prototyping new models, and anyone who needs a powerful and flexible fluid simulation toolkit.

## A Brief Primer on the Lattice Boltzmann Method

To understand how to use XLB, it's helpful to understand the basics of the Lattice Boltzmann Method. Unlike traditional CFD solvers that directly discretize the macroscopic Navier-Stokes equations, LBM is a **mesoscopic method**. It simulates fluid flow by modeling the collective behavior of fluid particles.

The core variable in LBM is the **particle distribution function**, denoted as $f_i(\vec{x}, t)$. This function represents the population of particles at a lattice node $\vec{x}$ at time $t$, moving with a discrete velocity $\vec{e}_i$. The collection of these discrete velocities (e.g., 9 in 2D for the D2Q9 model) forms a `Velocity Set`.

The entire LBM algorithm elegantly evolves these particle populations through two simple steps in each time iteration:

1.  **Collision:** At each lattice node, the particle populations interact with each other. This interaction, or "collision," causes the distribution functions $f_i$ to relax towards a local **equilibrium distribution**, $f_i^{eq}$. This equilibrium state is a function of the macroscopic fluid properties like density ($\rho$) and velocity ($\vec{u}$), which are calculated directly from the particle distributions. The most common collision model is the Bhatnagar-Gross-Krook (BGK) operator, which simplifies this process to:

    $$f_i^* = f_i - \frac{1}{\tau} (f_i - f_i^{eq})$$

    where $f_i^*$ is the post-collision state and $\tau$ is the relaxation time, which controls the fluid's viscosity.

2.  **Streaming:** After collision, the particle populations propagate, or "stream," to their nearest neighbors in the direction of their velocity $\vec{e}_i$.

    $$f_i(\vec{x} + \vec{e}_i \Delta t, t + \Delta t) = f_i^*(\vec{x}, t)$$

By repeating these two simple, local steps, the complex, non-linear behavior of fluid flow described by the Navier-Stokes equations emerges automatically.

Finally, the macroscopic fluid variables are recovered at any point by taking moments of the distribution functions:
* **Density:** $\rho = \sum_i f_i$
* **Momentum:** $\rho\vec{u} = \sum_i f_i \vec{e}_i$

## The XLB Philosophy: Mapping Concepts to Code

XLB is designed to make its code structure directly reflect the concepts of the LBM. When you build a simulation, you are essentially assembling Python objects that represent each part of the method.

* **The Lattice (`Grid`, `Velocity Set`):** You first define the simulation domain by creating a `Grid` object and choosing a `Velocity Set` (e.g., D2Q9 for 2D simulations).

* **The Physics (`Equilibrium`, `Collision`, `Stream`):** The "rules" of the simulation are encapsulated in objects. The mathematical form of the equilibrium distribution $f_i^{eq}$ is handled by the `Equilibrium` module, the collision process is defined by an `Collision`, and the stepper process is defined by a `Stream`.

* **Boundaries and Forces (`BoundaryCondition`, `Force`):** Physical boundaries (like walls) and body forces (like gravity) are not part of the core LBM algorithm but are added as distinct `BoundaryCondition` and `Force` objects that interact with the particle distributions at specific nodes.

* **The Engine (`Stepper`):** The main simulation loop, which repeatedly calls the collision and streaming steps, is controlled by a `Stepper` object. You initialize it with your setup and tell it how many steps to run.

* **The Results (`Macroscopic`):** To get useful data out of your simulation, you use functions from the `Macroscopic` module to compute density, velocity, and other quantities from the `Distribution` object.

## Next Steps

Now that you have a conceptual overview, you're ready to get started!

* **Installation:** If you haven't already, head to the [**Installation**](./installation.md) page.
* **Dive into Code:** The best way to learn is by doing. Follow our [**Getting Started: Your First Simulation**](./getting_started.md) tutorial to build and run a simple simulation from scratch.