# 🔄 StreamingOperator & CollisionOperator

In Lattice Boltzmann Method (LBM) simulations, the two core steps are **Streaming** and **Collision**. These operations define how distribution functions move across the grid and how they interact locally.

The following operators abstract these steps:

- `Stream`: handles propagation (pull-based streaming)
- `Collision`: handles local distribution interactions (e.g., BGK)

---

## 🚀 Stream (Base Class)

The `Stream` operator performs the **streaming step** by pulling values from neighboring grid points, depending on the velocity set.

### 📌 Purpose

- Implements the **pull scheme** of LBM streaming.
- Ensures support for both 2D and 3D simulations.
- Compatible with **JAX** and **Warp** backends.

### 🧩 Key Properties

| Property               | Description                                               |
|------------------------|-----------------------------------------------------------|
| `implementation_step`  | `STREAMING`                                               |
| `backend_support`      | JAX and Warp                                              |
| `supports_auxiliary_data` | ❌ No                                                  |

### ⚙️ JAX Implementation

```python
@jit
def jax_implementation(self, f):
    def _streaming_jax_i(f, c):
        return jnp.roll(f, (c[0], c[1]), axis=(0, 1))  # for 2D

    return vmap(_streaming_jax_i, in_axes=(0, 0), out_axes=0)(f, jnp.array(self.velocity_set.c).T)
```

This uses jax.numpy.roll to shift distributions in each velocity direction.

## ⚙️ Warp Implementation

```python
@wp.kernel
def kernel(f_0, f_1):
    index = wp.vec3i(i, j, k)
    _f = functional(f_0, index)
    f_1[...] = _f
```
Warp handles periodic boundary corrections and shift indexing manually within the kernel for 3D arrays.

## 💥 Collision (Base Class)

The `Collision` operator defines how particles interact locally after streaming, typically by relaxing towards equilibrium.

### 📌 Purpose

- Base class for implementing collision models.
- Uses distribution function `f`, equilibrium `feq`, and local properties (`rho`, `u`, etc.).
- Meant to be subclassed (e.g., for `BGK`).

## 🧩 Key Properties

| Property                | Description                      |
|------------------------|----------------------------------|
| `implementation_step`  | `COLLISION`                      |
| `backend_support`      | JAX and Warp                     |
| `supports_auxiliary_data` | ✅ Yes (e.g., `omega`)         |

## 🧪 BGK: A Subclass of Collision

**BGK** (Bhatnagar–Gross–Krook) is a common collision model where the post-collision distribution is calculated by relaxing toward equilibrium.

### ⚙️ JAX Implementation

```python
@jit
def jax_implementation(self, f, feq, rho, u, omega):
    fneq = f - feq
    return f - self.compute_dtype(omega) * fneq
```


### ⚙️ Warp Implementation

```python
@wp.func
def functional(f, feq, rho, u, omega):
    fneq = f - feq
    return f - dtype(omega) * fneq
```
The Warp kernel loads and stores distribution values per node and performs the same BGK operation element-wise.

### 🛠 Backend Support Summary

| Operator   | JAX Support | Warp Support | Streaming | Collision | Supports Aux Data       |
|------------|-------------|--------------|-----------|-----------|------------------------|
| Stream     | ✅ Yes      | ✅ Yes       | ✅ Yes    | ❌ No     | ❌ No                  |
| Collision  | ✅ Yes      | ✅ Yes       | ❌ No     | ✅ Yes    | ✅ Yes                 |
| BGK        | ✅ Yes      | ✅ Yes       | ❌ No     | ✅ Yes    | ✅ Yes (omega)          |
| ForcedCollision | ✅ Yes | ✅ Yes       | ❌ No     | ✅ Yes    | ✅ Yes (force vector)   |
| KBC        | ✅ Yes      | ✅ Yes       | ❌ No     | ✅ Yes    | ✅ Yes                 |

---

### 🗂 Registry

All operator subclasses are registered via the Operator base class using decorators:

```python
@Operator.register_backend(ComputeBackend.JAX)
def jax_implementation(...)
```

This allows dynamic backend dispatch at runtime.