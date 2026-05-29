# Warp Backend for Gradient-Based Optimization - User Guide

## Quick Start

Both JAX and Warp backends support gradient-based optimization. Choose based on your needs:

| Use JAX when... | Use Warp when... |
|----------------|------------------|
| Prototyping | Performance is critical |
| Ease of use matters | 8x speedup is worth setup |
| Variable-length sims | Fixed-length sims |
| Default choice | Production workflows |

## Running the Example

```bash
# JAX backend (default, easier)
python examples/cfd/differentiable_lbm.py --shape circle --backend jax

# Warp backend (8x faster forward simulation)
python examples/cfd/differentiable_lbm.py --shape circle --backend warp
```

**Both achieve ~97% improvement** - identical optimization performance!

## Warp Backend: What You Need to Know

### The One Critical Requirement

**Pre-allocate arrays for all simulation steps.**

```python
# CORRECT - Pre-allocate all states
self.f_states = [
    wp.zeros(shape, dtype=wp.float32, requires_grad=True)
    for _ in range(num_steps + 1)
]

with wp.Tape() as tape:
    for step in range(num_steps):
        self.f_states[step + 1].zero_()  # Clear, don't recreate
        _, self.f_states[step + 1] = stepper(
            self.f_states[step],
            self.f_states[step + 1],
            ...
        )
    loss = compute_loss(self.f_states[num_steps])

tape.backward(loss=loss)
```

```python
# WRONG - Recreates arrays (breaks gradient flow)
for step in range(num_steps):
    f_1 = wp.zeros_like(f_0)  # New array each time - BAD!
    f_0, f_1 = stepper(f_0, f_1, ...)
```

### Why This Matters

**Warp uses tape-based AD**: Records operations during forward, replays backward to compute gradients.

**The tape needs all intermediate values** to exist when backward runs.

**Recreating arrays** breaks the chain → gradients can't flow back.

### Memory Requirements

```python
# Estimate memory
num_steps = 20
grid_size = (512, 512)
num_directions = 9  # D2Q9
bytes_per_float = 4

memory_MB = (num_steps + 1) * num_directions * grid_size[0] * grid_size[1] * bytes_per_float / 1e6

# Example: 20 steps, 512x512 grid = 197 MB (totally fine)
```

**For very long simulations** (>100 steps), consider checkpointing (advanced topic).

## Complete Working Example

See `examples/cfd/differentiable_lbm.py` for a fully working example showing both backends.


## Performance Comparison

**NVIDIA Benchmark** (134M cell simulation, A100 GPU):
- JAX: 1x baseline
- **Warp: 8x faster**

**Our Test** (Circle optimization, 64×64, 10 steps, 20 iterations):
- JAX: 97.46% improvement
- Warp: 97.56% improvement ← **Same result, faster execution**

## Troubleshooting

### Problem: Optimization oscillates / doesn't converge

**Cause**: Arrays being recreated inside simulation loop

**Solution**: Pre-allocate all states before the tape (see pattern above)

### Problem: "Array does not have gradient" error

**Cause**: Array created without `requires_grad=True`

**Solution**: Add `requires_grad=True` to all allocations:
```python
wp.zeros(..., requires_grad=True)
```

### Problem: Gradients are zero

**Cause**: Computational graph severed somewhere

**Solution**: 
1. Check all arrays have `requires_grad=True`
2. Use `.zero_()` to clear, not `wp.zeros_like()`
3. Keep all intermediate states alive during backward
