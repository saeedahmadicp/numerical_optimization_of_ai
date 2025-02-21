# Numerical Differentiation Methods

This module provides implementations of three common numerical differentiation methods: forward, backward, and central differences. These methods approximate derivatives when analytical differentiation is impractical or impossible.

## Overview

### Forward Difference
```python
f'(x) ≈ [f(x + h) - f(x)] / h
```
- First-order accuracy: $O(h)$
- Uses future point
- Error term: $\frac{h}{2}f''(\xi)$, where $\xi \in [x, x+h]$

### Backward Difference
```python
f'(x) ≈ [f(x) - f(x - h)] / h
```
- First-order accuracy: $O(h)$
- Uses past point
- Error term: $\frac{h}{2}f''(\xi)$, where $\xi \in [x-h, x]$

### Central Difference
```python
f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
```
- Second-order accuracy: $O(h^2)$
- Uses both points
- Error term: $\frac{h^2}{6}f'''(\xi)$, where $\xi \in [x-h, x+h]$

## Usage

```python
from methods.differentiation import forward_difference, backward_difference, central_difference

# Define function
def f(x):
    return x**2

# Parameters
x = 2.0  # Point of interest
h = 1e-5  # Step size

# Compute derivatives
fd = forward_difference(f, x, h)   # ≈ 4.00001
bd = backward_difference(f, x, h)   # ≈ 3.99999
cd = central_difference(f, x, h)    # ≈ 4.00000
```

## Step Size Selection

The choice of step size $h$ involves a tradeoff:
- Too large: truncation error dominates
- Too small: roundoff error dominates

Recommended: $h \approx \sqrt{\epsilon}$ where $\epsilon$ is machine epsilon
```python
import numpy as np
h = np.sqrt(np.finfo(float).eps)  # ≈ 1.49e-8
```

## Features

- Supports scalar and array inputs
- Handles arbitrary function signatures
- Provides clear error messages
- Numerically stable implementations
- Vectorized operations where possible

## Error Handling

All methods include robust error handling:
```python
try:
    result = central_difference(f, x, h)
except ValueError as e:
    print(f"Error: {e}")
```

## References

1. Burden, R. L., & Faires, J. D. (2011). Numerical Analysis (9th ed.)
2. Press, W. H., et al. (2007). Numerical Recipes (3rd ed.)
