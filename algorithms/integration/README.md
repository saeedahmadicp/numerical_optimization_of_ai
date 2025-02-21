# Numerical Integration Methods

This module provides implementations of two common numerical integration methods: the trapezoidal rule and Simpson's rule. These methods approximate definite integrals when analytical integration is impractical or impossible.

## Overview

### Trapezoidal Rule
The approximation for $\int_a^b f(x)dx$ is:

$\int_a^b f(x)dx \approx h\left[\frac{f(x_0)}{2} + \sum_{i=1}^{n-1} f(x_i) + \frac{f(x_n)}{2}\right]$

where $h = \frac{b-a}{n}$

- Second-order accuracy: $O(h^2)$
- Uses linear interpolation
- Error term: $-\frac{(b-a)h^2}{12}f''(\xi)$, where $\xi \in [a,b]$

### Simpson's Rule
The approximation for $\int_a^b f(x)dx$ is:

$\int_a^b f(x)dx \approx \frac{h}{3}\left[f(x_0) + 4\sum_{i=1}^{n/2} f(x_{2i-1}) + 2\sum_{i=1}^{n/2-1} f(x_{2i}) + f(x_n)\right]$

where $h = \frac{b-a}{n}$

- Fourth-order accuracy: $O(h^4)$
- Uses quadratic interpolation
- Error term: $-\frac{(b-a)h^4}{180}f^{(4)}(\xi)$, where $\xi \in [a,b]$

## Usage

```python
from methods.integration import trapezoidal, simpson

# Define function
def f(x):
    return x**2

# Parameters
a, b = 0.0, 1.0  # Integration bounds
n = 100          # Number of subintervals

# Compute integrals
trap = trapezoidal(f, a, b, n)    # ≈ 0.33336
simp = simpson(f, a, b, n)        # ≈ 0.33333
```

## Features

- Supports scalar and array inputs
- Handles arbitrary function signatures
- Provides clear error messages
- Numerically stable implementations
- Vectorized operations for efficiency

## Error Handling

All methods include robust error handling:
```python
try:
    result = simpson(f, a, b, n)
except ValueError as e:
    print(f"Error: {e}")
```

## References

1. Burden, R. L., & Faires, J. D. (2011). Numerical Analysis (9th ed.)
2. Press, W. H., et al. (2007). Numerical Recipes (3rd ed.)
