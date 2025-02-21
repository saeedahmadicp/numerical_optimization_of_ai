# Linear Algebra Methods

This module provides implementations of direct and iterative methods for solving linear systems of equations $Ax = b$.

## Direct Methods

### Gaussian Elimination
Basic elimination method that transforms the system into upper triangular form.

$A = LU$ decomposition:
```python
x, A_elim, b_elim = gauss_elim(A, b)
```

- Time complexity: $O(n^3)$
- Space complexity: $O(n^2)$
- No pivoting strategy
- May be unstable for ill-conditioned systems

### Gaussian Elimination with Partial Pivoting
Enhanced elimination method that selects the largest pivot in each column.

$PA = LU$ decomposition:
```python
x, A_elim, b_elim = gauss_elim_pivot(A, b)
```

- Time complexity: $O(n^3)$
- Space complexity: $O(n^2)$
- Better numerical stability
- Handles zero pivots through row exchanges

## Iterative Methods

### Gauss-Seidel Method
Iterative method that updates each component using the latest available values.

For each iteration:
$x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j<i} a_{ij}x_j^{(k+1)} - \sum_{j>i} a_{ij}x_j^{(k)}\right)$

```python
x, converged, iterations = gauss_seidel(A, b, x0, tol=1e-6)
```

- Convergence requires diagonal dominance: $|a_{ii}| > \sum_{j\neq i} |a_{ij}|$
- Faster convergence than Jacobi
- Sequential updates
- Memory efficient: $O(n)$ extra space

### Jacobi Method
Iterative method that updates all components using previous iteration values.

For each iteration:
$x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j\neq i} a_{ij}x_j^{(k)}\right)$

```python
x, converged, iterations = jacobi(A, b, x0, tol=1e-6)
```

- Convergence requires diagonal dominance
- Parallel-friendly updates
- Memory efficient: $O(n)$ extra space
- Generally slower convergence than Gauss-Seidel

## Usage Example

```python
import numpy as np
from methods.lin_algebra import gauss_elim_pivot, gauss_seidel

# Create system Ax = b
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
b = np.array([1, 5, 2])

# Direct solution
x_direct, _, _ = gauss_elim_pivot(A, b)

# Iterative solution
x0 = np.zeros_like(b)
x_iter, conv, iters = gauss_seidel(A, b, x0, tol=1e-8)
```

## Method Selection Guide

1. **Small Systems** ($n < 1000$):
   - Use `gauss_elim_pivot` for general cases
   - Use `gauss_elim` if system is well-conditioned

2. **Large Systems** ($n \geq 1000$):
   - Use `gauss_seidel` for diagonally dominant systems
   - Use `jacobi` if parallel computation is available

3. **Special Cases**:
   - Symmetric positive definite: `gauss_seidel` converges well
   - Strictly diagonally dominant: All methods are stable
   - Ill-conditioned: `gauss_elim_pivot` is most reliable

## References

1. Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.)
2. Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.)
3. Trefethen, L. N., & Bau III, D. (1997). Numerical Linear Algebra
