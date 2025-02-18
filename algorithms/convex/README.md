# Root-Finding and Optimization Methods

This module provides implementations of various numerical methods for finding roots of nonlinear equations and optimizing functions.

## Root-Finding Methods

### Bracketing Methods

#### Bisection Method
Simple and robust method that repeatedly bisects an interval containing a root.
```python
from methods.root_finding import bisection
x, errors, iters = bisection(f, a=1, b=2, tol=1e-6)
```
- Guaranteed convergence if root exists in interval
- Linear convergence rate
- Requires initial interval [a,b] where f(a)·f(b) < 0

#### Regula Falsi Method
Uses linear interpolation to find better approximations.
```python
from methods.root_finding import regula_falsi
x, errors, iters = regula_falsi(f, a=1, b=2, tol=1e-6)
```
- Faster than bisection in most cases
- Maintains bracket on root
- Superlinear convergence rate

### Derivative Methods

#### Newton-Raphson Method
Uses derivative information to find roots:
```python
from methods.root_finding import newton
x, errors, iters = newton(f, x0=1.5, tol=1e-6)
```
- Quadratic convergence near root
- Requires function to be differentiable
- May fail if derivative is zero
- Uses automatic differentiation via PyTorch

#### Secant Method
Approximates derivative using finite differences:
```python
from methods.root_finding import secant
x, errors, iters = secant(f, x0=1, x1=2, tol=1e-6)
```
- Superlinear convergence (≈1.618)
- Doesn't require derivatives
- More robust than Newton's method

## Optimization Methods

### Direct Search Methods

#### Golden Section Search
Uses golden ratio to optimally reduce search interval:
```python
from methods.root_finding import golden_search
x, errors, iters = golden_search(f, a=-1, b=1, tol=1e-6)
```
- No derivatives required
- Linear convergence rate
- Optimal for unimodal functions

#### Fibonacci Search
Similar to golden section but uses Fibonacci numbers:
```python
from methods.root_finding import fibonacci_search
x, errors, iters = fibonacci_search(f, a=-1, b=1, n_terms=30)
```
- Slightly more efficient than golden section
- Requires predetermined number of iterations

### Gradient-Based Methods

#### Steepest Descent
Uses gradient to move in direction of steepest descent:
```python
from methods.root_finding import steepest_descent
x, errors, iters, history = steepest_descent(f, x0=np.array([1.0, 1.0]))
```
- First-order method
- Linear convergence
- Can be slow near minimum
- Uses automatic differentiation

#### Newton-Hessian Method
Uses both gradient and Hessian information:
```python
from methods.root_finding import newton_hessian
x, errors, iters, history = newton_hessian(f, x0=np.array([1.0, 1.0]))
```
- Second-order method
- Quadratic convergence
- Best for smooth, well-conditioned problems
- Requires twice-differentiable function

#### Powell's Conjugate Direction Method
Builds conjugate directions through successive line minimizations:
```python
from methods.root_finding import powell_conjugate_direction
x, errors, iters, history = powell_conjugate_direction(f, x0=np.array([1.0, 1.0]))
```
- No derivatives required
- Superlinear convergence
- Good for ill-conditioned problems
- Builds quadratic model of function

## Method Selection Guide

1. For Root Finding:
   - If you have a bracketing interval: Use `bisection` (safe) or `regula_falsi` (faster)
   - If you have derivatives: Use `newton` (fast near root)
   - If you don't have derivatives: Use `secant` (good compromise)

2. For Optimization:
   - 1D problems: Use `golden_search` or `fibonacci_search`
   - Small, smooth problems: Use `newton_hessian`
   - Large-scale problems: Use `steepest_descent`
   - Ill-conditioned problems: Use `powell_conjugate_direction`

## Implementation Notes

- All methods return:
  - The solution found
  - List of errors during iteration
  - Number of iterations used
- Optimization methods also return:
  - History of points visited
- All methods accept tolerance and maximum iteration parameters
- Gradient-based methods use PyTorch for automatic differentiation

## References

1. Nocedal, J., & Wright, S. (2006). Numerical Optimization
2. Press, W. H., et al. (2007). Numerical Recipes
3. Burden, R. L., & Faires, J. D. (2010). Numerical Analysis

## License

MIT License - See repository root for details

