# Linear Programming Algorithms

This module provides implementations of various linear programming algorithms for solving continuous optimization problems.

### Simplex Methods
- Primal Simplex Algorithm
- Revised Simplex Method
- Dual Simplex Method
- Network Simplex
- Parametric Simplex

### Interior Point Methods
- Primal-Dual Methods
- Predictor-Corrector Methods
- Path-Following Algorithms
- Potential Reduction Methods
- Affine Scaling

### Decomposition Methods
- Dantzig-Wolfe Decomposition
- Benders Decomposition
- Column Generation
- Row Generation
- Block Angular Decomposition

### Special Purpose Methods
- Network Flow Algorithms
- Transportation Problems
- Assignment Problems
- Minimum Cost Flow
- Maximum Flow

### Implementation Details
Each algorithm will be implemented with:
- Clear mathematical formulation
- Efficient matrix operations
- Numerical stability considerations
- Degeneracy handling
- Cycling prevention
- Warm start capabilities
- Infeasibility detection

## Usage Example

```python
from algorithms.linear_programming import SimplexSolver

# Define problem
c = [-4, -3, -6]  # Objective coefficients
A = [[3, 1, 3],   # Constraint matrix
     [2, 2, 3]]
b = [30, 40]      # Right-hand side

# Create solver instance
solver = SimplexSolver(c, A, b)

# Solve problem
solution = solver.solve()
print(f"Optimal value: {solution.objective_value}")
print(f"Optimal point: {solution.x}")
```

## References

1. Dantzig, G.B. (1963). Linear Programming and Extensions
2. Karmarkar, N. (1984). A New Polynomial-Time Algorithm for Linear Programming
3. Vanderbei, R.J. (2014). Linear Programming: Foundations and Extensions

## License

MIT License - See repository root for details 