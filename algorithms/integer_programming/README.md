# Integer Programming Algorithms

This module provides implementations of various integer programming algorithms for solving discrete optimization problems.

### Branch and Bound
- Systematic enumeration of candidate solutions
- Uses bounds to prune search space
- Can handle general integer programming problems

### Cutting Plane Methods
- Gomory Cuts
- Lift and Project
- Strengthens LP relaxation
- Can be combined with branch and bound

### Branch and Cut
- Combines branch and bound with cutting planes
- Dynamic generation of cuts
- State-of-the-art for many problems

### Column Generation
- Handles problems with many variables
- Generates variables as needed
- Useful for large-scale problems

### Dynamic Programming
- Solves problems by breaking into subproblems
- Optimal solutions for certain problem classes
- Requires problem-specific structure

### Lagrangian Relaxation
- Relaxes difficult constraints
- Provides bounds on optimal value
- Useful for decomposition

## Usage Example

```python
from algorithms.integer_programming import Item, Knapsack

# Create items
items = [
    Item(weight=2, value=3),
    Item(weight=3, value=4),
    Item(weight=4, value=5),
]

# Create knapsack instance
max_weight = 6
knapsack = Knapsack(items, max_weight)

# Solve using different methods
dp_solution = knapsack.solve("dynamic_programming")
greedy_solution = knapsack.solve("greedy")
brute_solution = knapsack.solve("brute_force")

print(f"Dynamic Programming: {dp_solution}")
print(f"Greedy: {greedy_solution}")
print(f"Brute Force: {brute_solution}")
```

## References

1. Nemhauser, G.L., & Wolsey, L.A. (1988). Integer Programming and Combinatorial Optimization
2. Schrijver, A. (1998). Theory of Linear and Integer Programming
3. Wolsey, L.A. (1998). Integer Programming

## License

MIT License - See repository root for details 