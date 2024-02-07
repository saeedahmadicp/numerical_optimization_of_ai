# Simplex Algorithm
The Simplex Algorithm is a popular method for solving linear programming problems. It is an iterative algorithm that starts with an initial feasible solution and moves to a new feasible solution in each iteration. The algorithm terminates when it reaches an optimal solution.

## How it works
The Simplex Algorithm works by finding the optimal solution to a linear programming problem. It does this by moving from one feasible solution to another, improving the value of the objective function at each step. The algorithm starts with an initial feasible solution and iteratively moves to a new feasible solution in each iteration. The algorithm terminates when it reaches an optimal solution.

## Pseudocode
The Simplex Algorithm can be described using the following pseudocode:

```plaintext
1. Start with an initial feasible solution.
2. While there is a pivot column:
    a. Choose a pivot column.
    b. Choose a pivot row.
    c. Perform a pivot operation.
3. Return the optimal solution.
```

## Usage
To use the Simplex Solver, follow these steps:

1. **Define the coefficients of your objective function:** This should be done in the form of a list, where each element corresponds to the coefficient of a decision variable in the objective function. Note that the objective function should be defined to maximize the outcome.

2. **Define the left-hand side coefficients of your inequality constraints:** These should be provided as a list of lists, where each sub-list represents the coefficients of the decision variables in one constraint.

3. **Define the right-hand side coefficients of your inequality constraints:** This should be a list where each element corresponds to the right-hand side of an inequality constraint.

4. Create an instance of the `SimplexSolver` class with your problem definition.
4. Call the `solve` method on your instance to solve the problem.

## Example
```python
from simplex_solver import SimplexSolver

# Objective function (Maximize Z = 4x1 + 3x2 + 6x3)
obj = [-4, -3, -6]  # Coefficients for the objective function (note the negative sign for maximization)

# Constraints (3x1 + x2 + 3x3 <= 30, 2x1 + 2x2 + 3x3 <= 40)
lhs_ineq = [
  [3, 1, 3],  # Coefficients for the first constraint
  [2, 2, 3]   # Coefficients for the second constraint
]
rhs_ineq = [30, 40]  # Right-hand side of the constraints

# Variable names
var_names = ["x1", "x2", "x3"]

# Creating an instance of SimplexSolver
solver = SimplexSolver(obj, lhs_ineq, rhs_ineq, var_names)

# Solving the problem
max_value, solution = solver.solve()

# Displaying the results
print("Maximum value of Z:", max_value)
print("Solution:")
for var, value in solution.items():
    print(f"{var} = {value}")
```