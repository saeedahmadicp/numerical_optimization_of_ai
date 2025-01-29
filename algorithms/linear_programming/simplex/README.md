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
"""
LINEAR PROGRAMMING PROBLEM:

A farmer has a 320 acre farm on which she plants two crops: corn and soybeans. For
each acre of corn planted, her expenses are $50 and for each acre of soybeans planted, her expenses
are $100. Each acre of corn requires 100 bushels of storage and yields a profit of $60; each acre of
soybeans requires 40 bushels of storage and yields a profit of $90. If the total amount of storage
space available is 19,200 bushels and the farmer has only $20,000 on hand, how many acres of each
crop should she plant in order to maximize her profit? What will her profit be if she follows this
strategy?

Let x1 be the number of acres of corn and x2 be the number of acres of soybeans. 

OBJECTIVE FUNCTION:
Maximize: 60x1 + 90x2

CONSTRAINTS:
50x1 + 100x2 <= 20000 (total expenses)
100x1 + 40x2 <= 19200 (total storage space)
x1, x2 >= 0 (non-negativity)
"""

# objective function to maximize
obj = [-60, -90]

# left-hand side of inequalities
lhs_ineq = [[50, 100], [100, 40]]

# right-hand side of inequalities
rhs_ineq = [20000, 19200]

var_names = ["corn", "soybeans"]

solver = SimplexSolver(obj, lhs_ineq, rhs_ineq, var_names)
max_value, solution = solver.solve()

print("Maximum profit:", max_value)
print("-" * 25)
print("Optimal planting strategy:")
for var, val in zip(var_names, solution):
    print(f"{var}: {val}")
```