from simplex import SimplexSolver

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
lhs_ineq = [[50, 100],
            [100, 40]]

# right-hand side of inequalities
rhs_ineq = [20000, 19200]

var_names = ['corn', 'soybeans']

solver = SimplexSolver(obj, lhs_ineq, rhs_ineq, var_names)
max_value, solution = solver.solve()

print("Maximum profit:", max_value)
print("-" * 25)
print("Optimal planting strategy:")
for var, val in zip(var_names, solution):
    print(f"{var}: {val}")
