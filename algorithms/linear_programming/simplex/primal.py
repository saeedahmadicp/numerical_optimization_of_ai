# algorithms/linear_programming/simplex/simplex.py

import numpy as np

__all__ = ["SimplexSolver"]


class SimplexSolver:
    def __init__(self, obj, lhs_ineq, rhs_ineq, var_names=None):
        """
        obj: list of floats
            Coefficients of the objective function to be maximized
        lhs_ineq: list of lists of floats
            Coefficients of the left-hand side of the inequalities
        rhs_ineq: list of floats
            Right-hand side of the inequalities
        var_names: list of strings (optional) as decision variable names
        """
        self.obj = np.array(obj)
        self.lhs_ineq = np.array(lhs_ineq)
        self.rhs_ineq = np.array(rhs_ineq)
        self.num_vars = len(obj)
        self.var_names = (
            var_names if var_names else [f"x{i+1}" for i in range(self.num_vars)]
        )

    def add_slack_variables(self):
        """Add slack variables to the inequalities to convert them to equations
        and convert the objective function to a maximization problem.
        """
        num_rows, _ = self.lhs_ineq.shape
        slack_size = num_rows
        self.slack_vars = np.eye(slack_size)
        self.tableau = np.hstack(
            (self.lhs_ineq, self.slack_vars, self.rhs_ineq.reshape(-1, 1))
        )
        self.obj_row = np.concatenate((self.obj, np.zeros(slack_size + 1)))
        self.tableau = np.vstack((self.tableau, self.obj_row))

    def pivot(self, row, col):
        """Apply the pivot operation to the row and column given.
        The pivot operation makes all entries in the pivot column zero except the pivot element.
        row (int): Row index of the pivot element
        col (int): Column index of the pivot element
        """
        self.tableau[row, :] /= self.tableau[row, col]
        num_rows, _ = self.tableau.shape
        for r in range(num_rows):
            if r != row:
                self.tableau[r, :] -= self.tableau[r, col] * self.tableau[row, :]

    def find_pivot(self):
        """Find the pivot element in the current tableau.
        The pivot column is the most negative entry in the bottom row.
        The pivot row is the row with the minimum ratio of the right-hand side to the pivot column.
        Returns:
        (int, int): The pivot row and column
        """
        pivot_col = np.argmin(self.tableau[-1, :-1])
        if self.tableau[-1, pivot_col] >= 0:
            return None  # Optimal solution found
        ratios = np.array(
            [
                (
                    self.tableau[i, -1] / self.tableau[i, pivot_col]
                    if self.tableau[i, pivot_col] > 0
                    else np.inf
                )
                for i in range(len(self.tableau) - 1)
            ]
        )
        pivot_row = ratios.argmin()
        return pivot_row, pivot_col

    def simplex_algorithm(self):
        """Apply the simplex algorithm to solve the linear programming problem."""
        self.add_slack_variables()
        pivot = self.find_pivot()
        while pivot is not None:
            self.pivot(*pivot)
            pivot = self.find_pivot()
        return self.tableau[-1, -1]

    def solve(self):
        """Solve the linear programming problem and return the maximum value of the objective function
        and the values of the variables at which the maximum value occurs.
        """
        max_value = self.simplex_algorithm()
        solution = {
            self.var_names[i]: self.tableau[i, -1] for i in range(self.num_vars)
        }
        return max_value, solution
