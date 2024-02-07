import numpy as np

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
        self.var_names = vars if vars else [f'x{i+1}' for i in range(self.num_vars)]

    def add_slack_variables(self):
        """Add slack variables to the inequalities to convert them to equations
        and convert the objective function to a maximization problem.
        """
        num_rows, _ = self.lhs_ineq.shape
        self.slack_vars = np.eye(num_rows)
        self.tableau = np.hstack((self.lhs_ineq, self.slack_vars, self.rhs_ineq.reshape(-1, 1)))
        self.obj_with_slack = np.concatenate((self.obj, np.zeros(num_rows)))
        self.tableau = np.vstack((self.tableau, np.concatenate((self.obj_with_slack, np.array([0])))))

    def pivot(self, row, col):
        """Apply the pivot operation to the row and column given.
        The pivot operation makes all entries in the pivot column zero except the pivot element.
        row (int): Row index of the pivot element
        col (int): Column index of the pivot element
        """
        self.tableau[row] /= self.tableau[row, col]
        for i in range(len(self.tableau)):
            if i != row:
                self.tableau[i] -= self.tableau[i, col] * self.tableau[row]

    def simplex_algorithm(self):
        """Apply the simplex algorithm to solve the linear programming problem."""
        self.add_slack_variables()
        while any(self.tableau[-1, :-1] < 0):
            pivot_col = np.argmin(self.tableau[-1, :-1])
            ratios = self.tableau[:-1, -1] / self.tableau[:-1, pivot_col]
            valid_ratios = [ratios[i] if ratios[i] > 0 else float('inf') for i in range(len(ratios))]
            pivot_row = np.argmin(valid_ratios)
            self.pivot(pivot_row, pivot_col)
        return self.tableau[-1, -1]

    def solve(self):
        """Solve the linear programming problem and return the maximum value of the objective function
        and the values of the variables at which the maximum value occurs.
        """
        max_value = self.simplex_algorithm()
        variables = self.tableau[:-1, -1] - np.dot(self.tableau[:-1, :-self.num_vars-1], self.obj)
        return max_value, variables[:self.num_vars]
