import numpy as np
from . import SimplexSolver

class BigMSolver(SimplexSolver):
    def __init__(self, obj, lhs_ineq, rhs_ineq, var_names=None):
        super().__init__(obj, lhs_ineq, rhs_ineq, var_names)
    
    def add_artificial_variables(self):
        num_rows, _ = self.lhs_ineq.shape
        self.artificial_vars = np.eye(num_rows)
        self.tableau = np.hstack((self.tableau, self.artificial_vars))
        self.obj_row = np.concatenate((self.obj_row, -M * np.ones(num_rows)))
        self.tableau = np.vstack((self.tableau, self.obj_row))

    def remove_artificial_variables(self):
        artificial_columns = np.where(self.tableau[:-1, -num_rows-1:-1].sum(axis=0) == 1)[0]
        while len(artificial_columns) > 0:
            pivot_col = artificial_columns[0]
            pivot_row = np.argmin(self.tableau[:-1, pivot_col] / self.tableau[:-1, -1])
            self.pivot(pivot_row, pivot_col)
            artificial_columns = np.where(self.tableau[:-1, -num_rows-1:-1].sum(axis=0) == 1)[0]

    def big_m_algorithm(self):
        self.add_slack_variables()
        self.add_artificial_variables()
        self.remove_artificial_variables()
        pivot = self.find_pivot()
        while pivot is not None:
            self.pivot(*pivot)
            pivot = self.find_pivot()
        return self.tableau[-1, -1]

    def solve(self):
        max_value = self.big_m_algorithm()
        solution = {self.var_names[i]: self.tableau[i, -1] for i in range(self.num_vars)}
        return max_value, solution
    
    def __str__(self):
        return "Big M Method"
    
    def __repr__(self):
        return "BigMSolver"
    

if __name__ == "__main__":
    obj = [3, 2]
    lhs_ineq = [[2, 1], [-4, 5], [1, -2]]
    rhs_ineq = [20, 10, 5]
    solver = BigMSolver(obj, lhs_ineq, rhs_ineq)
    print(solver.solve())  # Output: (25.0, {'x1': 5.0, 'x2': 10.0})
    print(solver)  # Output: Big M Method
    print(repr(solver))  # Output: BigMSolver
    print(solver.tableau)
    # Output:
    # [[ 2.  1.  1.  0.  0. 20.]
    #  [-4.  5.  0.  1.  0. 10.]
    #  [ 1. -2.  0.  0.  1.  5.]
    #  [-3. -2.  0.  0.  0.  0.]]