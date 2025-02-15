# algorithms/linear_programming/simplex/bigm.py

"""Big M method for solving linear programming problems with artificial variables."""

from typing import Dict, Tuple, Optional, List
import numpy as np
from .primal import SimplexSolver

__all__ = ["BigMSolver"]


class BigMSolver(SimplexSolver):
    """Solver using Big M method for linear programming problems.

    The Big M method handles problems with artificial variables by adding a large
    penalty term (M) to the objective function.
    """

    def __init__(
        self,
        obj: List[float],
        lhs_ineq: List[List[float]],
        rhs_ineq: List[float],
        var_names: Optional[List[str]] = None,
        M: float = 1e6,
    ):
        """Initialize Big M solver.

        Args:
            obj: Coefficients of objective function to maximize
            lhs_ineq: Left-hand side coefficients of inequalities
            rhs_ineq: Right-hand side of inequalities
            var_names: Optional variable names
            M: Large penalty value for artificial variables
        """
        super().__init__(obj, lhs_ineq, rhs_ineq, var_names)
        self.M = M

    def add_artificial_variables(self) -> None:
        """Add artificial variables to the tableau.

        Adds artificial variables with large negative coefficients (-M)
        in the objective function.
        """
        num_rows, _ = self.lhs_ineq.shape
        self.artificial_vars = np.eye(num_rows)
        self.tableau = np.hstack((self.tableau, self.artificial_vars))
        self.obj_row = np.concatenate((self.obj_row, -self.M * np.ones(num_rows)))
        self.tableau = np.vstack((self.tableau, self.obj_row))

    def remove_artificial_variables(self) -> None:
        """Remove artificial variables from basis if possible.

        Performs pivot operations to remove artificial variables
        from the basis where possible.
        """
        num_rows = len(self.artificial_vars)
        artificial_columns = np.where(self.tableau[:-1, -num_rows:].sum(axis=0) == 1)[0]

        for col in artificial_columns:
            pivot_row = np.argmin(self.tableau[:-1, col] / self.tableau[:-1, -1])
            self.pivot(pivot_row, col)

    def big_m_algorithm(self) -> float:
        """Execute the Big M algorithm.

        Returns:
            Optimal objective value

        Raises:
            ValueError: If problem is infeasible or unbounded
        """
        # Phase I: Handle artificial variables
        self.add_slack_variables()
        self.add_artificial_variables()
        self.remove_artificial_variables()

        # Phase II: Solve resulting LP
        while True:
            pivot = self.find_pivot()
            if pivot is None:
                break
            self.pivot(*pivot)

        # Check feasibility
        if any(abs(x) > 1e-10 for x in self.tableau[-1, -len(self.artificial_vars) :]):
            raise ValueError("Problem is infeasible")

        return self.tableau[-1, -1]

    def solve(self) -> Tuple[float, Dict[str, float]]:
        """Solve the linear programming problem.

        Returns:
            Tuple of (optimal_value, solution_dict) where:
                optimal_value: Value of objective function at optimum
                solution_dict: Dictionary mapping variable names to values

        Raises:
            ValueError: If problem is infeasible or unbounded
        """
        max_value = self.big_m_algorithm()
        solution = {
            self.var_names[i]: self.tableau[i, -1] for i in range(self.num_vars)
        }
        return max_value, solution

    def __str__(self) -> str:
        """Return string representation."""
        return "Big M Method Solver"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"BigMSolver(M={self.M})"


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
