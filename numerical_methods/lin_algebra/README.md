# Solving Systems of Linear Equations
This is a collection of algorithms for solving systems of linear equations. The algorithms are implemented in Python.

## Algorithms
1. [Gaussian Elimination](#gaussian-elimination)
2. [Gaussian Elimination with Partial Pivoting](#gaussian-elimination-with-partial-pivoting)
6. [Jacobi Iteration](#jacobi-iteration)
7. [Gauss-Seidel Iteration](#gauss-seidel-iteration)

## Gaussian Elimination
Gaussian elimination is a method used to solve systems of linear equations. The method involves transforming the system of equations into an equivalent system of equations in upper triangular form (called row echelon form). The solution to the system of equations can then be found by back substitution. This method is not stable and can lead to division by zero (or very small numbers which can lead to approximation errors).

## Gaussian Elimination with Partial Pivoting
Gaussian elimination with partial pivoting uses the same method as Gaussian elimination. The difference between this method and Gaussian elimination is that partial pivoting is used to avoid division by zero. Partial pivoting involves swapping rows in the matrix to ensure that the largest element in the column is on the diagonal. This method is more stable than Gaussian elimination.

> Both Gaussian elimination and Gaussian elimination with partial pivoting can be used to solve systems of linear equations. However, for larger systems of linear equations, they can be computationally expensive - on the order of `O(n^3)` - and thus are not practical. For larger systems of linear equations, it is better to use iterative methods such as Jacobi iteration or Gauss-Seidel iteration.

## Jacobi Iteration
Jacobi iteration is an iterative method used to solve systems of linear equations. The method involves iteratively solving for the unknowns in the system of equations. It starts with an initial guess for the unknowns and then iteratively improves the guess.

## Gauss-Seidel Iteration
Gauss-Seidel iteration is an iterative method used to solve systems of linear equations. The method involves iteratively solving for the unknowns in the system of equations.

> The Jacobi and Gauss-Seidel iterative methods are two popular iterative methods used to solve linear systems of equations. The only difference between these two methods is that, in the Jacobi method, the value of the variables is not modified until the next iteration, whereas in Gauss-Seidel method, the value of the variables is modified as soon as a new value is evaluated