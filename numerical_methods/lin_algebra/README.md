# Linear Algebra
This directory contains implementations of four different methods for solving linear equations and systems of linear equations.

## What is Linear Algebra?
Linear algebra is the branch of mathematics that deals with vector spaces. It is used to solve systems of linear equations.

## Methods
1. [Gaussian Elimination](#gaussian-elimination)
2. [Gaussian Elimination with Pivoting](#gaussian-elimination-with-partial-pivoting)
3. [Gauss-Seidel Iteration](#gauss-seidel-iteration)
4. [Jacobi Iteration](#jacobi-iteration)

### Gaussian Elimination
> Gaussian elimination is a method of solving systems of linear equations by transforming the augmented matrix into row echelon form using elementary row operations, then back-substituting to find the solution. This method is efficient for small systems, but can be computationally expensive for large systems. This method is also not stable, meaning that it can produce inaccurate results for certain systems of linear equations like those with small pivots.

### Gaussian Elimination with Pivoting
> Gaussian elimination with pivoting is similar to Gaussian elimination, but includes a step to swap rows if necessary to avoid dividing by small numbers (pivot elements). This can improve numerical stability and prevent errors that can arise from rounding errors in floating point arithmetic.

### Gauss-Seidel Iteration
> The Gauss-Seidel method is an iterative method for solving systems of linear equations. It starts with an initial guess for the solution and then repeatedly updates each component of the solution using the latest values of the other components. This method can converge faster than Gaussian elimination for certain types of systems.

### Jacobi Iteration
> The Jacobi method is another iterative method for solving systems of linear equations. Like the Gauss-Seidel method, it starts with an initial guess and updates each component of the solution iteratively. However, the Jacobi method uses the values from the previous iteration to update each component, rather than the latest values. This method can also converge faster than Gaussian elimination for certain types of systems.

## Usage
To use a method, import the corresponding file and call the function with the augmented matrix. For example:

```python
from gauss_elim import gauss_elim
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
b = [1, 2, 3]
print(gauss_elim(A, b))
```

This will print the solution to the system of linear equations Ax = b using Gaussian elimination.

## Note
The choice of method depends on the specific problem and the accuracy required. Gaussian elimination is simpler to implement and can be used for any system of linear equations, but it can be computationally expensive for large systems and can produce inaccurate results for certain systems. Gaussian elimination with pivoting is more computationally efficient and can be used for large systems, but it can be more difficult to implement and is not suitable for all systems. The Gauss-Seidel and Jacobi methods are iterative methods that can be used for large systems of linear equations. They are more computationally efficient than Gaussian elimination for certain types of systems, but they can be more difficult to implement and are not suitable for all systems. In general, the Gauss-Seidel method is preferred over the Jacobi method because it can converge faster for certain types of systems.