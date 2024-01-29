# Linear Algebra

This directory contains implementations of four different methods for solving linear equations and systems of linear equations.

## What is Linear Algebra?
Linear algebra is a branch of mathematics that focuses on solving systems of linear equations using vector spaces and matrices. It plays a crucial role in various fields such as engineering, physics, computer science, and more.

## Methods
1. [Gaussian Elimination](#gaussian-elimination)
2. [Gaussian Elimination with Pivoting](#gaussian-elimination-with-pivoting)
3. [Gauss-Seidel Iteration](#gauss-seidel-iteration)
4. [Jacobi Iteration](#jacobi-iteration)

### Gaussian Elimination
Gaussian elimination is a method for solving linear systems by converting the matrix to row echelon form using elementary row operations, then back-substituting to find the solution. It's efficient for small systems but can be computationally intensive for large ones. This method is not inherently stable, as it can produce inaccurate results for systems with small pivot elements.

### Gaussian Elimination with Pivoting
Gaussian elimination with pivoting involves swapping rows during the elimination process to ensure that the largest, or a relatively large, element is used as the pivot. This technique improves numerical stability, reducing errors caused by floating-point arithmetic.

### Gauss-Seidel Iteration
The Gauss-Seidel method is an iterative technique for solving linear systems. It starts with an initial guess and refines it iteratively, using the latest values from the previous iterations. This method can be more efficient than Gaussian elimination for certain types of matrices.

### Jacobi Iteration
The Jacobi method is another iterative approach, similar to Gauss-Seidel, but it uses the values from the previous iteration for all updates in the current iteration. While it may converge slower than Gauss-Seidel, it is often easier to implement and can be more suitable for parallel processing.

## Usage
To use these methods, import the corresponding Python file and call the function with your matrix (and vector for Gaussian methods). For example:

```python
from gauss_elim import gauss_elim
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
b = [1, 2, 3]
print(gauss_elim(A, b))
```

This will solve the system of linear equations Ax = b using Gaussian elimination.

## Note
The choice of method should be based on the problem's size, the matrix's properties, and the desired accuracy. Gaussian elimination is versatile but may not be the best choice for large systems or systems with certain numerical properties. The iterative methods, Gauss-Seidel and Jacobi, are often preferred for large systems due to their computational efficiency and suitability for parallelization.