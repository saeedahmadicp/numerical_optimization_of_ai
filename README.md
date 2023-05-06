# Numerical Optimization of AI
This repository is a collection of numerical functions implemented in Python, aimed at aiding the optimization of artificial intelligence models. The functions are organized into different folders based on their function, making it easy to find the required function for a specific task.

The folders are organized as follows:

- **[Differentiation](/numerical_methods/differentiation/):** This folder contains functions for numerical differentiation methods, including the [`forward difference`](/numerical_methods/differentiation/forward_difference.py), [`backward difference`](/numerical_methods/differentiation/backward_difference.py), and [`central difference`](/numerical_methods/differentiation/central_difference.py). <br>
- **[Integration](/numerical_methods/integration/):** This folder contains functions for numerical integration methods, including the [`trapezoidal rule`](/numerical_methods/integration/trapezoidal.py) and [`Simpson's rule`](/numerical_methods/integration/simpson.py). <br>
- **[Interpolation](/numerical_methods/interpolation/):** This folder contains functions for numerical interpolation methods, including the [`Lagrange interpolation`](/numerical_methods/interpolation/lagrange.py) and [`Spline interpolation`](/numerical_methods/interpolation/spline.py). <br>
- **[Linear Algebra](/numerical_methods/lin_algebra/):** This folder contains functions for numerical linear algebra, including the [`Gaussian elimination without pivoting`](/numerical_methods/lin_algebra/GE.py), [`Gaussain elimination with pivoting`](/numerical_methods/lin_algebra/GEpivot.py), [`Jacobi`](/numerical_methods/lin_algebra/Jacobi.py), and [`Gauss-Seidel`](/numerical_methods/lin_algebra/GaussSeidel.py) methods. <br>
- **[Regression](/numerical_methods/regression/):** This folder contains functions for regression analysis, including [`linear regression`](/numerical_methods/regression/linear.py) and polynomial regression with the [`chebyshev polynomials`](/numerical_methods/regression/chebyshev.py). <br>
- **[Root Finding](/numerical_methods/root_finding/):** This folder contains functions for finding roots of equations, including the [`bisection method`](/numerical_methods/root_finding/bisection.py), [`secant method`](/numerical_methods/root_finding/secant.py), [`Newton-Raphson`](/numerical_methods/root_finding/newton.py), [`Newton-Hessian`](/numerical_methods/root_finding/newton_hessian.py), [`Powell method`](/numerical_methods/root_finding/powell.py), [`Nelder-Mead`](/numerical_methods/root_finding/nelder_mead.py), [`Regula Falsi`](/numerical_methods/root_finding/regula_falsi.py), and the [`Steepest Descent`](/numerical_methods/root_finding/steepest_descent.py) method.

Each function has been implemented in a modular fashion, making it easy to integrate into your own code. Additionally, extensive mathematical documentation has been provided to aid in understanding of the functions. We hope that these functions will help you in the optimization of your AI models.

## Contribute
We welcome any contributions to the repository, including new functions, bug fixes, and documentation improvements. To contribute, please follow the steps below:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Push your changes to your fork
6. Create a pull request