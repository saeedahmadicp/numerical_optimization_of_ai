# Interpolation

This directory contains two methods for interpolating data: Lagrange Interpolation and Natural Cubic Spline Interpolation.

## What is Interpolation?
Interpolation is the process of estimating the values of a function at points between known data points. It involves constructing a new function that passes through the known points and using this function to estimate values at other points. Interpolation is widely used in various fields such as engineering, physics, and computer graphics to estimate unknown data points within the range of a discrete set of known data points.

## Methods
1. [Lagrange Interpolation](#lagrange-interpolation)
2. [Natural Cubic Spline Interpolation](#spline-interpolation)

## Lagrange Interpolation 
Lagrange interpolation is a polynomial interpolation method that constructs an interpolating polynomial passing through all given data points. The interpolating polynomial is a linear combination of Lagrange basis polynomials, each corresponding to a given data point. A linear Lagrange polynomial is given by:

$$P_1(x) = y_1 L_1(x) + y_2 L_2(x)$$

where $L_1(x)$ and $L_2(x)$ are the Lagrange basis polynomials defined by:

$$L_1(x) = \frac{(x-x_2)}{(x_1-x_2)}$$

$$L_2(x) = \frac{(x-x_1)}{(x_2-x_1)}$$

A Lagrange polynomial of degree $n$ is given by:

$$P_n(x) = y_1 L_1(x) + y_2 L_2(x) + \dots + y_n L_n(x)$$

where $L_i(x)$ is the Lagrange basis polynomial defined by:

$$L_i(x) = \frac{(x-x_1)(x-x_2)\dots(x-x_{i-1})(x-x_{i+1})\dots(x-x_{n+1})}{(x_i-x_1)(x_i-x_2)\dots(x_i-x_{i-1})(x_i-x_{i+1})\dots(x_i-x_{n+1})}$$

where $x_1, x_2, \dots, x_{n+1}$ are the data points used to construct the polynomial. The basis polynomials are multiplied by the corresponding data point to construct the interpolating polynomial. The interpolating polynomial is then used to estimate the function at points between the given data points.

## Spline Interpolation
Natural cubic spline interpolation is a method of interpolating a smooth curve between a given set of points. The method involves constructing a piecewise cubic polynomial function that goes through the points and has continuous first and second derivatives. The natural cubic spline is the interpolating function that has zero second derivatives at the endpoints. Spline interpolation is particularly useful when dealing with noisy data or data that contains gaps, as it allows for a more flexible and accurate representation of the underlying function. In practice, the cubice spline interpolation is used to construct the curve. A computationally efficient algorithm for computing the spline function is given by:

$$s(x) = \frac{(x_j-x)^3M_{j-1} + (x-x_{j-1})^3M_j}{6(x_j-x_{j-1})} + \frac{(x_j-x)y_{j-1} + (x-x_{j-1})y_j}{x_j-x_{j-1}} - \frac{1}{6} (x_j-x_{j-1})[(x_j-x) M_{j-1} + (x-x_{j-1}) M_j]$$

for $x_{j-1} \leq x \leq x_j$ and $j = 1, 2, \dots, n$, where $x_1, x_2, \dots, x_n$ are the data points and $y_1, y_2, \dots, y_n$ are the corresponding function values. The $M_i$ are the second derivatives of the function at the data points, for $i = 0, 1, 2, \dots, n$. To compute the second derivatives, the tridiagonal matrix algorithm is used. The algorithm is given by:

$$M_0 = 0$$

$$M_n = 0$$

$$M_i = \frac{(x_j-x_{j-1})*M_{j-1}}{6} + \frac{(x_{j+1} - x_{j-1})*M_j}{3} + \frac{(x_{j+1}-x_j)*M_{j+1}}{6} = \frac{y_{j+1}-y_j}{x_{j+1}-x_j} - \frac{y_j-y_{j-1}}{x_j-x_{j-1}}$$

for $j = 1, 2, \dots, n-1$


## Usage
Each interpolation method is implemented in a separate Python file in this directory. To use a method, import the corresponding file and call the function with the data points and the points at which to estimate the function. For example:

```python
from lagrange import lagrange_interpolation
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]
x0 = 2.5
print(lagrange_interpolation(x, y, x0))
```

This code will estimate the value of the function $f(x) = x^2$ at $x = 2.5$ using Lagrange interpolation.

## Note
The choice between Lagrange interpolation and spline interpolation depends on the specific problem and the desired accuracy. Lagrange interpolation is straightforward but may not be suitable for large datasets or for functions with high oscillations. Spline interpolation, on the other hand, is more suitable for large datasets and provides a smoother approximation but requires more complex calculations.