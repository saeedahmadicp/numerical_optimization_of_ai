# Interpolation Algorithms
This repository contains numerical interpolation algorithms implemented in Python.

## Algorithms
1. [Lagrange Interpolation](#lagrange-interpolation)
2. [Spline Interpolation](#spline-interpolation)

## Lagrange Interpolation 
Lagrange interpolation is a method used to approximate a function by a polynomial that passes through all of the given data points. The method uses Lagrange polynomials, which are a set of basis polynomials that are used to construct the interpolating polynomial. The interpolating polynomial is then used to estimate the function at points between the given data points.

A linear Lagrange polynomial is given by:

$P_1(x) = y_1 L_1(x) + y_2 L_2(x)$

where $L_1(x)$ and $L_2(x)$ are the Lagrange basis polynomials defined by:

$L_1(x) = \frac{(x-x_2)}{(x_1-x_2)}$

$L_2(x) = \frac{(x-x_1)}{(x_2-x_1)}$

where $x_1$ and $x_2$ are the two data points used to construct the polynomial. The basis polynomials are multiplied by the corresponding data point to construct the interpolating polynomial. The interpolating polynomial is then used to estimate the function at points between the given data points.

A Lagrange polynomial of degree $n$ is given by:

$P_n(x) = y_1 L_1(x) + y_2 L_2(x) + \dots + y_n L_n(x)$

where $L_i(x)$ is the Lagrange basis polynomial defined by:

$L_i(x) = \frac{(x-x_1)(x-x_2)\dots(x-x_{i-1})(x-x_{i+1})\dots(x-x_{n+1})}{(x_i-x_1)(x_i-x_2)\dots(x_i-x_{i-1})(x_i-x_{i+1})\dots(x_i-x_{n+1})}$

where $x_1, x_2, \dots, x_{n+1}$ are the data points used to construct the polynomial. The basis polynomials are multiplied by the corresponding data point to construct the interpolating polynomial. The interpolating polynomial is then used to estimate the function at points between the given data points.


## Spline Interpolation
Spline interpolation is a technique used to construct a smooth curve that passes through a given set of points. Unlike Lagrange interpolation, spline interpolation uses piecewise-defined polynomial functions to approximate the data, resulting in a smoother and more accurate curve. The technique involves dividing the data into smaller sections and fitting a polynomial function to each section. The polynomial functions are then matched at the points where the sections meet, ensuring a continuous and smooth curve. Spline interpolation is particularly useful when dealing with noisy data or data that contains gaps, as it allows for a more flexible and accurate representation of the underlying function. It is commonly used in fields such as engineering, physics, and computer graphics.

In practice, the cubic spline is widely used and is given by:

$s(x) = \frac{(x_j-x)^3M_{j-1} + (x-x_{j-1})^3M_j}{6(x_j-x_{j-1})} + \frac{(x_j-x)y_{j-1} + (x-x_{j-1})y_j}{x_j-x_{j-1}} - \frac{(x_j-x)[(x_j-x)M_{j-1}] + (x-x_{j-1})M_j}{6}$

for $x_{j-1} \leq x \leq x_j$ and $j = 1, 2, \dots, n$

where $x_1, x_2, \dots, x_n$ are the data points and $y_1, y_2, \dots, y_n$ are the corresponding function values. The $M_i$ are the second derivatives of the function at the data points, for $i = 0, 1, 2, \dots, n$.

To compute the second derivatives, the tridiagonal matrix algorithm is used. The algorithm is given by:

$M_0 = 0$

$M_n = 0$

$M_i = \frac{(x_j-x_{j-1})*M_{j-1}}{6} + \frac{(x_{j+1} - x_{j-1})*M_j}{3} + \frac{(x_{j+1}-x_j)*M_{j+1}}{6} = \frac{y_{j+1}-y_j}{x_{j+1}-x_j} - \frac{y_j-y_{j-1}}{x_j-x_{j-1}}$

for $j = 1, 2, \dots, n-1$
