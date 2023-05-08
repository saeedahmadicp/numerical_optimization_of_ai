# Interpolation
This directory contains two methods for interpolating data: Lagrange Interpolation and Natural Cubic Spline.

## What is interpolation?
Interpolation is the process of estimating the value of a function at points between the given data points. The process involves constructing a function that passes through the given data points and using the function to estimate the value of the function at points between the data points. Interpolation is commonly used in fields such as engineering, physics, and computer graphics.

## Methods
1. [Lagrange Interpolation](#lagrange-interpolation)
2. [Natural Cubic Spline Interpolation](#spline-interpolation)

### Lagrange Interpolation 
> Lagrange interpolation is a method used to approximate a function by a polynomial that passes through all of the given data points. The method uses Lagrange polynomials, which are a set of basis polynomials that are used to construct the interpolating polynomial. The interpolating polynomial is then used to estimate the function at points between the given data points. A linear Lagrange polynomial is given by:
>
> $$P_1(x) = y_1 L_1(x) + y_2 L_2(x)$$
> where $L_1(x)$ and $L_2(x)$ are the Lagrange basis polynomials defined by:
> $$L_1(x) = \frac{(x-x_2)}{(x_1-x_2)}$$
> $$L_2(x) = \frac{(x-x_1)}{(x_2-x_1)}$$
>
> A Lagrange polynomial of degree $n$ is given by:
> $$P_n(x) = y_1 L_1(x) + y_2 L_2(x) + \dots + y_n L_n(x)$$
> where $L_i(x)$ is the Lagrange basis polynomial defined by:
> $$L_i(x) = \frac{(x-x_1)(x-x_2)\dots(x-x_{i-1})(x-x_{i+1})\dots(x-x_{n+1})}{(x_i-x_1)(x_i-x_2)\dots(x_i-x_{i-1})(x_i-x_{i+1})\dots(x_i-x_{n+1})}$$
> where $x_1, x_2, \dots, x_{n+1}$ are the data points used to construct the polynomial. The basis polynomials are multiplied by the corresponding data point to construct the interpolating polynomial. The interpolating polynomial is then used to estimate the function at points between the given data points.

### Spline Interpolation
> Natural cubic spline interpolation is a method of interpolating a smooth curve between a given set of points. The method involves constructing a piecewise cubic polynomial function that goes through the points and has continuous first and second derivatives. The natural cubic spline is the interpolating function that has zero second derivatives at the endpoints. Spline interpolation is particularly useful when dealing with noisy data or data that contains gaps, as it allows for a more flexible and accurate representation of the underlying function. In practice, the cubice spline interpolation is used to construct the curve. A computationally efficient algorithm for computing the spline function is given by:
>
> $$s(x) = \frac{(x_j-x)^3M_{j-1} + (x-x_{j-1})^3M_j}{6(x_j-x_{j-1})} + \frac{(x_j-x)y_{j-1} + (x-x_{j-1})y_j}{x_j-x_{j-1}} - \frac{1}{6} (x_j-x_{j-1})[(x_j-x) M_{j-1} + (x-x_{j-1}) M_j]$$
> for $x_{j-1} \leq x \leq x_j$ and $j = 1, 2, \dots, n$
> where $x_1, x_2, \dots, x_n$ are the data points and $y_1, y_2, \dots, y_n$ are the corresponding function values. The $M_i$ are the second derivatives of the function at the data points, for $i = 0, 1, 2, \dots, n$.
>
> To compute the second derivatives, the tridiagonal matrix algorithm is used. The algorithm is given by:
> $$M_0 = 0$$
> $$M_n = 0$$
> $$M_i = \frac{(x_j-x_{j-1})*M_{j-1}}{6} + \frac{(x_{j+1} - x_{j-1})*M_j}{3} + \frac{(x_{j+1}-x_j)*M_{j+1}}{6} = \frac{y_{j+1}-y_j}{x_{j+1}-x_j} - \frac{y_j-y_{j-1}}{x_j-x_{j-1}}$$
> for $j = 1, 2, \dots, n-1$

## Usage
Each of the two interpolation methods is implemented in a separate Python file in this directory. To use a method, import the corresponding file and call the function with the data points and the points at which to estimate the function. For example:

```python
from lagrange import lagrange_interpolation
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]
x0 = 2.5
print(lagrange_interpolation(x, y, x0))
```

This will print an estimate of the function $f(x) = x^2$ at $x = 2.5$ using Lagrange interpolation.

## Note
Both methods have their own advantages and disadvantages, and the choice of method depends on the specific problem and the accuracy required. Lagrange interpolation is simpler to implement and can be used for any function, but it can be computationally expensive for large data sets. Spline interpolation is more computationally efficient and can be used for large data sets, but it can be more difficult to implement and is not suitable for all functions. In general, spline interpolation is preferred over Lagrange interpolation because it is more computationally efficient and can be used for large data sets.