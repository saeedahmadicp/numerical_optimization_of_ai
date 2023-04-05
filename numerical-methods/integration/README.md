# Integration Algorithms
This repository contains numerical integration algorithms implemented in Python.

## Algorithms
* [Trapezoidal Rule](#trapezoidal-rule)
* [Simpson's Rule](#simpsons-rule)

## Trapezoidal Rule
The trapezoidal rule is a method used to approximate the definite integral of a function. The method uses a series of trapezoids to approximate the area under the curve of the function. The area of each trapezoid is calculated by multiplying the height of the trapezoid by the average of the base lengths. The areas of the trapezoids are then summed to approximate the area under the curve of the function. The trapezoidal rule is a first-order method, meaning that the error is proportional to the square of the step size.

The trapezoidal rule is given by:

$A = h  [\frac{f(x_0)}{2} + f(x_1) + f(x_2) + ... + f(x_{n-1}) + \frac{f(x_n)}{2}]$

where: 
* $A$ is the area under the curve of the function
* $h$ is the step size
* $f(x)$ is the function
* $x_0, x_1, x_2, ..., x_n$ are the points at which the function is evaluated

## Simpson's Rule
Simpson's rule is a method used to approximate the definite integral of a function. The method uses a series of parabolas to approximate the area under the curve of the function. The parabolic functions are calculated by interpolating the function through the Lagrange polynomial of degree 2 $(a, \frac{a + b}{2}, b)$, where $a$ and $b$ are the endpoints of the interval and $f(a)$, $f(\frac{a + b}{2})$, and $f(b)$ are the function values at the endpoints and the midpoint of the interval. The areas of the parabolas are then summed to approximate the area under the curve of the function.

The Simpson's rule is given by:

$A = \frac{h}{3}  [f(x_0) + 4 f(x_1) + 2 f(x_2) + 4 f(x_3) + ... + 2 f(x_{n-2}) + 4 f(x_{n-1}) + f(x_n)]$

where:
* $A$ is the area under the curve of the function
* $h$ is the step size
* $f(x)$ is the function
* $x_0, x_1, x_2, ..., x_n$ are the points at which the function is evaluated