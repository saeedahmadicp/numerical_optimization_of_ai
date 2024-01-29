# Integration

The "Integration" directory contains two different methods for numerically computing the definite integral of a function using Python:

## What is Numerical Integration?
In calculus, integration is the process of finding the area under the curve of a function. Sometimes, it may be difficult or impossible to find this integral analytically. Numerical integration, therefore, refers to the process of approximating the integral of a function using numerical methods.

## Methods
* [Trapezoidal Rule](#trapezoidal-rule)
* [Simpson's Rule](#simpsons-rule)

## Trapezoidal Rule
The trapezoidal rule approximates the definite integral of a function by using a series of trapezoids to estimate the area under the curve. The area under each trapezoid is calculated by multiplying the height (or the interval length) by the average of the base lengths. The formula is given by:

$$A = h  \left[\frac{f(x_0)}{2} + f(x_1) + f(x_2) + ... + f(x_{n-1}) + \frac{f(x_n)}{2}\right]$$

where:
- $A$ is the area under the curve of the function
- $h$ is the step size
- $f(x)$ is the function
- $x_0, x_1, x_2, ..., x_n$ are the points at which the function is evaluated.

## Simpson's Rule
Simpson's rule uses parabolic arcs to approximate the area under the curve. The parabolic functions are calculated by interpolating the function through the Lagrange polynomial of degree 2 $(a, \frac{a + b}{2}, b)$, where $a$ and $b$ are the endpoints of the interval and $f(a)$, $f(\frac{a + b}{2})$, and $f(b)$ are the function values at the endpoints and the midpoint of the interval. The areas of these parabolas are then summed to approximate the total area under the curve. The formula for Simpson's rule is:

$$A = \frac{h}{3}  [f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + ... + 2f(x_{n-2}) + 4f(x_{n-1}) + f(x_n)]$$

where:
- $A$ is the area under the curve
- $h$ is the step size
- $f(x)$ is the function
- $x_0, x_1, x_2, ..., x_n$ are the points at which the function is evaluated.

## Usage
Each integration method is implemented in a separate Python file in this directory. To use a method, import the corresponding file and call the function with the function to integrate, the lower and upper limits of integration, and the number of subintervals. For example:

```python
from trapezoidal import trapezoidal
f = lambda x: x**2
a = 0
b = 1
n = 100
print(trapezoidal(f, a, b, n))
```

This will print an approximation of the definite integral of $f = x^2$ from $x = 0$ to $x = 1$ using the trapezoidal rule with 100 subintervals.

## Note
Each method has its own advantages and considerations. The trapezoidal rule is straightforward but may converge slowly for certain functions. Simpson's rule, while generally more accurate for smooth functions, can be less precise for functions with sharp features or discontinuities. The choice of method should be based on the nature of the function and the required accuracy.