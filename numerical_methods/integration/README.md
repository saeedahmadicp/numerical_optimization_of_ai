# Integration
The "Integration" directory contains two different methods to numerically compute the definite integral of a function:

## What is numerical integration?
In calculus, integration is the process of finding the area under the curve of a function. In some cases, it may be difficult or impossible to find the integral analytically. Numerical integration refers to the process of approximating the integral of a function using numerical methods.

## Methods
* [Trapezoidal Rule](#trapezoidal-rule)
* [Simpson's Rule](#simpsons-rule)

### Trapezoidal Rule
> The trapezoidal rule is a method used to approximate the definite integral of a function. The method uses a series of trapezoids to approximate the area under the curve of the function. The area of each trapezoid is calculated by multiplying the height of the trapezoid by the average of the base lengths. The areas of the trapezoids are then summed to approximate the area under the curve of the function. The trapezoidal rule is given by:
>
> $$A = h  [\frac{f(x_0)}{2} + f(x_1) + f(x_2) + ... + f(x_{n-1}) + \frac{f(x_n)}{2}]$$
>
> where: 
> * $A$ is the area under the curve of the function
> * $h$ is the step size
> * $f(x)$ is the function
> * $x_0, x_1, x_2, ..., x_n$ are the points at which the function is evaluated

### Simpson's Rule
> Simpson's rule is a method used to approximate the definite integral of a function. The method uses a series of parabolas to approximate the area under the curve of the function. The parabolic functions are calculated by interpolating the function through the Lagrange polynomial of degree 2 $(a, \frac{a + b}{2}, b)$, where $a$ and $b$ are the endpoints of the interval and $f(a)$, $f(\frac{a + b}{2})$, and $f(b)$ are the function values at the endpoints and the midpoint of the interval. The areas of the parabolas are then summed to approximate the area under the curve of the function. The Simpson's rule is given by:
>
> $$A = \frac{h}{3}  [f(x_0) + 4 f(x_1) + 2 f(x_2) + 4 f(x_3) + ... + 2 f(x_{n-2}) + 4 f(x_{n-1}) + f(x_n)]$$
>
> where:
> * $A$ is the area under the curve of the function
> * $h$ is the step size
> * $f(x)$ is the function
> * $x_0, x_1, x_2, ..., x_n$ are the points at which the function is evaluated

## Usage
Each of the two integration methods is implemented in a separate Python file in this directory. To use a method, import the corresponding file and call the function with the function to integrate, the lower limit of integration, the upper limit of integration, and the number of subintervals. For example:

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
Both methods have their own advantages and disadvantages, and the choice of method depends on the specific problem and the accuracy required. The trapezoidal rule is simpler to implement and can be used for any function, but it converges to the true value of the integral slower than Simpson's rule. Simpson's rule is more accurate for functions that are smooth and have no discontinuities, but it can be less accurate for functions with sharp corners or singularities. In general, Simpson's rule is preferred over the trapezoidal rule because it converges to the true value of the integral faster.