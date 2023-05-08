# Differentiation
This directory contains three different numerical differentiation methods implemented in Python: forward difference, backward difference, and central difference.

## What is numerical differentiation?
In calculus, differentiation is the process of finding the derivative of a function. In some cases, it may be difficult or impossible to find the derivative analytically. Numerical differentiation refers to the process of approximating the derivative of a function using numerical methods.

## Methods
* [Forward Difference](#forward-difference)
* [Backward Difference](#backward-difference)
* [Central Difference](#central-difference) 

### Forward Difference
> The forward difference method is a numerical method for approximating the derivative of a function at a given point by using the difference between the function values at the given point and a nearby point. Specifically, it approximates the derivative as the ratio of the change in the function value to the change in the input value:
> 
> $$\frac{d}{dx} f(x) = \frac{f(x + h) - f(x)}{h}$$
> 
> where $h$ is a small step size. This method is called "forward" because it uses the value of the function at $x + h$ to approximate the derivative at $x$. The approximation error in the forward difference is given by:
> 
> $$\left|\frac{d}{dx} f(x) - \frac{f(x + h) - f(x)}{h}\right| = \frac{h}{2} f^{(2)}(\xi)$$
> 
> where $\xi$ is a value between $x$ and $x + h$. The approximation error in forward difference is on the order of the step size: $O(h)$.

### Backward Difference
> The backward difference method is similar to the forward difference method, except that it uses the value of the function at $x - h$ to approximate the derivative at $x$:
> 
> $$\frac{d}{dx} f(x) = \frac{f(x) - f(x - h)}{h}$$
> 
> This method is called "backward" because it uses the value of the function at $x - h$ to approximate the derivative at $x$. The approximation error in the backward difference is given by:
> 
> $$\left|\frac{d}{dx} f(x) - \frac{f(x) - f(x - h)}{h}\right| = \frac{h}{2} f^{(2)}(\xi)$$
> 
> where $\xi$ is a value between $x - h$ and $x$. The approximation error in backward difference is also on the order of the step size: $O(h)$.

### Central Difference
> The central difference method is a more accurate numerical method for approximating the derivative of a function at a given point. It approximates the derivative as the difference between the function values at nearby points, divided by the distance between those points:
>
> $$\frac{d}{dx} f(x) = \frac{f(x + h) - f(x - h)}{2h}$$
> 
> This method is called "central" because it uses the values of the function at $x + h$ and $x - h$ to approximate the derivative at $x$. The approximation error in the central difference is given by:
> 
> $$\left|\frac{d}{dx} f(x) - \frac{f(x + h) - f(x - h)}{2h}\right| = \frac{h^2}{12} (f^{(3)}(\xi_1) - f^{(3)}(\xi_2))$$
> 
> where $\xi_1$ is a value between $x - h$ and $x$ and $\xi_2$ is a value between $x$ and $x + h$. The approximation error in central difference is of second order: $O(h^2)$. The central difference is the most accurate of the three methods because as $h$ gets smaller, the approximation error gets smaller faster than the other methods.

## Usage
Each of the three differentiation methods is implemented in a separate Python file in this directory. To use a method, import the corresponding file and call the function with the function to differentiate, the point at which to differentiate, and the step size. For example:

```python
from forward_difference import forward_difference
f = lambda x: x**2
x = 2
h = 0.01
print(forward_difference(f, x, h))
```

This will print an approximation of the derivative of $f = x^2$ at $x = 2$ using the forward difference method with step size $h = 0.01$.

## Note
The step size $h$ should be small, but not too small. If $h$ is too small, the approximation error will be large due to round-off error. If $h$ is too large, the approximation error will be large due to truncation error. The optimal step size depends on the function being differentiated and the method being used. In general, the optimal step size is on the order of the square root of the machine epsilon, which is the smallest number that can be added to 1 and produce a result different from 1. In Python, the machine epsilon can be found using `numpy.finfo(float).eps`.
