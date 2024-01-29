# Differentiation

This directory contains three distinct numerical differentiation methods implemented in Python: forward difference, backward difference, and central difference.

## What is Numerical Differentiation?
In calculus, differentiation is the process of finding the derivative of a function. In some scenarios, particularly when dealing with complex or data-derived functions, it may be difficult or impossible to find the derivative analytically. Numerical differentiation comes into play here, providing a means to approximate the derivative of a function using numerical methods.

## Methods
* [Forward Difference](#forward-difference)
* [Backward Difference](#backward-difference)
* [Central Difference](#central-difference)

### Forward Difference
The forward difference method approximates the derivative of a function at a given point using the difference between the function values at that point and a nearby point. Specifically, it approximates the derivative as the ratio of the change in the function value to the change in the input value:

$$\frac{d}{dx} f(x) = \frac{f(x + h) - f(x)}{h}$$

where $h$ is a small step size. This method is called "forward" because it uses the value of the function at $x + h$ to approximate the derivative at $x$.

The approximation error in the forward difference is given by:

$$\left|\frac{d}{dx} f(x) - \frac{f(x + h) - f(x)}{h}\right| = \frac{h}{2} f^{(2)}(\xi)$$

where $\xi$ is a value between $x$ and $x + h$. The approximation error in forward difference is on the order of the step size: $O(h)$. This means that as $h$ gets smaller, the approximation error gets smaller, but not as fast as the central difference method.

### Backward Difference
The backward difference method uses the function value at a point slightly before the point of interest to approximate the derivative. Specifically, it approximates the derivative as the ratio of the change in the function value to the change in the input value:
    
$$\frac{d}{dx} f(x) = \frac{f(x) - f(x - h)}{h}$$

where $h$ is a small step size. This method is called "backward" because it uses the value of the function at $x - h$ to approximate the derivative at $x$.

The approximation error in the backward difference is given by:

$$\left|\frac{d}{dx} f(x) - \frac{f(x) - f(x - h)}{h}\right| = \frac{h}{2} f^{(2)}(\xi)$$

where $\xi$ is a value between $x - h$ and $x$. The approximation error in backward difference is also on the order of the step size: $O(h)$. This means that as $h$ gets smaller, the approximation error gets smaller, but not as fast as the central difference method.

### Central Difference
The central difference method provides a more accurate approximation by taking the average of the forward and backward differences. Specifically, it approximates the derivative as the average of the forward and backward differences:

$$\frac{d}{dx} f(x) = \frac{f(x + h) - f(x - h)}{2h}$$

where $h$ is a small step size. This method is called "central" because it uses the values of the function at $x + h$ and $x - h$ to approximate the derivative at $x$.

The approximation error in the central difference is given by:

$$\left|\frac{d}{dx} f(x) - \frac{f(x + h) - f(x - h)}{2h}\right| = \frac{h^2}{12} (f^{(3)}(\xi_1) - f^{(3)}(\xi_2))$$

where $\xi_1$ is a value between $x - h$ and $x$ and $\xi_2$ is a value between $x$ and $x + h$. The approximation error in central difference is of second order: $O(h^2)$. The central difference is the most accurate of the three methods because as $h$ gets smaller, the approximation error gets smaller faster than the other methods.

## Usage
Each method is implemented in a separate Python file within this directory. To use a method, import the corresponding file and call the function with the desired function, point of differentiation, and step size. For example:

```python
from forward_difference import forward_difference
f = lambda x: x**2
x = 2
h = 0.01
print(forward_difference(f, x, h))
```
This code approximates the derivative of $f = x^2$ at $x = 2$ using the forward difference method with step size $h = 0.01$.

## Note
Choosing the appropriate step size $h$ is crucial. Too small a step size can lead to significant round-off errors, while too large a step size may introduce truncation errors. The optimal step size typically depends on the specific function and method used. A general guideline is to use a step size on the order of the square root of the machine epsilon, which represents the smallest difference discernible from 1 in the computer's arithmetic system. In Python, this value can be obtained using `np.finfo(float).eps`.