# Differentiation Algorithms
This repository contains numerical differentiation algorithms implemented in Python.

## Algorithms
* [Forward Difference](#forward-difference)
* [Backward Difference](#backward-difference)
* [Central Difference](#central-difference)

In the following algorithms, $f(x)$ is the function and $h$ is the step size. 

## Forward Difference
The forward difference is a method used to approximate the derivative of a function. The method uses a series of forward differences to approximate the derivative of the function. The forward difference is given by:

$\frac{d}{dx} f(x) = \frac{f(x + h) - f(x)}{h}$

### Approximation Error in Forward Difference
The approximation error in the forward difference is given by:

$\frac{d}{dx} f(x) - \frac{f(x + h) - f(x)}{h} = \frac{h f^{(2)}(\xi)}{2}$

where $\xi$ is a number between $x$ and $x + h$. The approximation error in forward difference is on the order of the step size: $O(h)$.

## Backward Difference
The backward difference is a method used to approximate the derivative of a function. The method uses a series of backward differences to approximate the derivative of the function. The backward difference is given by:

$\frac{d}{dx} f(x) = \frac{f(x) - f(x - h)}{h}$

### Approximation Error in Backward Difference
The approximation error in the backward difference is given by:

$\frac{d}{dx} f(x) - \frac{f(x) - f(x - h)}{h} = \frac{h f^{(2)}(\xi)}{2}$

where $\xi$ is a number between $x - h$ and $x$. The approximation error in backward difference is on the order of the step size: $O(h)$.

## Central Difference
The central difference is a method used to approximate the derivative of a function. The method uses a series of central differences to approximate the derivative of the function. The central difference is given by:

$\frac{d}{dx} f(x) = \frac{f(x + h) - f(x - h)}{2h}$

### Approximation Error in Central Difference
The approximation error in the central difference is given by:

$\frac{d}{dx} f(x) - \frac{f(x + h) - f(x - h)}{2h} = \frac{h^2 f^{(3)}(\xi)}{12}$

where $\xi$ is a number between $x - h$ and $x + h$. The approximation error in central difference is of second order: $O(h^2)$. The central difference is the most accurate of the three methods because as $h$ gets smaller, the approximation error gets smaller faster than the other methods.