# Numerical Optimization of AI

This repository contains numerical optimization algorithms for AI. The algorithms are primarily implemented in Python.

## Algorithms
1. [Bisection Method](#bisection-method)
2. [Newton's Method](#newtons-method)
3. [Secant Method](#secant-method)
4. [Regula Falsi Method](#regula-falsi-method)
5. [Elimination Method](#elimination-method)
6. [Fibonacci Search](#fibonacci-search)
7. [Golden Section Search](#golden-section-search)


#### Usage
```python
from methods import regula_falsi_method
root, error, iterations = regula_falsi_method(function, lower_bound, upper_bound, tolerance, max_iter)
```

#### Example
```python
from methods import regula_falsi_method
root, error, iterations = regula_falsi_method(lambda x: x**3 - 2*x - 5, 1, 2, 0.0001, 100)
print(root, error, iterations)
```

### Bisection Method
----------------
The bisection method is a root-finding method that applies to any continuous function for which one knows two values with opposite signs. The method consists of repeatedly bisecting the interval defined by these values and then selecting the subinterval in which the function changes sign, and therefore must contain a root. The algorithm applies to any continuous function $f(x)$ on an interval $[a,b]$ where the value of the function $f(x)$ changes sign from $a$ to $b$. The algorithm begins with an interval $[a,b]$ that contains a root, and at each iteration, the interval is bisected into two subintervals, and the subinterval containing the root is selected. This process is repeated until the desired level of accuracy is reached, or until a maximum number of iterations is reached. It is a very simple and robust method, but it is also relatively slow.

### Newton's Method
---------------
Newton's method is a root-finding method that uses the first derivative of a function to find successively better approximations to the roots (or zeroes) of a real-valued function. It is one of the most widely used algorithms for finding numerical solutions of equations. Given a function $f(x),$ the method starts with an initial guess $x_0$ and then iteratively refines this guess using the formula:

> $x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$

where $f'(x_n)$ is the derivative of $f$ evaluated at $x_n$. This formula gives the equation of the tangent line to the graph of $f$ at the point $(x_n, f(x_n))$, and $x_{n+1}$ is the $x$-intercept of this tangent line. In other words, we are finding the point where the tangent line to the graph of $f$ crosses the $x$-axis and using that as our new guess for the root.

### Secant Method
-------------
The secant method is a root-finding algorithm that uses a succession of secant lines to approximate the root of a function $f(x)$. It is similar to the Newton-Raphson method, but instead of using the derivative of the function, it approximates the derivative using the slope of a secant line between two points. The secant method is a variant of Newton's method that avoids the use of the derivative of $f(x)$ — which can be very helpful when dealing with the derivative is not easy. It avoids the use of the derivative by approximating $f'(x)$ by $f(x+h)−f(x)/h$ for some $h$. The secant method begins with two initial guesses, $x_0$ and $x_1$, which do not need to bracket the root. It then uses these two points to form the first secant line, which intersects the $x$-axis at a point $x_2$. This new point $x_2$ is then used along with $x_1$ to form the next secant line, which intersects the $x$-axis at $x_3$, and so on. The process continues until the desired level of accuracy is achieved, or until a maximum number of iterations is reached.

### Regula Falsi Method
-------------------
The Regula Falsi method, also known as the false position method, is a numerical algorithm used to find the roots of a given equation. This method iteratively narrows down the possible location of the root within a given interval, using linear interpolation to select the subinterval in which the root lies. The algorithm starts with two initial guesses, and at each iteration, a linear function is used to interpolate the root, which is then used to update the interval. The Regula Falsi method is based on the Intermediate Value Theorem, which states that if a continuous function $f(x)$ has opposite signs at two points $a$ and $b$, then there exists at least one root of the equation $f(x) = 0$ between $a$ and $b$. The algorithm terminates when the desired level of accuracy is reached. Unlike the Secant Method, one interval always remains constant in this method.

### Elimination Method
----------------
The elimination method is a root-finding algorithm that uses a series of elimination steps to find the roots of a function. It begins with an interval containing the root, and then applies a series of elimination steps to narrow down the interval containing the root. The process is repeated until the desired level of accuracy is achieved, or until a maximum number of iterations is reached. 

### Fibonacci Search
----------------
The Fibonacci Search is a golden section search algorithm that divides an interval into Fibonacci subintervals and eliminates one of the subintervals based on the function values at the endpoints. The algorithm starts with an initial interval, and at each iteration, the interval is divided into Fibonacci subintervals, and the subinterval containing the lower function value is eliminated. This process is repeated until the desired level of accuracy is reached. Read more about the Fibonacci Search [here](https://www.sciencedirect.com/science/article/pii/S0096300302008287).

### Golden Section Search
--------------------
The golden section search is a root-finding algorithm that uses the golden ratio to efficiently narrow down the search interval to find the minimum or maximum of a function. The algorithm works by dividing the search interval into two parts in such a way that the ratio of the longer part to the shorter part is equal to the golden ratio. The function is then evaluated at the two new points created by this division, and the interval is updated to exclude the point that gives the higher function value. This process is repeated until the interval becomes small enough to yield a satisfactory approximation to the minimum or maximum of the function. Read more about the Golden Section Search [here](https://docest.com/golden-section-search-method).
