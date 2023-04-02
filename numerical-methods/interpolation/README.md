# Interpolation Algorithms
This directory contains implementations of various interpolation algorithms.

## Algorithms
1. [Lagrange Interpolation](#lagrange-interpolation)
2. [Cubic Spline Interpolation](#cubic-spline-interpolation)


## Lagrange Interpolation
The Lagrange interpolation method constructs a polynomial that passes through all the data points provided. The polynomial is constructed by multiplying a series of Lagrange basis polynomials, which are defined as:

![Lagrange Basis Polynomial](https://latex.codecogs.com/gif.latex?L_%7Bi%7D%28x%29%20%3D%20%5Cprod_%7Bj%20%5Cneq%20i%7D%5E%7Bn%7D%20%5Cfrac%7Bx%20-%20x_%7Bj%7D%7D%7Bx_%7Bi%7D%20-%20x_%7Bj%7D%7D)

where ![n](https://latex.codecogs.com/gif.latex?n) is the number of data points, ![x_i](https://latex.codecogs.com/gif.latex?x_i) is the ![i](https://latex.codecogs.com/gif.latex?i)th data point, and ![x_j](https://latex.codecogs.com/gif.latex?x_j) is the ![j](https://latex.codecogs.com/gif.latex?j)th data point.

The Lagrange interpolation polynomial is then defined as:

![Lagrange Interpolation Polynomial](https://latex.codecogs.com/gif.latex?P_%7Bn%7D%28x%29%20%3D%20%5Csum_%7Bi%3D0%7D%5E%7Bn%7D%20y_%7Bi%7DL_%7Bi%7D%28x%29)

where ![y_i](https://latex.codecogs.com/gif.latex?y_i) is the ![i](https://latex.codecogs.com/gif.latex?i)th data point.

The Lagrange interpolation method is very simple to implement, but it is not very efficient. The time complexity of the algorithm is ![O(n^2)](https://latex.codecogs.com/gif.latex?O%28n%5E2%29), which means that the algorithm becomes very slow as the number of data points increases.

## Cubic Spline Interpolation
The cubic spline interpolation method constructs a piecewise function that consists of multiple cubic polynomials to approximate the data. The cubic polynomials are defined as:

![Cubic Polynomial](https://latex.codecogs.com/gif.latex?P_%7Bi%7D%28x%29%20%3D%20a_%7Bi%7D%20&plus;%20b_%7Bi%7D%28x%20-%20x_%7Bi%7D%29%20&plus;%20c_%7Bi%7D%28x%20-%20x_%7Bi%7D%29%5E2%20&plus;%20d_%7Bi%7D%28x%20-%20x_%7Bi%7D%29%5E3)

where ![a_i](https://latex.codecogs.com/gif.latex?a_i), ![b_i](https://latex.codecogs.com/gif.latex?b_i), ![c_i](https://latex.codecogs.com/gif.latex?c_i), and ![d_i](https://latex.codecogs.com/gif.latex?d_i) are constants that are determined by the data points.

The cubic polynomials are connected together at the data points to form a piecewise function. The first and last cubic polynomials are defined as:

![First Cubic Polynomial](https://latex.codecogs.com/gif.latex?P_%7B0%7D%28x%29%20%3D%20a_%7B0%7D%20&plus;%20b_%7B0%7D%28x%20-%20x_%7B0%7D%29%20&plus;%20c_%7B0%7D%28x%20-%20x_%7B0%7D%29%5E2%20&plus;%20d_%7B0%7D%28x%20-%20x_%7B0%7D%29%5E3)

![Last Cubic Polynomial](https://latex.codecogs.com/gif.latex?P_%7Bn-1%7D%28x%29%20%3D%20a_%7Bn-1%7D%20&plus;%20b_%7Bn-1%7D%28x%20-%20x_%7Bn-1%7D%29%20&plus;%20c_%7Bn-1%7D%28x%20-%20x_%7Bn-1%7D%29%5E2%20&plus;%20d_%7Bn-1%7D%28x%20-%20x_%7Bn-1%7D%29%5E3)

where ![n](https://latex.codecogs.com/gif.latex?n) is the number of data points.

The remaining cubic polynomials are defined as:

![Remaining Cubic Polynomial](https://latex.codecogs.com/gif.latex?P_%7Bi%7D%28x%29%20%3D%20a_%7Bi%7D%20&plus;%20b_%7Bi%7D%28x%20-%20x_%7Bi%7D%29%20&plus;%20c_%7Bi%7D%28x%20-%20x_%7Bi%7D%29%5E2%20&plus;%20d_%7Bi%7D%28x%20-%20x_%7Bi%7D%29%5E3)

where ![i](https://latex.codecogs.com/gif.latex?i) is the index of the data point that the cubic polynomial passes through.

The cubic spline interpolation method is more efficient than the Lagrange interpolation method. The time complexity of the algorithm is ![O(n)](https://latex.codecogs.com/gif.latex?O%28n%29), which means that the algorithm is much faster as the number of data points increases.

## References
1. [Lagrange Interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial)
2. [Cubic Spline Interpolation](https://en.wikipedia.org/wiki/Spline_interpolation)