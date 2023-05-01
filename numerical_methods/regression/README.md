# Fitting a Curve to Data - Regression
This folder contains the following algorithms for fitting a curve to data:

## Algorithms
* [Linear Regression](linear_regression.py)
* [Polynomial Regression](polynomial_regression.py)

## Linear Regression
Linear regression is a method used to fit a line to a set of data points. The method involves finding the line of best fit (the line that minimizes the sum of the squared errors). The line of best fit is found by minimizing the sum of the squared errors. Consider the following line:

$y = a x + b$

where $y$ is the dependent variable, $x$ is the independent variable, $b$ is the intercept, and $a$ is the slope. The vertical distance between each data point and the regression line is called the error and is given as:

$e = | a x_i + b - y_i|$

The total error is given as:

$E = \sum_{i=1}^{n} e_i = \sum_{i=1}^{n} | a x_i + b - y_i|$

$E = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (a x_i + b - y_i)^2$

The squared error function is a measure of how well the model fits the data, with lower values indicating a better fit. However, squared error alone can be difficult to interpret, as it is dependent on the scale of the data and the number of data points. To make the squared error easier to interpret, the root mean squared error (RMSE) is often used. The RMSE is given as:

$RMSE = \sqrt{\frac{1}{2 n} \sum_{i=1}^{n} (a x_i + b - y_i)^2}  = F(a, b)$

Since we want to minimize MSE, we set the derivate equal to zero to compute the weights. Note that the derivative of the quadratic function has a leading 2 in it, therefore, we divide it by 2 as well to make the math easier. To minimize the error and optimize the weights, we calculate the partial derivatives of the error function with respect to the weights and set them equal to zero. The partial derivatives are given as:

$\frac{\partial E}{\partial a} = \frac{1}{n} \sum_{i=1}^{n} 2 x_i (a x_i + b - y_i) = 0$

$\frac{\partial E}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} 2 (a x_i + b - y_i) = 0$

By solving the above equations, we get the following equations for the weights:

$a = \frac{n \sum_{i=1}^{n} x_i y_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{n \sum_{i=1}^{n} x_i^2 - (\sum_{i=1}^{n} x_i)^2}$

$b = \frac{\sum_{i=1}^{n} y_i - a \sum_{i=1}^{n} x_i}{n}$
