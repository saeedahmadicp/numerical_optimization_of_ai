# Fitting a Curve to Data - Regression
This folder contains the following algorithms for fitting a curve to data:

## Algorithms
* [Linear Regression](linear_regression.py)
* [Polynomial Regression](polynomial_regression.py)

## Linear Regression
Linear regression is a method for fitting a linear curve to data. The linear curve is defined as:

$$y = a x + b$$

where $a$ is the slope of the line and $b$ is the y-intercept. The goal of linear regression is to find the values of $a$ and $b$ that minimize the error between the linear curve and the data. The error between the linear curve and a single data point is given as the y-distance between the linear curve and the data point. The total error is given as the sum of the errors between the linear curve and all of the data points.

Error for a single data point:
$$e = | a x_i + b - y_i|$$

Total error:
$$E = \sum_{i=1}^{n} | a x_i + b - y_i|$$

However, the absolute value function is not differentiable, so we use the squared error instead. The squared error is given as:
$$E = \sum_{i=1}^{n} (a x_i + b - y_i)^2$$

The squared error function is a measure of how well the model fits the data, with lower values indicating a better fit. However, squared error alone can be difficult to interpret, as it is dependent on the scale of the data and the number of data points. To make the squared error easier to interpret, the root mean squared error (RMSE) is often used. The RMSE is given as:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (a x_i + b - y_i)^2}  = F(a, b)$$

Since we want to minimize MSE, we set the derivate equal to zero to compute the weights. Note that the derivative of the quadratic function has a leading 2 in it, therefore, we divide it by 2 as well to make the math easier. To minimize the error and optimize the weights, we calculate the partial derivatives of the error function with respect to the weights and set them equal to zero. The partial derivatives are given as:

$$\frac{\partial E}{\partial a} = \frac{1}{2 n} \sum_{i=1}^{n} 2 x_i (a x_i + b - y_i) = 0$$

$$\frac{\partial E}{\partial b} = \frac{1}{2 n} \sum_{i=1}^{n} 2 (a x_i + b - y_i) = 0$$

By solving the above equations, we get the following equations for the weights:

$$a = \frac{n \sum_{i=1}^{n} x_i y_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{n \sum_{i=1}^{n} x_i^2 - (\sum_{i=1}^{n} x_i)^2}$$

$$b = \frac{\sum_{i=1}^{n} y_i - a \sum_{i=1}^{n} x_i}{n}$$
