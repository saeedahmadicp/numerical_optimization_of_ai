# Fitting a Curve to Data - Regression
This folder contains the following algorithms for fitting a curve to data:

## Algorithms
* [Linear Regression](linear.py)
* [Polynomial Regression](chebyshev.py)

## Linear Regression
Linear regression is a method for fitting a linear curve to data. The linear curve is defined as:

$$y = a x + b$$

where $a$ is the slope of the line and $b$ is the y-intercept. The goal of linear regression is to find the values of $a$ and $b$ that minimize the error between the linear curve and the data. The error between the linear curve and a single data point is given as the y-distance between the linear curve and the data point. The total error is given as the sum of the errors between the linear curve and all of the data points.

Error for a single data point:
$$e = | a x_i + b - y_i|$$

Total error:
$$E = \sum_{i=0}^{n} | a x_i + b - y_i|$$

However, the absolute value function is not differentiable, so we use the squared error instead. The squared error is given as:
$$E = \sum_{i=0}^{n} (a x_i + b - y_i)^2$$

The squared error function is a measure of how well the model fits the data, with lower values indicating a better fit. However, squared error alone can be difficult to interpret, as it is dependent on the scale of the data and the number of data points. To make the squared error easier to interpret, the root mean squared error (MSE) is often used. The MSE is given as:

$$MSE = \frac{1}{n+1} {\sum_{i=0}^{n} (a x_i + b - y_i)^2} = F(a, b)$$

Since we want to minimize MSE, we set the derivate equal to zero to compute the weights. Note that the derivative of the quadratic function has a leading 2 in it, therefore, we divide it by 2 as well to make the math easier. To minimize the error and optimize the weights, we calculate the partial derivatives of the error function with respect to the weights and set them equal to zero. The partial derivatives are given as:

$$\frac{\partial F}{\partial a} = 0$$

$$\frac{\partial F}{\partial b} = 0$$

By solving the above equations, we get the following equations for the weights:

$$a = \dfrac{(n+1) \sum\limits_{i=0}^{n} x_i y_i - \sum\limits_{i=0}^{n} x_i \sum\limits_{i=0}^{n} y_i}{(n+1) \sum\limits_{i=0}^{n} x_i^2 - (\sum\limits_{i=0}^{n} x_i)^2}$$

$$b = \frac{1}{n+1} {\sum_{i=0}^{n} y_i - a \sum_{i=0}^{n} x_i}$$

Example:
Find the linear least square approximation of $f(x)=e^x$ over $[-1,1]$ using 5 data points equally spaced between -1 and 1.

Solution:
Based on the above equations, we calculate the following weights:

$$a = 1.16, b = 1.26$$

Therefore, the linear least square approximation of $f(x)=e^x$ over $[-1,1]$ using 5 data points equally spaced between -1 and 1 is:

$$y = 1.16 x + 1.26$$

## Polynomial Regression
In the above example, the linear fit was not a good approximation of the data. This is because the data is not linear. Instead, the data is exponential. To fit a curve to the data, we can use polynomial regression. One way to to this is to use the least squares approximation using Chebychev polynomials. To approximate a function $f(x)$ using Chebychev polynomials, we use the following equation:

$$f(x) \approx c_0 T_0(x) + c_1 T_1(x) + c_2 T_2(x) + \cdots + c_n T_n(x)$$

where $T_i(x)$ are the basis Chebychev polynomials. To find the basis Chebychev polynomials, we use the following Chebychev recurrence relations:

$$ T_n(x) = cos(n \cdot cos^{-1}(x)), -1 \leq x \leq 1$$

$$ T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x), n \geq 1$$

The first few Chebychev polynomials are given as:

$$T_0(x) = 1$$

$$T_1(x) = x$$

$$T_2(x) = 2x^2 - 1$$

$$T_3(x) = 4x^3 - 3x$$

$$T_4(x) = 8x^4 - 8x^2 + 1$$

### Orthogonality of Chebychev Polynomials
The Chebychev polynomials are orthogonal over the interval $[-1,1]$. This means that the integral of the product of any two Chebychev polynomials is zero. This is given as:

$$
\int_{-1}^{1} T_n(x) T_m(x) \frac{1}{\sqrt{1-x^2}} dx = 
\begin{cases} 
0 & n \neq m \\ 
\pi & n = m = 0 \\ 
\frac{\pi}{2} & n = m \neq 0 
\end{cases}
$$

The coefficients $c_i$ can be found by using the following equation:

$$
c_m = \begin{cases}
\frac{2}{\pi} \int_{-1}^{1} f(x) P_m(x) \frac{1}{\sqrt{1-x^2}} dx & m = 0 \\
\\
\frac{1}{\pi} \int_{-1}^{1} f(x) P_m(x) \frac{1}{\sqrt{1-x^2}} dx & m \neq 0 
\end{cases}
$$

Where $P_m(x)$ is the $m^{th}$ Chebychev basis polynomial.

Example. Find the Chebychev polynomial approximation of $f(x)=e^x$ over $[-1,1]$ using 5 data points equally spaced between -1 and 1.

Solution. The Chebychev polynomial approximation of $f(x)=e^x$ over $[-1,1]$ using 5 data points equally spaced between -1 and 1 is:

$$f(x) \approx c_0 T_0(x) + c_1 T_1(x) + c_2 T_2(x) + c_3 T_3(x) + c_4 T_4(x)$$

where $c_i$ are the coefficients of the Chebychev polynomials. To find the coefficients, we use the above equation for $c_m$. The coefficients are given as:

$$c_0 = 2.53, c_1 = 0.56, c_2 = 0.13, c_3 = 0.02, c_4 = 0.002$$

Therefore, the Chebychev polynomial approximation of $f(x)=e^x$ over $[-1,1]$ using 5 data points equally spaced between -1 and 1 is:

$$f(x) \approx 2.53 + 0.56 x + 0.13 (2x^2 - 1) + 0.02 (4x^3 - 3x) + 0.002 (8x^4 - 8x^2 + 1)$$

$$f(x) \approx 0.016 x^4 + 0.08 x^3 + 0.244 x^2 + 0.5 x + 2.402$$