# Regression

This directory contains two Python scripts that implement different regression methods for fitting a curve to a set of data points: linear regression using the least squares method and polynomial regression with Chebyshev polynomials.

## Regression in Machine Learning
Regression is a key technique in machine learning for estimating the relationship between a dependent variable and one or more independent variables. It is instrumental in predicting the value of the dependent variable based on the independent variables. Regression finds applications in diverse fields, enabling the understanding and prediction of complex relationships in data.

## Methods
* [Linear Regression](#linear-regression)
* [Multiple Linear Regression](#multiple-linear-regression)
* [Polynomial Regression](#polynomial-regression)

### Linear Regression
Linear regression is a statistical method to model the relationship between a dependent variable and one or more independent variables. The objective is to find a linear equation that best fits the data. In simple linear regression with one independent variable, the relationship is modeled using a straight line:

$$y = ax + b$$

where \(y\) is the dependent variable, \(x\) is the independent variable, \(a\) is the slope of the line, and \(b\) is the y-intercept. The goal is to find the values of \(a\) and \(b\) that minimize the sum of the squared differences between the observed values and the values predicted by the model. The error between the linear curve and a single data point is given as the y-distance between the linear curve and the data point. The total error is given as the sum of the errors between the linear curve and all of the data points. The error function is given as:

$$E = \sum_{i=0}^{n} (a x_i + b - y_i)^2$$

where \(n\) is the number of data points. The error function is also known as the sum of squared errors (SSE). The goal is to minimize the SSE. To minimize the SSE, we calculate the partial derivatives of the error function with respect to the weights and set them equal to zero. To make the squared error easier to interpret, the root mean squared error (MSE) is often used. The MSE is given as:

$$MSE = \frac{1}{n+1} {\sum_{i=0}^{n} (a x_i + b - y_i)^2} = F(a, b)$$
 
Since we want to minimize MSE, we set the derivate equal to zero to compute the weights. Note that the derivative of the quadratic function has a leading 2 in it, therefore, we divide it by 2 as well to make the math easier. To minimize the error and optimize the weights, we calculate the partial derivatives of the error function with respect to the weights and set them equal to zero. The partial derivatives are given as:

$$\frac{\partial F}{\partial a} = 0$$

$$\frac{\partial F}{\partial b} = 0$$

By solving the above equations, we get the following equations for the weights:

$$a = \dfrac{(n+1) \sum\limits_{i=0}^{n} x_i y_i - \sum\limits_{i=0}^{n} x_i \sum\limits_{i=0}^{n} y_i}{(n+1) \sum\limits_{i=0}^{n} x_i^2 - (\sum\limits_{i=0}^{n} x_i)^2}$$

$$b = \frac{1}{n+1} {\sum_{i=0}^{n} y_i - a \sum_{i=0}^{n} x_i}$$

### Partitioning the Variance
The total sum of squares (SST) is divided into two parts: the regression sum of squares (SSR) and the residual sum of squares (SSE). The regression sum of squares measures the amount of variation in the response that is explained by the regression model. The residual sum of squares measures the amount of variation in the response that is not explained by the regression model.

| Source | DF | SS | MS | F |
| --- | --- | --- | --- | --- |
| Regression | 1 | SSR | MSR = SSR | MSR/MSE |
| Residual | n - 2 | SSE | MSE = SSE/(n - 2) | |
| Total | n - 1 | SST | | |

**Question:** How important is $x$ in predicting $y$?
**Answer:** We can use the $t$-test to test the null hypothesis $H_0: \beta_1 = 0$.

$$H_0: \beta_1 = 0 \quad H_a: \beta_1 \neq 0$$

The test statistic is:

$$t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)}$$

where:

$$SE(\hat{\beta_1}) = \frac{S_{\epsilon}}{\sqrt{S_{xx}}} = \frac{\sqrt{SSE/(n - 2)}}{\sqrt{S_{xx}}}$$

The $p$-value is:

$$p = P(|T| > |t|) = 2P(T > |t|)$$

where $T$ is a $t$-distribution with $n - 2$ degrees of freedom.

**Question:** How well does the model fit the data?
**Answer:** Assess the goodness of fit using the coefficient of determination $R^2$.

The coefficient of determination $R^2$ is the proportion of the variance in the response variable that is explained by the regression model:

$$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$

**Question:** What is the relationship between $R^2$ and the correlation coefficient $r$?
**Answer:** $R^2 = r^2$

### Multiple Linear Regression
Multiple linear regression is a statistical method that allows us to summarize and study relationships between two or more continuous (quantitative) variables:

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_pX_p + \epsilon$$

where:
- $\beta_0$ is the intercept
- $\beta_1, \beta_2, \ldots, \beta_p$ are the slopes
- $\epsilon$ is the error term

The error term $\epsilon$ is assumed to have the following properties:
- $\epsilon$ is a random variable that is normally distributed
- $E(\epsilon) = 0$
- $Var(\epsilon) = \sigma^2$

The goal of multiple linear regression is to minimize the sum of squared residuals (SSR).

$$SSR = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

The least squares estimates of the regression coefficients $\beta_0, \beta_1, \ldots, \beta_p$ are the values that minimize the SSR.

$$\hat{\beta_j} = \frac{\sum_{i=1}^n (x_{ij} - \bar{x_j})(y_i - \bar{y})}{\sum_{i=1}^n (x_{ij} - \bar{x_j})^2} = \frac{S_{xy_j}}{S_{xx_j}} = r_{xy_j}\frac{S_y}{S_{x_j}}$$

$$\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x_1} - \hat{\beta_2}\bar{x_2} - \cdots - \hat{\beta_p}\bar{x_p}$$

**Question:** Are all the predictors useful for predicting $y$?
**Answer:** We can use the $F$-test to test the null hypothesis $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$.

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0 \quad H_a: \text{at least one } \beta_j \neq 0$$

The test statistic is:

$$F = \frac{(SST - SSE)/p}{SSE/(n - p - 1)}$$

where:
- $SST = \sum_{i=1}^n (y_i - \bar{y})^2$
- $SSE = \sum_{i=1}^n (y_i - \hat{y}_i)^2$

The $p$-value is:

$$p = P(F > f)$$

where $F$ is an $F$-distribution with $p$ and $n - p - 1$ degrees of freedom.

**Question:** If the hypothesis testing results in rejecting $H_0$, then which predictors are useful for predicting $y$?

### Backward Elimination (p-value approach)
1. Select a significance level $\alpha$ to stay in the model (e.g. $\alpha = 0.05$).
2. Fit the full model with all possible predictors.
3. Consider the predictor with the highest $p$-value. If $p > \alpha$, go to Step 4. Otherwise, go to Step 5.
4. Remove the predictor.
5. Fit the model without this predictor. Go to Step 3.


### Polynomial Regression
Polynomial regression extends linear regression by considering polynomial functions of the independent variable. Chebyshev polynomials provide a method for polynomial regression that minimizes the maximum error between the data points and the polynomial. This method is particularly useful when the data contains noise or when a more flexible model is required compared to linear regression.

To approximate a function $f(x)$ using Chebychev polynomials, we use the following equation:

$$f(x) \approx c_0 T_0(x) + c_1 T_1(x) + c_2 T_2(x) + \cdots + c_n T_n(x)$$

where $T_i(x)$ are the basis Chebychev polynomials. To find the basis Chebychev polynomials, we use the following Chebychev recurrence relations:

$$T_n(x) = cos(n \cdot cos^{-1}(x)), -1 \leq x \leq 1$$

$$T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x), n \geq 1$$

The first few Chebychev polynomials are given as:

$$T_0(x) = 1$$

$$T_1(x) = x$$

$$T_2(x) = 2x^2 - 1$$

$$T_3(x) = 4x^3 - 3x$$

$$T_4(x) = 8x^4 - 8x^2 + 1$$

### Orthogonality of Chebychev Polynomials
The Chebychev polynomials are orthogonal over the interval $[-1,1]$. This means that the integral of the product of any two Chebychev polynomials is zero. This is given as:

$$\int_{-1}^{1} T_n(x) T_m(x) \frac{1}{\sqrt{1-x^2}} dx = 
\begin{cases} 
0 & n \neq m \\ 
\pi & n = m = 0 \\ 
\frac{\pi}{2} & n = m \neq 0 
\end{cases}
$$

The coefficients $c_i$ can be found by using the following equation:

$$c_m = 
\begin{cases}
\frac{2}{\pi} \int_{-1}^{1} f(x) P_m(x) \frac{1}{\sqrt{1-x^2}} dx & m = 0 \\
\\
\frac{1}{\pi} \int_{-1}^{1} f(x) P_m(x) \frac{1}{\sqrt{1-x^2}} dx & m \neq 0 
\end{cases}
$$

where $P_m(x)$ is the $m^{th}$ Chebychev basis polynomial.

**Example:** Find the Chebychev polynomial approximation of $f(x)=e^x$ over $[-1,1]$ using 5 data points equally spaced between -1 and 1.

The Chebychev polynomial approximation of $f(x)=e^x$ over $[-1,1]$ using 5 data points equally spaced between -1 and 1 is:

$$f(x) \approx c_0 T_0(x) + c_1 T_1(x) + c_2 T_2(x) + c_3 T_3(x) + c_4 T_4(x)$$

where $c_i$ are the coefficients of the Chebychev polynomials. To find the coefficients, we use the above equation for $c_m$. The coefficients are given as:

$$c_0 = 2.53, c_1 = 0.56, c_2 = 0.13, c_3 = 0.02, c_4 = 0.002$$
 
Therefore, the Chebychev polynomial approximation of $f(x)=e^x$ over $[-1,1]$ using 5 data points equally spaced between -1 and 1 is:

$$f(x) \approx 2.53 + 0.56 x + 0.13 (2x^2 - 1) + 0.02 (4x^3 - 3x) + 0.002 (8x^4 - 8x^2 + 1)$$
$$f(x) \approx 0.016 x^4 + 0.08 x^3 + 0.244 x^2 + 0.5 x + 2.402$$

## Note
The choice between linear regression and polynomial regression depends on the nature of the data and the specific requirements of the problem. Linear regression is simpler and may be sufficient for datasets with a linear relationship. Polynomial regression, including methods using Chebyshev polynomials, is more suited for datasets that exhibit a more complex, non-linear relationship.
