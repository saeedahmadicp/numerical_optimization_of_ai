# algorithms/regression/normality_test_linear_regression.R

# Function to plot data and regression line
plot_data_with_regression <- function(x, y) {
  tryCatch({
    plot(x, y, main = "Scatter plot with Regression Line", xlab = "x", ylab = "y")
    modfit <- lm(y ~ x)
    abline(modfit, col = "red")
  }, error = function(e) {
    message("Error in plotting data: ", e$message)
  })
}

# Function to save residuals and plot QQ plot
plot_residuals_qq <- function(modfit) {
  tryCatch({
    residuals <- modfit$residuals
    qqnorm(residuals)
    qqline(residuals)
  }, error = function(e) {
    message("Error in plotting residuals QQ plot: ", e$message)
  })
}

# Function to perform Shapiro-Wilk test on residuals
shapiro_test_residuals <- function(modfit, alpha = 0.15) {
  tryCatch({
    residuals <- modfit$residuals
    shapiro_test_result <- shapiro.test(residuals)
    cat("Shapiro-Wilk test for normality:\n")
    print(shapiro_test_result)
    if (shapiro_test_result$p.value > alpha) {
      cat("The residuals are approximately normally distributed (p > alpha)\n")
    } else {
      cat("The residuals are not approximately normally distributed (p <= alpha)\n")
    }
  }, error = function(e) {
    message("Error in performing Shapiro-Wilk test: ", e$message)
  })
}

# Example usage
data <- data.frame(x = c(1,2,3,4,5,6,7,8,9,10), y = c(2,4,7,7.5,9,12,13,15,18.5,20.2))

# Plot data with regression line
plot_data_with_regression(data$x, data$y)

# Fit linear model
modfit <- lm(y ~ x, data = data)

# Plot residuals QQ plot
plot_residuals_qq(modfit)

# Perform Shapiro-Wilk test on residuals
shapiro_test_residuals(modfit)
