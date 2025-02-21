# algorithms/regression/conf_int.R

# Function to plot data and regression line
plot_data_with_regression <- function(x, y) {
  tryCatch({
    plot(x, y, main = "Scatter plot with Regression Line", xlab = "x", ylab = "y")
    data.fit <- lm(y ~ x)
    abline(data.fit, col = "red")
  }, error = function(e) {
    message("Error in plotting data: ", e$message)
  })
}

# Function to compute linear regression and print summary
compute_regression_summary <- function(x, y) {
  tryCatch({
    data.fit <- lm(y ~ x)
    summary(data.fit)
  }, error = function(e) {
    message("Error in computing regression summary: ", e$message)
    return(NULL)
  })
}

# Function to compute and print standard errors
compute_standard_errors <- function(data.fit) {
  tryCatch({
    standard_errors <- summary(data.fit)$coefficients[, "Std. Error"]
    se_b0 <- standard_errors[1] # Standard error of b0
    se_b1 <- standard_errors[2] # Standard error of b1
    cat("Standard Error of b0:", se_b0, "\n")
    cat("Standard Error of b1:", se_b1, "\n")
  }, error = function(e) {
    message("Error in computing standard errors: ", e$message)
  })
}

# Function to compute and print confidence intervals
compute_confidence_intervals <- function(data.fit, alpha = 0.05) {
  tryCatch({
    conf_intervals <- confint(data.fit, level = 1 - alpha)
    conf_interval_b0 <- conf_intervals[1, ]
    conf_interval_b1 <- conf_intervals[2, ]
    cat("Confidence interval for b0:\n", conf_interval_b0, "\n")
    cat("Confidence interval for b1:\n", conf_interval_b1, "\n")
  }, error = function(e) {
    message("Error in computing confidence intervals: ", e$message)
  })
}

# Example usage
data <- data.frame(x = c(1,2,3,4,5,6,7,8,9,10), y = c(2,4,7,7.5,9,12,13,15,18.5,20.2))

# Plot data with regression line
plot_data_with_regression(data$x, data$y)

# Compute and print regression summary
summary <- compute_regression_summary(data$x, data$y)
if (!is.null(summary)) {
  print(summary)
}

# Compute and print standard errors
data.fit <- lm(data$y ~ data$x)
compute_standard_errors(data.fit)

# Compute and print confidence intervals
compute_confidence_intervals(data.fit)
