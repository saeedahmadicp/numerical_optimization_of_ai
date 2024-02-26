# Function to check normality assumption using QQ plot and Shapiro-Wilk test
check_normality <- function(data) {
  tryCatch({
    qqnorm(data)
    qqline(data)
    shapiro_test_result <- shapiro.test(data)
    cat("Shapiro-Wilk test for normality:\n")
    print(shapiro_test_result)
    return(shapiro_test_result)
  }, error = function(e) {
    message("Error in checking normality: ", e$message)
    return(NULL)
  })
}

# Function to perform one-sample t-test
perform_t_test <- function(data, mu, alternative = "greater") {
  tryCatch({
    test_result <- t.test(x = data, mu = mu, alternative = alternative)
    return(test_result)
  }, error = function(e) {
    message("Error in performing t-test: ", e$message)
    return(NULL)
  })
}

# Example usage
Data <- c(20, 23, 22, 21, 19, 18, 20, 19, 24, 23, 22, 21, 20, 17, 15, 20, 18, 20)

## Important assumptions
# 1. Data is numeric
# 2. Data is assumed to be normally distributed
# 3. Sample standard deviation is known
# 4. Sample is randomly selected from the population

# Check normality assumption
normality_result <- check_normality(Data)

# Define the null and alternative hypotheses
mu <- 20
# H0: mu = 20
# Ha: mu > 20

# Perform one-sample t-test
t_test_result <- perform_t_test(Data, mu)
if (!is.null(t_test_result)) {
  cat("One-sample t-test result:\n")
  print(t_test_result)
}
