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

# Function to perform independent two-sample t-test
perform_t_test <- function(data1, data2, alternative = "two.sided") {
  tryCatch({
    test_result <- t.test(data1, data2, alternative = alternative)
    return(test_result)
  }, error = function(e) {
    message("Error in performing t-test: ", e$message)
    return(NULL)
  })
}

# Example usage
Data1 <- c(5,7,8,5,4,5,6,3,4,5,6,7,6,5,4,3,7,8,2,4)
Data2 <- c(8,8,8,9,7,6,9,8,4,6,7,9,8,10,9,4,11,6,7,8)

## Important assumptions
# 1. Both groups are normal
# 2. Both groups are numeric
# 3. Both groups are SRS
# 4. Both groups sd known
# 5. Groups are independent

# Check normality assumption for both groups
normality_result_data1 <- check_normality(Data1)
normality_result_data2 <- check_normality(Data2)

# Define the null and alternative hypotheses
# For this example, we will use:
# H0: mu1 = mu2
# Ha: mu1 < mu2

# Perform independent two-sample t-test
t_test_result <- perform_t_test(Data1, Data2, alternative = "less")
if (!is.null(t_test_result)) {
  cat("Independent two-sample t-test result:\n")
  print(t_test_result)
}
