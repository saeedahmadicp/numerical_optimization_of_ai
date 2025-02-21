# tests/test_regression/2_sample_prop_test.R

# Function to count the number of successes in a dataset
count_success <- function(data) {
  tryCatch({
    success_count <- length(data[data == 1])
    return(success_count)
  }, error = function(e) {
    print("Error in counting successes:", e$message)
    return(NULL)
  })
}

# Function to calculate sample size of a dataset
calculate_sample_size <- function(data) {
  tryCatch({
    sample_size <- length(data)
    return(sample_size)
  }, error = function(e) {
    print("Error in calculating sample size:", e$message)
    return(NULL)
  })
}

# Function to perform hypothesis test comparing two proportions
perform_hypothesis_test <- function(success_counts, sample_sizes, alternative = "greater") {
  tryCatch({
    hypothesis_test_result <- prop.test(x = success_counts, n = sample_sizes, alternative = alternative)
    return(hypothesis_test_result)
  }, error = function(e) {
    print("Error in performing hypothesis test:", e$message)
    return(NULL)
  })
}

# Function to print hypothesis test result
print_hypothesis_test_result <- function(hypothesis_test_result) {
  if (!is.null(hypothesis_test_result)) {
    print("Hypothesis Test Result:")
    print(paste("p-value:", hypothesis_test_result$p.value))
    print(paste("Confidence interval:", hypothesis_test_result$conf.int))
    print(paste("Sample proportion estimates:", hypothesis_test_result$estimate))
  }
}

# Main function to run the analysis
run_analysis <- function(data1, data2) {
  # Count successes and calculate sample size for data1
  l1 <- count_success(data1)
  n1 <- calculate_sample_size(data1)
  if (is.null(l1) || is.null(n1)) {
    return(NULL)
  }
  
  # Count successes and calculate sample size for data2
  l2 <- count_success(data2)
  n2 <- calculate_sample_size(data2)
  if (is.null(l2) || is.null(n2)) {
    return(NULL)
  }
  
  # Perform hypothesis test
  hypothesis_test_result <- perform_hypothesis_test(c(l1, l2), c(n1, n2))
  if (is.null(hypothesis_test_result)) {
    return(NULL)
  }
  
  # Print hypothesis test result
  print_hypothesis_test_result(hypothesis_test_result)
  return(hypothesis_test_result)
}

# Example usage:
data1 <- c(1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2)
data2 <- c(1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1)
run_analysis(data1, data2)
