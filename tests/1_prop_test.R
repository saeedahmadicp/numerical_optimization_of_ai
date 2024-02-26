# Function to count the number of successes
count_success <- function(data) {
  tryCatch(
    {
      success_count <- length(data[data == 1])
      return(success_count)
    },
    error = function(e) {
      print(paste("Error in counting successes:", e$message))
      return(NULL)
    }
  )
}

# Function to calculate sample size
calculate_sample_size <- function(data) {
  tryCatch(
    {
      sample_size <- length(data)
      return(sample_size)
    },
    error = function(e) {
      print(paste("Error in calculating sample size:", e$message))
      return(NULL)
    }
  )
}

# Function to perform hypothesis test
perform_hypothesis_test <- function(success_count, sample_size, null_prob = 0.5, alternative = "greater") {
  tryCatch(
    {
      hypothesis_test_result <- prop.test(x = success_count, n = sample_size, p = null_prob, alternative = alternative)
      return(hypothesis_test_result)
    },
    error = function(e) {
      print(paste("Error in performing hypothesis test:", e$message))
      return(NULL)
    }
  )
}

# Function to print hypothesis test result
print_hypothesis_test_result <- function(hypothesis_test_result) {
  if (!is.null(hypothesis_test_result)) {
    print("Hypothesis Test Result:")
    print(paste("p-value:", hypothesis_test_result$p.value))
    print(paste("Confidence interval:", hypothesis_test_result$conf.int))
    print(paste("Sample proportion estimate:", hypothesis_test_result$estimate))
  }
}

# Main function to run the analysis
run_analysis <- function(data) {
  # Count successes
  success_count <- count_success(data)
  if (is.null(success_count)) {
    return(NULL)
  }
  print(paste("Number of successes:", success_count))
  
  # Calculate sample size
  sample_size <- calculate_sample_size(data)
  if (is.null(sample_size)) {
    return(NULL)
  }
  print(paste("Sample size:", sample_size))
  
  # Perform hypothesis test
  hypothesis_test_result <- perform_hypothesis_test(success_count, sample_size)
  if (is.null(hypothesis_test_result)) {
    return(NULL)
  }
  print_hypothesis_test_result(hypothesis_test_result)
  return(hypothesis_test_result)
}

# Example usage:
data <- c(1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1)
run_analysis(data)

