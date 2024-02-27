# Function to perform ANOVA
perform_anova <- function(groups, values) {
  tryCatch({
    Res <- aov(values ~ groups)
    return(Res)
  }, error = function(e) {
    print("Error in performing ANOVA:", e$message)
    return(NULL)
  })
}

# Function to summarize ANOVA results
summarize_anova <- function(anova_result) {
  if (!is.null(anova_result)) {
    print(summary(anova_result))
  }
}

# Function to perform Shapiro-Wilk test for normality
perform_shapiro_test <- function(residuals) {
  tryCatch({
    shapiro_test <- shapiro.test(residuals)
    return(shapiro_test)
  }, error = function(e) {
    print("Error in performing Shapiro-Wilk test:", e$message)
    return(NULL)
  })
}

# Main function to run the analysis
run_analysis <- function(groups, values) {
  # Perform ANOVA
  anova_result <- perform_anova(groups, values)
  if (is.null(anova_result)) {
    return(NULL)
  }
  
  # Summarize ANOVA results
  print("ANOVA Summary:")
  summarize_anova(anova_result)
  
  # Check for normality using residuals
  residuals <- residuals(anova_result)
  shapiro_test <- perform_shapiro_test(residuals)
  if (!is.null(shapiro_test)) {
    print("\nShapiro-Wilk Test for Normality:")
    print(shapiro_test)
  }
  
  # Plot Q-Q plot
  print("\nQ-Q Plot of Residuals:")
  qqnorm(residuals)
  qqline(residuals)
  
  # Extract p-value from ANOVA
  p_val <- summary(anova_result)[[1]]$"Pr(>F)"[1]
  print(paste("\nANOVA p-value:", p_val))
  
  return(anova_result)
}

# Example usage:
group_A <- c("A", "A", "A", "A", "A")
group_B <- c("B", "B", "B", "B", "B")
group_C <- c("C", "C", "C", "C", "C")

numbers_A <- c(3, 5, 6, 4, 2)
numbers_B <- c(8, 9, 7, 6, 9)
numbers_C <- c(12, 10, 11, 9, 13)

groups <- factor(c(group_A, group_B, group_C))
values <- c(numbers_A, numbers_B, numbers_C)

run_analysis(groups, values)
