library(openxlsx)

# Set working directory
setwd("C:/Users/dtass/OneDrive/Desktop/R work")

# Read data from Excel file
tryCatch({
  D <- read.xlsx(xlsxFile = "2_way_example.xlsx")
}, error = function(e) {
  stop("Error reading data: ", e$message)
})

# Create factors
tryCatch({
  x2 <- factor(D[, 2], levels = c("c", "w", "h"))
  x1 <- factor(D[, 1], levels = c("x", "y"))
  y <- D[, 3]
}, error = function(e) {
  stop("Error creating factors: ", e$message)
})

# Make box plot
tryCatch({
  title <- "Dirt removal by detergent and temp"
  x_label <- "group combo"
  y_label <- "dirt removed"
  
  boxplot(y ~ x1 * x2, main = title,
          xlab = x_label,
          ylab = y_label, col = c("Green", "Red"))
}, error = function(e) {
  stop("Error creating box plot: ", e$message)
})

# Make interaction plot
tryCatch({
  interaction.plot(x.factor = x2, trace.factor = x1, response = y,
                   fun = mean, col = c("blue", "red"), trace.label = "detergent")
}, error = function(e) {
  stop("Error creating interaction plot: ", e$message)
})

# Find standard deviations
tryCatch({
  sds <- aggregate(y ~ x1 + x2, FUN = sd)
  print(sds)
}, error = function(e) {
  stop("Error finding standard deviations: ", e$message)
})

# Create ANOVA model
tryCatch({
  model <- aov(y ~ x1 * x2)
  print(summary(model))
}, error = function(e) {
  stop("Error creating ANOVA model: ", e$message)
})

# Check normality
tryCatch({
  RES <- model$residuals
  qqnorm(RES)
  qqline(RES)
  print(shapiro.test(RES))
}, error = function(e) {
  stop("Error checking normality: ", e$message)
})

# Print p-values of ANOVA if data is normal
tryCatch({
  P_vals <- summary(model)[[1]]$`Pr(>F)`
  print(P_vals)
}, error = function(e) {
  stop("Error printing p-values: ", e$message)
})
