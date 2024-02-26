data= data.frame(x = c(1,2,3,4,5,6,7,8,9,10),y=c(2,4,7,7.5,9,12,13,15,18.5,20.2))

plot(data$x,data$y)
abline(data.fit, col = "red")
data.fit = lm(data$y~data$x)
summary(data.fit)

b0 = data.fit$coefficients[1]
b1= data.fit$coefficients[2]

standard_errors <- summary(data.fit)$coefficients[, "Std. Error"]
se_b0 <- standard_errors[1] # Standard error of b0
se_b1 <- standard_errors[2] # Standard error of b1

se_b0
se_b1
alpha = .05
conf_intervals <- confint(data.fit, level = 1- alpha)

# Extract confidence intervals for b0 and b1
conf_interval_b0 <- conf_intervals[1, ]
conf_interval_b1 <- conf_intervals[2, ]

# Print confidence intervals

#b0 conf
print(conf_interval_b0)
#b1 conf
print(conf_interval_b1)

