### Read training data
#! Perhaps you need to set the working directory!?
setwd("D:/VScode/Time_series/ass_1")
D <- read.csv("DST_BIL54.csv")
str(D)

# See the help
?strftime
D$time <- as.POSIXct(paste0(D$time,"-01"), "%Y-%m-%d", tz="UTC")
D$time
class(D$time)

## Year to month for each of them
D$year <- 1900 + as.POSIXlt(D$time)$year + as.POSIXlt(D$time)$mon / 12

## Make the output variable a floating point (i.e.\ decimal number)
D$total <- as.numeric(D$total) / 1E6

## Divide intro train and test set
teststart <- as.POSIXct("2024-01-01", tz="UTC")
Dtrain <- D[D$time < teststart, ]
Dtest <- D[D$time >= teststart, ]

############### PLOT DATA #####################


# It works but does not plot nice plots for some reason
# library(ggplot2)

# ggplot(Dtrain, aes(x = year, y = total)) +
#   geom_line(color = "blue") +
#   labs(
#     x = "Time",
#     y = "Number of vehicles",
#     title = "Number of vehicles in Denmark"
#   ) +
#   theme_minimal()
  
#  ggsave("my_plot.png", width = 10, height = 5, dpi = 300)

library(ggplot2)

# Create the plot with a white background
plot <- ggplot(Dtrain, aes(x = year, y = total)) +
  geom_line(color = "blue") +
  labs(
    x = "Time",
    y = "Number of vehicles",
    title = "Number of vehicles in Denmark"
  ) +
  theme_minimal() +  
  theme(
    plot.background = element_rect(fill = "white", color = NA),  # White plot background
    panel.background = element_rect(fill = "white", color = NA), # White panel background
    legend.background = element_rect(fill = "white", color = NA) # White legend background
  )

# Save the plot with white background
ggsave("my_plot.png", plot = plot, bg = "white", width = 10, height = 5, dpi = 300)


##############################################

################# MODEL ######################

# Specify the population parameters
mu <- 0
sigma <- 1
# Number of observations
n <- 15000
# Randomly sample n observations the normal distribution
e_t <- rnorm(n = n, mean = mu, sd = sigma)

str(e_t)
##############################################

#2.2
# Output variable
y <- Dtrain$total

# First model, only an intercept, hence the design matrix
X <- cbind(1, Dtrain$year)

N <- nrow(X)
p <- ncol(X) 

# The parameter estimates
thetahatOLS <- solve(t(X) %*% X) %*% t(X) %*% y

# prebuilt function
# Use lm for the same
fit <- lm(total ~ year, Dtrain)
summary(fit)

y_OLS <- X %*% thetahatOLS
residuals_OLS <- y - y_OLS
Norm_residuals_OLS <- t(residuals_OLS)%*%(residuals_OLS) # Unweighted LS estimates For the case of no weights we have Σ = I.

theta_var <- as.numeric(Norm_residuals_OLS/(N-p))
cov_theta <- theta_var * solve(t(X) %*% X)




# ggplot(Dtrain, aes(x = year, y = total)) +
#   geom_point(color = "blue") +
#   labs(
#     x = "Time",
#     y = "Number of vehicles",
#     title = "Number of vehicles in Denmark"
#   ) + geom_line(aes(y = y_OLS), color="#d61313", linetype="twodash")+
#   theme_minimal()

# Assign the plot to a variable
plot <- ggplot(Dtrain, aes(x = year, y = total)) +
  geom_point(color = "blue") +
  labs(
    x = "Time",
    y = "Number of vehicles",
    title = "Number of vehicles in Denmark and OLS approximation"
  ) + 
  geom_line(aes(y = y_OLS), color="#d61313", linetype="twodash") +
  theme_minimal()

# Save the plot
ggsave("OLS_vehicles_plot.png", plot = plot, width = 8, height = 6, dpi = 300)

  # ggplot(Dtrain, aes(x = year, y = total)) +
  # geom_line(color = "blue") +
  # geom_line(aes(y = y_OLS), color="#d61313", linetype="twodash") +
  # labs(
  #   x = "Time",
  #   y = "Number of vehicles",
  #   title = "Number of vehicles in Denmark"
  # ) +
  # theme_minimal() +
  # coord_fixed(ratio = 5)  # Adjust this ratio as needed

######  2.3  #####
# Make the next 12 months
year <- 2024 + (0:11) / 12
X_year <- cbind(1, year)

# # Output variable
 y <- Dtrain$total

# # First model, only an intercept, hence the design matrix
 X <- cbind(1, Dtrain$year)

# The parameter estimates
thetahatOLS <- solve(t(X) %*% X) %*% t(X) %*% y

# The predicted values
y_OLS_pred <- X_year %*% thetahatOLS

N <- nrow(X_year)
P <- ncol(X_year) 

t_value <- qt(0.975, df = N-P)  # t-critical value for 95% confidence

# y_OLS <- X %*% thetahatOLS
# residuals_OLS <- y - y_OLS
# Norm_residuals_OLS <- t(residuals_OLS)%*%(residuals_OLS) # Unweighted LS estimates For the case of no weights we have Σ = I.

# theta_var <- as.numeric(Norm_residuals_OLS/(N-p))
# # The confidence interval
# XTX_inv <- solve(t(X) %*% X)
# XT_XTX_inv <- t(X) %*%  XTX_inv

#Conf_interva = t_value*(N-p)*theta_var*sqrt((1+diag(t(X)%*%solve(t(X) %*% X)%*%X)))

# Compute leverage and confidence intervals
leverage <- diag(X_year %*% solve(t(X_year) %*% X_year) %*% t(X_year))
Conf_interval <- t_value * sqrt((N - p) * theta_var * (1 + leverage))

results <- data.frame(
  Observation = 1:N,
  x = round(X_year, 3),
  y_OLS = round(y_OLS_pred, 3),
  Conf_Interval = round(Conf_interval, 3),
  max_value = round(y_OLS_pred + Conf_interval, 3),
  min_value = round(y_OLS_pred - Conf_interval, 3)
)

# Export to CSV
write.csv(results, file = "OLS_Predictions_and_Confidence_Intervals.csv", row.names = FALSE)


########## 2.4 ########## 

# Concatenate years into a single vector
#X_y <- c(Dtrain$year, year)

library(ggplot2)

# Convert the vector into a 1 x n matrix
X_year_total <- D$year
year <- 2024 + (0:11) / 12

#vector if the year 2024
Dnew <- data.frame(year = year)

# Predict with 97.5% confidence interval
new_predictions_ci <- predict(fit, newdata = Dnew, interval = "confidence", level = 0.975)

rest_of_predictions_ci <- predict(fit, newdata = X_year_total, interval = "confidence", level = 0.975)


y_default <- c(y, Dtest$total)
y_default_total <- matrix(y_default, ncol = 1)

Dnew <- data.frame(year = year)

# Predict with 97.5% confidence interval
new_predictions_ci <- predict(fit, newdata = Dnew, interval = "confidence", level = 0.975)


new_results <- data.frame(
  year = year,
  fit = new_predictions_ci[, "fit"],
  lwr = new_predictions_ci[, "lwr"],
  upr = new_predictions_ci[, "upr"]
)



plot <-ggplot() +
  # Plot New Predictions with CI
  geom_line(aes( y = y_default_total,x = X_year_total ), color = "blue", size = 1) +
  geom_line(aes(y = y_OLS, x =Dtrain$year ), color="#d61313", linetype="twodash") +
  # # Plot OLS Predictions with CI
   geom_point(data=new_results, aes(x = year, y = new_predictions_ci[, "fit"]), color = "red", size = 1) +
  geom_ribbon(aes(x = year, ymin = new_predictions_ci[, "lwr"], ymax = new_predictions_ci[, "upr"]), 
              alpha = 0.2, fill = "green") +

  
  # Add labels and title
  labs(x = "Year", y = "Predicted Values", 
       title = "Predictions and Confidence Intervals") +
  theme_minimal()

ggsave("fitted_model_forecast.png", plot = plot, width = 8, height = 6, dpi = 300)


###########2.6##############

total_X <- Dtrain$year
y <- Dtrain$total

# # First model, only an intercept, hence the design matrix
X <- cbind(1, Dtrain$year)

# The parameter estimates
thetahatOLS <- solve(t(X) %*% X) %*% t(X) %*% y

# The predicted values
y_OLS_pred <- X %*% thetahatOLS

residuals <- y - y_OLS_pred
png("residuals_vs_time.png", width = 800, height = 600)  # You can adjust the width and height

# Plot residuals against time
plot(total_X, residuals, 
     type = "o",         # 'o' means overplotted points with lines
     col = "blue",       # Line color
     xlab = "Time",      # x-axis label
     ylab = "Residuals", # y-axis label
     main = "Residuals vs Time",  # Plot title
     pch = 16)           # Point character (solid circle)

# Optionally, add a horizontal line at zero to indicate no residual error
abline(h = 0, col = "red", lty = 2)  # Red dashed line at y=0
dev.off()

png("qq_plot_residuals.png", width = 800, height = 600)  # You can adjust the width and height

# Create the QQ plot
qqnorm(residuals, 
       main = "QQ Plot of Residuals",  # Title
       col = "blue",                   # Color of points
       pch = 16)                       # Point character (solid circle)

# Add a reference line for normality
qqline(residuals, col = "red", lty = 2)  # Red dashed line

# Close the graphic device to save the plot
dev.off()