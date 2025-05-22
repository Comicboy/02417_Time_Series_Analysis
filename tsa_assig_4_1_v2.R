# Set random seed for reproducibility
set.seed(123)

# Parameters
a <- 0.9  # State transition coefficient
b <- 1    # Bias term
sigma1 <- 1  # Standard deviation of process noise
sigma2 <- 1  # Standard deviation of observation noise
n <- 100  # Number of time steps
X0 <- 5   # Initial value

### Task 1.1: Simulate 5 independent realizations of the process ###

# Number of realizations
num_realizations <- 5

# Initialize a matrix to store the trajectories
trajectories <- matrix(0, nrow = n, ncol = num_realizations)

# Simulate 5 independent realizations
for (i in 1:num_realizations) {
  X <- numeric(n)  # Initialize the state vector
  X[1] <- X0       # Set the initial value
  for (t in 2:n) {
    X[t] <- a * X[t - 1] + b + rnorm(1, mean = 0, sd = sigma1)
  }
  trajectories[, i] <- X
}

# Plot all 5 trajectories
plot(1:n, trajectories[, 1], type = "l", col = "blue", lwd = 2, ylim = range(trajectories),
     xlab = "Time", ylab = "State (X_t)", main = "5 Independent Realizations of the Process")
for (i in 2:num_realizations) {
  lines(1:n, trajectories[, i], col = i, lwd = 2)
}
legend("topright", legend = paste("Realization", 1:num_realizations), col = 1:num_realizations, lty = 1, lwd = 2)

### Task 1.2: Simulate a single realization with observations ###

# Initialize vectors for the latent state (X_t) and observations (Y_t)
X <- numeric(n)
Y <- numeric(n)

# Set the initial value
X[1] <- X0
Y[1] <- X[1] + rnorm(1, mean = 0, sd = sigma2)

# Simulate the process and observations
for (t in 2:n) {
  X[t] <- a * X[t - 1] + b + rnorm(1, mean = 0, sd = sigma1)  # Latent state
  Y[t] <- X[t] + rnorm(1, mean = 0, sd = sigma2)             # Observation
}

# Plot the latent state and observations
plot(1:n, X, type = "l", col = "blue", lwd = 2, ylim = range(c(X, Y)),
     xlab = "Time", ylab = "Value", main = "Latent State and Observations")
lines(1:n, Y, col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("Latent State (X_t)", "Observations (Y_t)"),
       col = c("blue", "red"), lty = c(1, 2), lwd = 2)

# 1.3
# Load required library
library(ggplot2)

# Set random seed for reproducibility
set.seed(123)

# Parameters
a <- 0.9  # State transition coefficient
b <- 1    # Bias term
sigma1 <- 1  # Standard deviation of process noise
sigma2 <- 1  # Standard deviation of observation noise
n <- 100  # Number of time steps
X0 <- 5   # Initial value

# Simulate the latent state and observations (from Task 1.2)
X <- numeric(n)
Y <- numeric(n)

X[1] <- X0
Y[1] <- X[1] + rnorm(1, mean = 0, sd = sigma2)

for (t in 2:n) {
  X[t] <- a * X[t - 1] + b + rnorm(1, mean = 0, sd = sigma1)  # Latent state
  Y[t] <- X[t] + rnorm(1, mean = 0, sd = sigma2)             # Observation
}

### Completed Kalman Filter Function ###
myKalmanFilter <- function(
  y,             # Vector of observations y_t
  theta,         # Model parameters for X_{t+1} = a*X_t + b + c*e_t
  R,             # Measurement noise variance
  x_prior = 0,   # Initial prior mean for X_0
  P_prior = 10   # Initial prior variance for X_0
) {
  # Unpack model parameters
  a <- theta[1]  # State transition coefficient
  b <- theta[2]  # Bias term
  sigma1 <- theta[3]  # Process noise standard deviation

  # Number of observations
  N <- length(y)

  # Initialize vectors to store results
  x_pred  <- numeric(N)  # Predicted means
  P_pred  <- numeric(N)  # Predicted variances
  x_filt  <- numeric(N)  # Filtered means
  P_filt  <- numeric(N)  # Filtered variances
  innovation     <- numeric(N)  # Pre-fit residuals: y[t] - x_pred[t]
  innovation_var <- numeric(N)  # Innovation covariance: P_pred[t] + R

  # Kalman filter loop
  for (t in seq_len(N)) {
    # Prediction step
    if (t == 1) {
      # Use the prior for the first step
      x_pred[t] <- x_prior
      P_pred[t] <- P_prior + sigma1^2
    } else {
      # Use the previous filtered estimate for subsequent steps
      x_pred[t] <- a * x_filt[t - 1] + b
      P_pred[t] <- a^2 * P_filt[t - 1] + sigma1^2
    }

    # Update step
    innovation[t] <- y[t] - x_pred[t]  # Prediction error
    innovation_var[t] <- P_pred[t] + R  # Prediction error variance
    K_t <- P_pred[t] / innovation_var[t]  # Kalman gain
    x_filt[t] <- x_pred[t] + K_t * innovation[t]  # Filtered estimate
    P_filt[t] <- (1 - K_t) * P_pred[t]  # Filtered estimate variance
  }

  # Return results as a list
  return(list(
    x_pred = x_pred,
    P_pred = P_pred,
    x_filt = x_filt,
    P_filt = P_filt,
    innovation = innovation,
    innovation_var = innovation_var
  ))
}

# Apply the Kalman filter to the simulated data
theta <- c(a, b, sigma1)  # Model parameters
R <- sigma2^2  # Observation noise variance
x_prior <- 0   # Initial prior mean
P_prior <- 10  # Initial prior variance

kf_results <- myKalmanFilter(Y, theta, R, x_prior, P_prior)

# Extract results
x_pred <- kf_results$x_pred
P_pred <- kf_results$P_pred

# Compute 95% confidence intervals
ci_upper <- x_pred + 1.96 * sqrt(P_pred)
ci_lower <- x_pred - 1.96 * sqrt(P_pred)

# Create a data frame for ggplot
data <- data.frame(
  Time = 1:n,
  LatentState = X,
  Observations = Y,
  PredictedState = x_pred,
  CI_Upper = ci_upper,
  CI_Lower = ci_lower
)

# Plot using ggplot2
ggplot(data, aes(x = Time)) +
  # Latent state (ground truth)
  geom_line(aes(y = LatentState, color = "Latent State (X_t)"), size = 1) +
  # Observations
  geom_point(aes(y = Observations, color = "Observations (Y_t)"), size = 1.5, alpha = 0.6) +
  # Predicted state
  geom_line(aes(y = PredictedState, color = "Predicted State (x_pred)"), size = 1) +
  # Confidence intervals
  geom_ribbon(aes(ymin = CI_Lower, ymax = CI_Upper, fill = "95% CI"), alpha = 0.2) +
  # Customize colors
  scale_color_manual(values = c("Latent State (X_t)" = "blue",
                                 "Observations (Y_t)" = "red",
                                 "Predicted State (x_pred)" = "green")) +
  scale_fill_manual(values = c("95% CI" = "green")) +
  # Labels and theme
  labs(title = "Kalman Filter Results",
       x = "Time",
       y = "Value",
       color = "Legend",
       fill = "Legend") +
  theme_minimal() +
  theme(legend.position = "top",
        legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10))

# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)

# Set random seed for reproducibility
set.seed(123)

### Kalman Filter Function ###
myKalmanFilter <- function(y, theta, R, x_prior = 0, P_prior = 10) {
  # Unpack parameters
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]

  # Number of observations
  N <- length(y)

  # Initialize vectors to store results
  x_pred  <- numeric(N)  # Predicted means
  P_pred  <- numeric(N)  # Predicted variances
  x_filt  <- numeric(N)  # Filtered means
  P_filt  <- numeric(N)  # Filtered variances
  innovation     <- numeric(N)  # Pre-fit residuals: y[t] - x_pred[t]
  innovation_var <- numeric(N)  # Innovation covariance: P_pred[t] + R

  # Kalman filter loop
  for (t in seq_len(N)) {
    # Prediction step
    if (t == 1) {
      # Use the prior for the first step
      x_pred[t] <- x_prior
      P_pred[t] <- P_prior + sigma1^2
    } else {
      # Use the previous filtered estimate for subsequent steps
      x_pred[t] <- a * x_filt[t - 1] + b
      P_pred[t] <- a^2 * P_filt[t - 1] + sigma1^2
    }

    # Ensure predicted variance is positive
    P_pred[t] <- max(P_pred[t], 1e-6)

    # Update step
    innovation[t] <- y[t] - x_pred[t]  # Prediction error
    innovation_var[t] <- P_pred[t] + R  # Prediction error variance

    # Ensure innovation variance is positive
    innovation_var[t] <- max(innovation_var[t], 1e-6)

    K_t <- P_pred[t] / innovation_var[t]  # Kalman gain
    x_filt[t] <- x_pred[t] + K_t * innovation[t]  # Filtered estimate
    P_filt[t] <- (1 - K_t) * P_pred[t]  # Filtered estimate variance
  }

  # Return results as a list
  return(list(
    x_pred = x_pred,
    P_pred = P_pred,
    x_filt = x_filt,
    P_filt = P_filt,
    innovation = innovation,
    innovation_var = innovation_var
  ))
}

### Log-Likelihood Function ###
myLogLikFun <- function(theta, y, R, x_prior = 0, P_prior = 10) {
  # Unpack parameters
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]

  # Ensure sigma1 is positive
  if (sigma1 <= 0) {
    return(Inf)  # Return a large value for invalid parameters
  }

  # Call the Kalman filter function
  kf_result <- myKalmanFilter(y, theta, R, x_prior, P_prior)

  # Extract innovations and their variances
  err <- kf_result$innovation       # Innovations (y[t] - x_pred[t])
  S <- kf_result$innovation_var     # Innovation variances (P_pred[t] + R)

  # Ensure all innovation variances are positive
  if (any(S <= 0)) {
    return(Inf)  # Return a large value for invalid computations
  }

  # Compute log-likelihood contributions from each time step
  logL <- -0.5 * sum(log(2 * pi * S) + (err^2 / S))

  return(-logL)
}

### Parameter Estimation Function ###
estimate_parameters <- function(y, R, x_prior = 0, P_prior = 10) {
  # Initial guesses for parameters
  theta_init <- c(1, 1, 1)  # Initial guesses for a, b, sigma1

  # Minimize the negative log-likelihood
  result <- optim(
    par = theta_init,
    fn = myLogLikFun,
    y = y,
    R = R,
    x_prior = x_prior,
    P_prior = P_prior,
    method = "L-BFGS-B",  # Constrained optimization
    lower = c(-Inf, -Inf, 0.001)  # Ensure sigma1 > 0
  )

  # Return the estimated parameters
  return(result$par)
}

### Simulation and Estimation ###
simulate_and_estimate <- function(a, b, sigma1, sigma2, n, num_realizations) {
  # Storage for estimated parameters
  estimates <- matrix(NA, nrow = num_realizations, ncol = 3)
  colnames(estimates) <- c("a", "b", "sigma1")

  # Simulate and estimate for each realization
  for (i in 1:num_realizations) {
    # Simulate data
    X <- numeric(n)
    Y <- numeric(n)
    X[1] <- 0  # Initial state
    Y[1] <- X[1] + rnorm(1, mean = 0, sd = sigma2)

    for (t in 2:n) {
      X[t] <- a * X[t - 1] + b + rnorm(1, mean = 0, sd = sigma1)
      Y[t] <- X[t] + rnorm(1, mean = 0, sd = sigma2)
    }

    # Cap extreme values in Y to prevent numerical instability
    Y <- pmin(pmax(Y, -1e3), 1e3)

    # Estimate parameters
    tryCatch({
      estimates[i, ] <- estimate_parameters(Y, R = sigma2^2)
    }, error = function(e) {
      estimates[i, ] <- c(NA, NA, NA)  # Assign NA for failed realizations
    })
  }

  return(estimates)
}

### Main Simulation ###
set.seed(123)  # For reproducibility
n <- 100
num_realizations <- 100
sigma2 <- 1  # Observation noise standard deviation

# Parameter combinations
params_1 <- simulate_and_estimate(a = 1, b = 0.9, sigma1 = 1, sigma2 = sigma2, n = n, num_realizations = num_realizations)
params_2 <- simulate_and_estimate(a = 5, b = 0.9, sigma1 = 1, sigma2 = sigma2, n = n, num_realizations = num_realizations)
params_3 <- simulate_and_estimate(a = 1, b = 0.9, sigma1 = 5, sigma2 = sigma2, n = n, num_realizations = num_realizations)

### Summarize Results ###
# Combine results into a single data frame for plotting
results <- bind_rows(
  data.frame(params_1, Combination = "a = 1, b = 0.9, sigma1 = 1"),
  data.frame(params_2, Combination = "a = 5, b = 0.9, sigma1 = 1"),
  data.frame(params_3, Combination = "a = 1, b = 0.9, sigma1 = 5")
)

# Remove rows with NA values
results <- na.omit(results)

# Reshape data for ggplot
results_long <- results %>%
  pivot_longer(cols = c("a", "b", "sigma1"), names_to = "Parameter", values_to = "Estimate")

# Create boxplots
ggplot(results_long, aes(x = Parameter, y = Estimate, fill = Combination)) +
  geom_boxplot(outlier.size = 1, alpha = 0.7) +
  facet_wrap(~ Combination, scales = "free") +
  labs(title = "Parameter Estimates Across Simulations",
       x = "Parameter",
       y = "Estimated Value") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10))

# Summarize results by combination and parameter
summary_stats <- results_long %>%
  group_by(Combination, Parameter) %>%
  summarize(
    Mean = mean(Estimate, na.rm = TRUE),
    Median = median(Estimate, na.rm = TRUE),
    SD = sd(Estimate, na.rm = TRUE),
    IQR = IQR(Estimate, na.rm = TRUE),
    .groups = "drop"
  )

# Print the summary statistics
print(summary_stats)

# 1.5

# Load required libraries
library(ggplot2)

# Set random seed for reproducibility
set.seed(123)

### Simulation Parameters ###
n <- 100  # Number of time steps
num_realizations <- 100  # Number of realizations
a <- 1  # State transition coefficient
b <- 0.9  # Bias term
sigma1 <- 1  # Scale for system noise
sigma2 <- 1  # Standard deviation of observation noise
nu_values <- c(100, 5, 2, 1)  # Degrees of freedom for t-distribution

### Simulate the Model ###
simulate_t_noise <- function(a, b, sigma1, sigma2, nu, n, num_realizations) {
  # Storage for simulated data
  simulations <- list()

  for (i in 1:num_realizations) {
    # Initialize latent state and observations
    X <- numeric(n)
    Y <- numeric(n)
    X[1] <- 0  # Initial state
    Y[1] <- X[1] + rnorm(1, mean = 0, sd = sigma2)

    for (t in 2:n) {
      # System noise from Student's t-distribution
      lambda_t <- rt(1, df = nu)  # Draw from t-distribution with df = nu
      X[t] <- a * X[t - 1] + b + sigma1 * lambda_t  # Latent state
      Y[t] <- X[t] + rnorm(1, mean = 0, sd = sigma2)  # Observation
    }

    # Store the results
    simulations[[i]] <- list(X = X, Y = Y)
  }

  return(simulations)
}

# Simulate the model for each value of nu
simulations_by_nu <- lapply(nu_values, function(nu) {
  simulate_t_noise(a, b, sigma1, sigma2, nu, n, num_realizations)
})

### Plot the Density of t-Distributions ###
# Generate data for the density plots
t_density_data <- data.frame(
  x = seq(-4, 4, length.out = 1000),
  Normal = dnorm(seq(-4, 4, length.out = 1000))  # Standard normal density
)

# Add densities for each t-distribution
for (nu in nu_values) {
  t_density_data[[paste0("t_df_", nu)]] <- dt(t_density_data$x, df = nu)
}

# Reshape the data for ggplot
t_density_long <- t_density_data %>%
  pivot_longer(cols = -x, names_to = "Distribution", values_to = "Density")

# Plot the densities
ggplot(t_density_long, aes(x = x, y = Density, color = Distribution)) +
  geom_line(size = 1) +
  scale_color_manual(
    values = c(
      "Normal" = "black",  # Normal distribution in black
      "t_df_100" = "green",  # t-distribution with nu = 100 in green
      "t_df_5" = "red",      # t-distribution with nu = 5 in red
      "t_df_2" = "blue",     # t-distribution with nu = 2 in blue
      "t_df_1" = "yellow"    # t-distribution with nu = 1 in yellow
    )
  ) +
  labs(title = "Density of t-Distributions vs Standard Normal",
       x = "x",
       y = "Density",
       color = "Distribution") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10))

# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)

# Set random seed for reproducibility
set.seed(123)

### Simulation Parameters ###
n <- 100  # Number of time steps
num_realizations <- 100  # Number of realizations
a <- 1  # State transition coefficient
b <- 0.9  # Bias term
sigma1 <- 1  # Scale for system noise
sigma2 <- 1  # Standard deviation of observation noise
nu_values <- c(100, 5, 2, 1)  # Degrees of freedom for t-distribution

### Kalman Filter Function ###
myKalmanFilter <- function(y, theta, R, x_prior = 0, P_prior = 10) {
  # Unpack parameters
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]

  # Number of observations
  N <- length(y)

  # Initialize vectors to store results
  x_pred  <- numeric(N)  # Predicted means
  P_pred  <- numeric(N)  # Predicted variances
  x_filt  <- numeric(N)  # Filtered means
  P_filt  <- numeric(N)  # Filtered variances
  innovation     <- numeric(N)  # Pre-fit residuals: y[t] - x_pred[t]
  innovation_var <- numeric(N)  # Innovation covariance: P_pred[t] + R

  # Kalman filter loop
  for (t in seq_len(N)) {
    # Prediction step
    if (t == 1) {
      # Use the prior for the first step
      x_pred[t] <- x_prior
      P_pred[t] <- P_prior + sigma1^2
    } else {
      # Use the previous filtered estimate for subsequent steps
      x_pred[t] <- a * x_filt[t - 1] + b
      P_pred[t] <- a^2 * P_filt[t - 1] + sigma1^2
    }

    # Ensure predicted variance is positive
    P_pred[t] <- max(P_pred[t], 1e-6)

    # Update step
    innovation[t] <- y[t] - x_pred[t]  # Prediction error
    innovation_var[t] <- P_pred[t] + R  # Prediction error variance

    # Ensure innovation variance is positive
    innovation_var[t] <- max(innovation_var[t], 1e-6)

    K_t <- P_pred[t] / innovation_var[t]  # Kalman gain
    x_filt[t] <- x_pred[t] + K_t * innovation[t]  # Filtered estimate
    P_filt[t] <- (1 - K_t) * P_pred[t]  # Filtered estimate variance
  }

  # Return results as a list
  return(list(
    x_pred = x_pred,
    P_pred = P_pred,
    x_filt = x_filt,
    P_filt = P_filt,
    innovation = innovation,
    innovation_var = innovation_var
  ))
}

### Log-Likelihood Function ###
myLogLikFun <- function(theta, y, R, x_prior = 0, P_prior = 10) {
  # Unpack parameters
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]

  # Ensure sigma1 is positive
  if (sigma1 <= 0) {
    return(Inf)  # Return a large value for invalid parameters
  }

  # Call the Kalman filter function
  kf_result <- myKalmanFilter(y, theta, R, x_prior, P_prior)

  # Extract innovations and their variances
  err <- kf_result$innovation       # Innovations (y[t] - x_pred[t])
  S <- kf_result$innovation_var     # Innovation variances (P_pred[t] + R)

  # Ensure all innovation variances are positive
  if (any(S <= 0)) {
    return(Inf)  # Return a large value for invalid computations
  }

  # Compute log-likelihood contributions from each time step
  logL <- -0.5 * sum(log(2 * pi * S) + (err^2 / S))

  return(-logL)
}

### Parameter Estimation Function ###
estimate_parameters <- function(y, R, x_prior = 0, P_prior = 10) {
  # Initial guesses for parameters
  theta_init <- c(1, 1, 1)  # Initial guesses for a, b, sigma1

  # Minimize the negative log-likelihood
  result <- optim(
    par = theta_init,
    fn = myLogLikFun,
    y = y,
    R = R,
    x_prior = x_prior,
    P_prior = P_prior,
    method = "L-BFGS-B",  # Constrained optimization
    lower = c(-Inf, -Inf, 0.001)  # Ensure sigma1 > 0
  )

  # Return the estimated parameters
  return(result$par)
}

### Simulate the Model ###
simulate_t_noise <- function(a, b, sigma1, sigma2, nu, n, num_realizations) {
  # Storage for simulated data
  simulations <- list()

  for (i in 1:num_realizations) {
    # Initialize latent state and observations
    X <- numeric(n)
    Y <- numeric(n)
    X[1] <- 0  # Initial state
    Y[1] <- X[1] + rnorm(1, mean = 0, sd = sigma2)

    for (t in 2:n) {
      # System noise from Student's t-distribution
      lambda_t <- rt(1, df = nu)  # Draw from t-distribution with df = nu
      X[t] <- a * X[t - 1] + b + sigma1 * lambda_t  # Latent state
      Y[t] <- X[t] + rnorm(1, mean = 0, sd = sigma2)  # Observation
    }

    # Store the results
    simulations[[i]] <- Y
  }

  return(simulations)
}

### Apply Estimation Pipeline ###
apply_pipeline <- function(simulations, sigma2) {
  # Storage for estimated parameters
  estimates <- matrix(NA, nrow = length(simulations), ncol = 3)
  colnames(estimates) <- c("a", "b", "sigma1")

  for (i in seq_along(simulations)) {
    tryCatch({
      estimates[i, ] <- estimate_parameters(simulations[[i]], R = sigma2^2)
    }, error = function(e) {
      estimates[i, ] <- c(NA, NA, NA)  # Assign NA for failed realizations
    })
  }

  return(estimates)
}

### Main Simulation and Estimation ###
set.seed(123)  # For reproducibility

# Simulate and estimate for each value of nu
results_by_nu <- lapply(nu_values, function(nu) {
  simulations <- simulate_t_noise(a, b, sigma1, sigma2, nu, n, num_realizations)
  estimates <- apply_pipeline(simulations, sigma2)
  data.frame(estimates, Combination = paste0("t (nu = ", nu, ")"))
})

# Combine results into a single data frame
results <- bind_rows(results_by_nu)

# Reshape data for ggplot
results_long <- results %>%
  pivot_longer(cols = c("a", "b", "sigma1"), names_to = "Parameter", values_to = "Estimate")

### Summarize Results with Boxplots ###
ggplot(results_long, aes(x = Parameter, y = Estimate, fill = Combination)) +
  geom_boxplot(outlier.size = 1, alpha = 0.7) +
  facet_wrap(~ Parameter, scales = "free") +
  labs(title = "Estimated Parameters (t-Distribution)",
       x = "Parameter",
       y = "Estimated Value",
       fill = "Noise Type") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10))