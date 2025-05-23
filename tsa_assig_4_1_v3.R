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
plot(1:n, trajectories[, 1], type = "l", col = "blue", lwd = 2, ylim = range(trajectories), # nolint: line_length_linter.
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
  theta,         # Model parameters for X_{t+1} = a*X_t + b + noise
  R,             # Measurement noise variance
  x_prior = 0,   # Initial prior mean for X_0
  P_prior = 10   # Initial prior variance for X_0
) {
  # Unpack model parameters
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]

  N <- length(y)

  x_pred  <- numeric(N)
  P_pred  <- numeric(N)
  x_filt  <- numeric(N)
  P_filt  <- numeric(N)
  innovation     <- numeric(N)
  innovation_var <- numeric(N)

  for (t in seq_len(N)) {
    if (t == 1) {
      x_pred[t] <- x_prior
      P_pred[t] <- P_prior + sigma1^2
    } else {
      x_pred[t] <- a * x_filt[t - 1] + b
      P_pred[t] <- a^2 * P_filt[t - 1] + sigma1^2
    }

    innovation[t] <- y[t] - x_pred[t]

    # Important: enforce a minimum positive variance to avoid zero or negative values
    innovation_var[t] <- P_pred[t] + R
    if (innovation_var[t] < 1e-6) {
      innovation_var[t] <- 1e-6
    }

    K_t <- P_pred[t] / innovation_var[t]
    x_filt[t] <- x_pred[t] + K_t * innovation[t]
    P_filt[t] <- (1 - K_t) * P_pred[t]

    # Safety checks - ensure no NA or NaN sneaks in
    if (anyNA(c(x_pred[t], P_pred[t], x_filt[t], P_filt[t], innovation[t], innovation_var[t]))) {
      stop("NA detected in Kalman filter calculations at time ", t)
    }
  }

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

install.packages("reshape2")

library(ggplot2)
library(dplyr)
library(tidyr)

# ---- 1. Kalman Filter implementation ----
myKalmanFilter <- function(y, theta, R, X0, P0) {
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]

  n <- length(y)

  # Initialize
  x_pred <- numeric(n)
  P_pred <- numeric(n)
  x_filt <- numeric(n)
  P_filt <- numeric(n)
  v <- numeric(n)
  S <- numeric(n)

  # Initial state
  x_filt[1] <- X0
  P_filt[1] <- P0

  for (t in 1:n) {
    # Predict step (for t=1, prediction uses previous filtered estimate)
    if (t == 1) {
      x_pred[t] <- a * X0 + b
      P_pred[t] <- a^2 * P0 + sigma1^2
    } else {
      x_pred[t] <- a * x_filt[t-1] + b
      P_pred[t] <- a^2 * P_filt[t-1] + sigma1^2
    }

    # Innovation
    v[t] <- y[t] - x_pred[t]
    S[t] <- P_pred[t] + R
    if (S[t] < 1e-8) S[t] <- 1e-8

    # Kalman gain
    K <- P_pred[t] / S[t]

    # Update step
    x_filt[t] <- x_pred[t] + K * v[t]
    P_filt[t] <- (1 - K) * P_pred[t]
    if (P_filt[t] < 1e-8) P_filt[t] <- 1e-8
  }

  return(list(
    x_pred = x_pred,
    P_pred = P_pred,
    x_filt = x_filt,
    P_filt = P_filt,
    v = v,
    S = S
  ))
}

# ---- 2. Negative log-likelihood function ----
myLogLikFun <- function(theta, y, R, X0, P0) {
  a <- theta[1]
  b <- theta[2]
  sigma1 <- theta[3]

  # Basic parameter checks to avoid invalid values
  if (sigma1 <= 1e-6 || sigma1 > 1e2) return(1e10)

  # Run Kalman filter
  kf <- tryCatch({
    myKalmanFilter(y, theta, R, X0, P0)
  }, error = function(e) NULL)

  if (is.null(kf)) return(1e10)

  v <- kf$v
  S <- kf$S

  if (any(!is.finite(v)) || any(!is.finite(S)) || any(S <= 0)) return(1e10)

  # Compute log-likelihood (sum of log of Normal pdf of innovations)
  logLikVec <- -0.5 * (log(2 * pi) + log(S) + (v^2) / S)

  totalLogLik <- sum(logLikVec)

  # Penalize extremely large negative likelihood
  if (totalLogLik < -1e10) return(1e10)

  return(-totalLogLik)  # negative log-likelihood for minimization
}

# ---- 3. Simulation function ----
simulate_process <- function(a, b, sigma1, sigma2, X0, n) {
  X <- numeric(n)
  Y <- numeric(n)

  X[1] <- X0
  Y[1] <- X[1] + rnorm(1, 0, sigma2)

  for (t in 2:n) {
    X[t] <- a * X[t-1] + b + rnorm(1, 0, sigma1)
    Y[t] <- X[t] + rnorm(1, 0, sigma2)
  }

  list(X = X, Y = Y)
}

# ---- 4. Parameter estimation for multiple datasets ----
estimate_parameters <- function(a_true, b_true, sigma1_true, N = 100, n = 100, X0 = 5, sigma2 = 1) {
  results <- matrix(NA, nrow = N, ncol = 3)
  colnames(results) <- c("a", "b", "sigma1")

  for (i in 1:N) {
    sim <- simulate_process(a_true, b_true, sigma1_true, sigma2, X0, n)
    Y <- sim$Y

    res <- optim(par = c(0.5, 0.5, 1),
                 fn = myLogLikFun,
                 y = Y,
                 R = sigma2^2,
                 X0 = X0,
                 P0 = 1e6,
                 method = "L-BFGS-B",
                 lower = c(-Inf, -Inf, 1e-6),
                 upper = c(Inf, Inf, Inf),
                 control = list(maxit = 1000))

    results[i, ] <- res$par
  }

  df <- as.data.frame(results)
  df$Combination <- paste0("a=", a_true, ", b=", b_true, ", σ1=", sigma1_true)
  return(df)
}

# ---- 5. Run estimation for three parameter combinations ----
set.seed(123)
X0 <- 5
sigma2 <- 1
n <- 100
N <- 100

est_case1 <- estimate_parameters(1, 0.9, 1, N, n, X0, sigma2)
est_case2 <- estimate_parameters(5, 0.9, 1, N, n, X0, sigma2)
est_case3 <- estimate_parameters(1, 0.9, 5, N, n, X0, sigma2)

# Combine results
all_estimates <- bind_rows(est_case1, est_case2, est_case3)

# Reshape for plotting
results_long <- all_estimates %>%
  pivot_longer(cols = c("a", "b", "sigma1"), names_to = "Parameter", values_to = "Estimate")

# Plotting
ggplot(results_long, aes(x = Parameter, y = Estimate, fill = Combination)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  facet_wrap(~Combination, scales = "free") +
  labs(title = "Parameter Estimates Across Simulations",
       x = "Parameter",
       y = "Estimated Value") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10))

library(ggplot2)
library(dplyr)
library(tidyr)

# Plot densities of t-distributions and Normal(0,1)
plot_densities <- function() {
  x <- seq(-6, 6, length.out = 1000)
  df <- data.frame(
    x = rep(x, 5),
    dist = factor(rep(c("Normal(0,1)", "t(100)", "t(5)", "t(2)", "t(1)"), each = length(x)))
  )

  df$y <- with(df, ifelse(
    dist == "Normal(0,1)",
    dnorm(x),
    dt(x, df = as.numeric(gsub("t\\((\\d+)\\)", "\\1", dist)))
  ))

  ggplot(df, aes(x = x, y = y, color = dist)) +
    geom_line(size = 1) +
    labs(title = "Density of Normal(0,1) and Student's t-distributions",
         x = "x", y = "Density", color = "Distribution") +
    theme_minimal()
}

# Simulate process with t-distributed system noise
simulate_process_t <- function(a, b, sigma1, nu, sigma2, X0, n) {
  X <- numeric(n)
  Y <- numeric(n)

  X[1] <- X0
  Y[1] <- X[1] + rnorm(1, 0, sigma2)

  for (t in 2:n) {
    # System noise scaled t-dist: multiply by sigma1
    system_noise <- sigma1 * rt(1, df = nu)
    X[t] <- a * X[t-1] + b + system_noise
    Y[t] <- X[t] + rnorm(1, 0, sigma2)
  }

  list(X = X, Y = Y)
}

# Estimate parameters for multiple datasets with different nu's
estimate_parameters_t <- function(a_true, b_true, sigma1_true, nu, N = 100, n = 100, X0 = 5, sigma2 = 1) {
  results <- matrix(NA, nrow = N, ncol = 3)
  colnames(results) <- c("a", "b", "sigma1")

  for (i in 1:N) {
    sim <- simulate_process_t(a_true, b_true, sigma1_true, nu, sigma2, X0, n)
    Y <- sim$Y

    res <- optim(par = c(0.5, 0.5, 1),
                 fn = myLogLikFun,
                 y = Y,
                 R = sigma2^2,
                 X0 = X0,
                 P0 = 1e6,
                 method = "L-BFGS-B",
                 lower = c(-Inf, -Inf, 1e-6),
                 upper = c(Inf, Inf, Inf),
                 control = list(maxit = 1000))

    results[i, ] <- res$par
  }

  df <- as.data.frame(results)
  df$nu <- nu
  return(df)
}

# Parameters and constants
a_true <- 1
b_true <- 0.9
sigma1_true <- 1
sigma2 <- 1
X0 <- 5
n <- 100
N <- 100
nus <- c(100, 5, 2, 1)  # Degrees of freedom

# Run density plot
plot_densities()

# Run estimation for all nu values
all_estimates_t <- lapply(nus, function(nu) {
  cat("Estimating for nu =", nu, "\n")
  estimate_parameters_t(a_true, b_true, sigma1_true, nu, N, n, X0, sigma2)
}) %>% bind_rows()

# Add factor for plotting
all_estimates_t$nu <- factor(all_estimates_t$nu, levels = nus)

# Plot parameter estimates by nu
results_long_t <- all_estimates_t %>%
  pivot_longer(cols = c("a", "b", "sigma1"), names_to = "Parameter", values_to = "Estimate")

ggplot(results_long_t, aes(x = nu, y = Estimate, fill = nu)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~Parameter, scales = "free_y") +
  labs(title = "Parameter Estimates under t-distributed system noise",
       x = "Degrees of freedom (ν)",
       y = "Estimate") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10))
