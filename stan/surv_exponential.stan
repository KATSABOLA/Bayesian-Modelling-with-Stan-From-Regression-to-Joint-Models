// =============================================================================
// surv_exponential.stan  —  Session 2, Model 1
// Bayesian Exponential (constant hazard) survival model for PBC
//
// Model:  h(t | x_i) = lambda * exp(x_i' gamma)
//         H(t | x_i) = lambda * t * exp(x_i' gamma)   [analytically tractable]
//
// Log-likelihood:
//   Event   (delta=1): log f(t) = log(lambda) + x'gamma - lambda*t*exp(x'gamma)
//   Censored (delta=0): log S(t) = -lambda * t * exp(x'gamma)
//
// Note: Exponential is a special case of Weibull with shape alpha = 1.
//       If the posterior of alpha in surv_weibull.stan is near 1, this model suffices.
// =============================================================================

data {
  int<lower=1> N;
  vector<lower=0>[N] t_event;
  array[N] int<lower=0, upper=1> event;
  int<lower=1> P;
  matrix[N, P] X;
}

parameters {
  real<lower=0> lambda;   // baseline hazard rate (constant)
  vector[P] gamma;        // log-hazard ratios
}

transformed parameters {
  vector[N] log_lambda_x = log(lambda) + X * gamma;
  // H(t|x) = lambda * t * exp(x'gamma) = t * exp(log_lambda_x)
}

model {
  lambda ~ gamma(1, 1);        // E[lambda] = 1; allows small and moderate rates
  gamma  ~ normal(0, 1);

  for (i in 1:N) {
    real H_i = t_event[i] * exp(log_lambda_x[i]);
    if (event[i] == 1)
      target += log_lambda_x[i] - H_i;   // log f(t) = log h(t) + log S(t)
    else
      target += -H_i;                     // log S(t)
  }
}

generated quantities {
  vector[P] HR = exp(gamma);

  // Median survival time for reference subject (x=0): t_med = log(2)/lambda
  real t_median_ref = log(2.0) / lambda;

  // Log-likelihood per subject (for LOO-CV)
  vector[N] log_lik;
  for (i in 1:N) {
    real H_i = t_event[i] * exp(log_lambda_x[i]);
    if (event[i] == 1)
      log_lik[i] = log_lambda_x[i] - H_i;
    else
      log_lik[i] = -H_i;
  }

  // Cox-Snell residuals: r_i = H(t_i | x_i) ~ Exp(1) if model fits
  vector[N] cox_snell;
  for (i in 1:N)
    cox_snell[i] = t_event[i] * exp(log_lambda_x[i]);
}
