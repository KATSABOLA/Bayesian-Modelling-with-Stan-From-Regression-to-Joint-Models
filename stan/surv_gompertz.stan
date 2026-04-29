// =============================================================================
// surv_gompertz.stan  —  Session 2, Model 5
// Bayesian Gompertz proportional hazards survival model for PBC
//
// Model:  h(t | x_i) = lambda * exp(rho * t) * exp(x_i' gamma)
//   lambda > 0 : baseline initial hazard
//   rho         : growth rate of hazard (rho > 0 → increasing; rho < 0 → decreasing)
//   gamma       : log-hazard ratios for covariates
//
// Cumulative hazard (analytically tractable):
//   H(t|x) = (lambda / rho) * (exp(rho*t) - 1) * exp(x'gamma)   [rho != 0]
//   H(t|x) = lambda * t * exp(x'gamma)                           [rho = 0, exponential]
//
// Log-likelihood:
//   Event   (delta=1): log f(t) = log h(t|x) - H(t|x)
//                              = log(lambda) + rho*t + x'gamma - H(t|x)
//   Censored (delta=0): log S(t) = -H(t|x)
//
// Biological relevance: Gompertz describes mortality that increases exponentially
// with age — a good fit for many chronic disease datasets.
// =============================================================================

data {
  int<lower=1> N;
  vector<lower=0>[N] t_event;
  array[N] int<lower=0, upper=1> event;
  int<lower=1> P;
  matrix[N, P] X;
}

parameters {
  real<lower=0> lambda;   // initial hazard (must be positive)
  real rho;               // hazard growth rate (unconstrained: + or -)
  vector[P] gamma;        // log-hazard ratios
}

transformed parameters {
  vector[N] xgamma = X * gamma;

  // Cumulative hazard H(t|x) for each subject
  vector[N] H;
  for (i in 1:N) {
    if (abs(rho) > 1e-6)
      H[i] = (lambda / rho) * (exp(rho * t_event[i]) - 1.0) * exp(xgamma[i]);
    else
      // rho ≈ 0: L'Hôpital gives H → lambda * t * exp(x'gamma)  (exponential)
      H[i] = lambda * t_event[i] * exp(xgamma[i]);
  }
}

model {
  lambda ~ gamma(1, 1);        // initial hazard: E[lambda]=1
  rho    ~ normal(0, 1);       // growth rate: centred at 0, allows + and -
  gamma  ~ normal(0, 1);

  for (i in 1:N) {
    real log_h = log(lambda) + rho * t_event[i] + xgamma[i];
    if (event[i] == 1)
      target += log_h - H[i];   // log f(t)
    else
      target += -H[i];           // log S(t)
  }
}

generated quantities {
  vector[P] HR = exp(gamma);

  // Posterior probability of increasing hazard (rho > 0)
  int increasing_hazard = (rho > 0) ? 1 : 0;

  vector[N] log_lik;
  vector[N] cox_snell;
  for (i in 1:N) {
    real log_h = log(lambda) + rho * t_event[i] + xgamma[i];
    if (event[i] == 1)
      log_lik[i] = log_h - H[i];
    else
      log_lik[i] = -H[i];
    cox_snell[i] = H[i];   // Cox-Snell residual = cumulative hazard
  }
}
