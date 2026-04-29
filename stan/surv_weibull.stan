// =============================================================================
// surv_weibull.stan  —  Session 2, Model 2
// Bayesian Weibull proportional hazards survival model for PBC
//
// Model:  h(t | x_i) = h_0(t) * exp(x_i' gamma)
//
// Parameterisation: Weibull(alpha, scale_i)
//   scale_i = exp(mu + x_i' gamma)     (subject-specific scale)
//   h_0(t)  = alpha / exp(mu) * (t / exp(mu))^(alpha-1)
//   h(t|x)  = alpha * t^(alpha-1) * exp(-(alpha*mu + x'gamma))
//   H(t|x)  = (t / scale_i)^alpha
//
// Log-likelihood uses Stan built-in weibull_lpdf / weibull_lccdf for
// numerical stability — avoids manual pow() overflow in plain formulation.
//
// Special cases:
//   alpha = 1  Exponential (constant hazard)
//   alpha > 1  Increasing hazard over time
//   alpha < 1  Decreasing hazard over time
// =============================================================================

data {
  int<lower=1> N;
  vector<lower=0>[N] t_event;
  array[N] int<lower=0, upper=1> event;
  int<lower=1> P;
  matrix[N, P] X;
}

parameters {
  real<lower=0> alpha;   // shape parameter (alpha=1 -> exponential)
  real mu;               // log baseline scale: median ~ exp(mu) at x=0
  vector[P] gamma;       // log-hazard ratios
}

transformed parameters {
  // Subject-specific Weibull scale: exp(mu + x'gamma)
  vector[N] scale = exp(mu + X * gamma);
}

model {
  // Priors
  alpha  ~ gamma(2, 1);       // E[alpha]=2; increasing hazard a priori for PBC
  mu     ~ normal(2, 2);      // log scale ~7 yr at baseline
  gamma  ~ normal(0, 0.5);    // narrower prior reduces pathological geometry

  // Log-likelihood using numerically stable built-ins
  for (i in 1:N) {
    if (event[i] == 1)
      target += weibull_lpdf(t_event[i] | alpha, scale[i]);
    else
      target += weibull_lccdf(t_event[i] | alpha, scale[i]);
  }
}

generated quantities {
  // Hazard ratios for covariates (same as for Cox PH)
  vector[P] HR = exp(gamma);

  // Median survival for reference patient (all x = 0)
  // Weibull median: t_{1/2} = scale * (log 2)^{1/alpha}
  real t_median_ref = exp(mu) * pow(log(2.0), 1.0 / alpha);

  // Log-likelihood per subject (for LOO-CV)
  vector[N] log_lik;
  // Cox-Snell residuals: H(t|x) = (t/scale)^alpha ~ Exp(1) under true model
  vector[N] cox_snell;
  for (i in 1:N) {
    if (event[i] == 1)
      log_lik[i] = weibull_lpdf(t_event[i] | alpha, scale[i]);
    else
      log_lik[i] = weibull_lccdf(t_event[i] | alpha, scale[i]);
    cox_snell[i] = pow(t_event[i] / scale[i], alpha);
  }
}
