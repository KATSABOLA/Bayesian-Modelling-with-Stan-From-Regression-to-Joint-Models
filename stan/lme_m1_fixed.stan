// =============================================================================
// lme_m1_fixed.stan  —  Session 1, Model 1
// Fixed-effects-only linear regression for log(bilirubin) over time
// (Sorensen et al. 2016, Model 1 adapted to PBC data)
//
// Model:  Y_ij = beta0 + beta1 * t_ij + eps_ij
//         eps_ij ~ Normal(0, sigma^2)
//
// This model IGNORES within-subject correlation — used as baseline
// to motivate the need for random effects in Models 2 and 3.
// =============================================================================

data {
  int<lower=1> N;         // total number of observations
  vector[N] y;            // log(serum bilirubin)
  vector[N] t;            // time in years
}

parameters {
  real beta0;             // population intercept
  real beta1;             // slope (change in log-bili per year)
  real<lower=0> sigma;    // residual SD
}

model {
  // Priors
  beta0 ~ normal(0, 5);
  beta1 ~ normal(0, 2);
  sigma ~ student_t(4, 0, 1);

  // Likelihood
  y ~ normal(beta0 + beta1 * t, sigma);
}

generated quantities {
  vector[N] mu = beta0 + beta1 * t;
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n]   = normal_rng(mu[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}
