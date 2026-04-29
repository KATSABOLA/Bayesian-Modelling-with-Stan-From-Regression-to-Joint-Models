// =============================================================================
// lme_m2_intercepts.stan  —  Session 1, Model 2
// Varying intercepts linear mixed model for log(bilirubin)
// (Sorensen et al. 2016, Model 2 adapted to PBC data)
//
// Model:  Y_ij = (beta0 + u_{0i}) + beta1 * t_ij + eps_ij
//         u_{0i} ~ Normal(0, sigma_u^2)   [random intercept per subject]
//         eps_ij  ~ Normal(0, sigma^2)
//
// Non-centred parameterisation:
//   u_raw_i ~ Normal(0,1),   u_{0i} = sigma_u * u_raw_i
// =============================================================================

data {
  int<lower=1> N;                               // total observations
  int<lower=1> n_subj;                          // number of subjects
  vector[N] y;                                  // log(bilirubin)
  vector[N] t;                                  // time in years
  array[N] int<lower=1, upper=n_subj> subj_id; // subject indicator
}

parameters {
  real beta0;                       // population intercept
  real beta1;                       // population slope
  real<lower=0> sigma_u;            // SD of random intercepts
  vector[n_subj] u_raw;             // non-centred random intercepts
  real<lower=0> sigma;              // residual SD
}

transformed parameters {
  vector[n_subj] u = sigma_u * u_raw;  // actual random intercepts

  vector[N] mu;
  for (n in 1:N)
    mu[n] = (beta0 + u[subj_id[n]]) + beta1 * t[n];
}

model {
  // Priors
  beta0   ~ normal(0, 5);
  beta1   ~ normal(0, 2);
  sigma_u ~ student_t(4, 0, 1);
  sigma   ~ student_t(4, 0, 1);
  u_raw   ~ std_normal();   // non-centred: u = sigma_u * u_raw

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n]   = normal_rng(mu[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
  real sigma_u_sq = square(sigma_u);  // variance of random intercepts
}
