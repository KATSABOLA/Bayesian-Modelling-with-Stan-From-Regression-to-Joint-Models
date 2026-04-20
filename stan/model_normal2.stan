// model_normal2.stan
/
// Medical application: N HIV patients; CD4 count (cells/100 uL) at 6 months post-ART
//   cd4_6m[n] = CD4 count / 100 for patient n
//
// Model:  cd4_6m[n] ~ Normal(mu, sigma),  n = 1, ..., N
//   mu    = mean CD4 recovery (cells/100 uL)   -- unknown
//   sigma = variability (SD) in CD4 recovery   -- unknown, must be > 0
//
// Priors (noninformative, Matsuura Ch. 3 default):
//   mu    ~ Normal(0, 100)   -- flat over plausible CD4 range
//   sigma ~ Normal(0, 100)   -- half-normal (constrained lower=0)

data {
  int<lower=1> N;         // number of patients
  vector[N] cd4_6m;       // CD4/100 at 6 months post-ART
}

parameters {
  real mu;                     // mean CD4 recovery
  real<lower=0> sigma;         // SD of CD4 recovery (must be positive)
}

model {
  // Priors (noninformative)
  mu    ~ normal(0, 100);
  sigma ~ normal(0, 100);

  // Likelihood
  cd4_6m ~ normal(mu, sigma);
}

generated quantities {
  // Posterior predictive replications
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    y_rep[n]   = normal_rng(mu, sigma);
    log_lik[n] = normal_lpdf(cd4_6m[n] | mu, sigma);
  }
}
