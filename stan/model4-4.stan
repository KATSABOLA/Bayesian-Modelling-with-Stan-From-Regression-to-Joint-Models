// model4-4.stan
// Matsuura (2022), Bayesian Statistical Modeling with Stan, R, and Python
// Chapter 4: Simple linear regression
//
// Medical application: N = 15 HIV patients on long-term ART
//   years_art  = years on antiretroviral therapy
//   cd4_per100 = CD4 count (cells / 100 µL)
//
// Research question: does CD4 count increase with longer ART duration?

data {
  int N;
  vector[N] years_art;    // years on ART
  vector[N] cd4_per100;   // CD4 count (cells / 100 µL)
}

parameters {
  real a;               // intercept (CD4 at ART initiation)
  real b;               // slope (CD4 change per year on ART)
  real<lower=0> sigma;  // residual standard deviation
}

model {
  // Weakly informative priors (Matsuura 2022, Ch. 4)
  a     ~ normal(0, 100);
  b     ~ normal(0, 100);
  sigma ~ uniform(0, 100);

  // Vectorised likelihood
  cd4_per100[1:N] ~ normal(a + b * years_art[1:N], sigma);
}

generated quantities {
  vector[N] mu = a + b * years_art;   // fitted values
  vector[N] y_rep;                    // posterior predictive replicates
  vector[N] log_lik;                  // pointwise log-likelihood (for LOO-CV)
  for (n in 1:N) {
    y_rep[n]   = normal_rng(mu[n], sigma);
    log_lik[n] = normal_lpdf(cd4_per100[n] | mu[n], sigma);
  }
}
