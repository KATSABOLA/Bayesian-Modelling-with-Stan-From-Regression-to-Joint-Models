// model_poisson.stan
// Matsuura (2022), Chapter 5, Section 5.5 — Poisson Regression (log link)
//
// Medical application: N = 50 HIV patients followed for 6 months
//   Sex       = patient sex (0 = female, 1 = male)
//   cd4_init  = CD4 count at ART initiation (cells/100 uL), pass as cd4_init/100
//   n_visits  = number of clinic visits attended (count outcome)
//
// Model:  log_lam[n] = b[1] + b[2]*Sex[n] + b[3]*cd4_init[n]
//         n_visits[n] ~ Poisson(exp(log_lam[n]))
//
// Uses poisson_log for numerical stability (avoids explicit exp).

data {
  int N;
  vector<lower=0, upper=1>[N] Sex;    // 0 = female, 1 = male
  vector<lower=0>[N] cd4_init;        // CD4/100 at ART initiation
  array[N] int<lower=0> n_visits;     // clinic visit count (outcome)
}

parameters {
  vector[3] b;   // b[1]=intercept, b[2]=sex effect, b[3]=CD4 effect
}

transformed parameters {
  // Log of expected visit count (log link)
  vector[N] log_lam = b[1] + b[2]*Sex[1:N] + b[3]*cd4_init[1:N];
}

model {
  // Noninformative priors (Matsuura Ch. 5 default)
  n_visits[1:N] ~ poisson_log(log_lam[1:N]);
}

generated quantities {
  // Posterior predictive replications
  array[N] int mp = poisson_log_rng(log_lam[1:N]);

  // Pointwise log-likelihood for LOO-CV
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = poisson_log_lpmf(n_visits[n] | log_lam[n]);
}
