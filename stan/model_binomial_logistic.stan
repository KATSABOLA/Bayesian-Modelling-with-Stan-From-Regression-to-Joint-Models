// model_binomial_logistic.stan
// Matsuura (2022), Chapter 5, Section 5.3 — Binomial Logistic Regression
//
// Medical application: N = 50 HIV patients followed for 6 months
//   Sex       = patient sex (0 = female, 1 = male)
//   cd4_init  = CD4 count at ART initiation (cells/100 uL), pass as cd4_init/100
//   n_visits  = number of scheduled clinic appointments
//   n_supp    = number of visits where VL was suppressed
//
// Model:  q[n]      = inv_logit(b[1] + b[2]*Sex[n] + b[3]*cd4_init[n])
//         n_supp[n] ~ Binomial(n_visits[n], q[n])
//
// q[n] is the per-visit viral suppression probability for patient n.

data {
  int N;
  vector<lower=0, upper=1>[N] Sex;       // 0 = female, 1 = male
  vector<lower=0>[N] cd4_init;           // CD4/100 at ART initiation
  array[N] int<lower=0> n_visits;        // number of clinic visits
  array[N] int<lower=0> n_supp;          // suppressed visits
}

parameters {
  vector[3] b;   // b[1]=intercept, b[2]=sex effect, b[3]=CD4 effect
}

transformed parameters {
  // Suppression probability per visit via logistic link
  vector[N] q = inv_logit(b[1] + b[2]*Sex[1:N] + b[3]*cd4_init[1:N]);
}

model {
  // Noninformative priors (Matsuura Ch. 5 default)
  // b implicitly ~ improper flat prior (no prior statement = uniform)

  // Binomial likelihood: n_supp successes in n_visits trials
  n_supp[1:N] ~ binomial(n_visits[1:N], q[1:N]);
}

generated quantities {
  // Posterior predictive replications
  array[N] int yp = binomial_rng(n_visits[1:N], q[1:N]);

  // Pointwise log-likelihood for LOO-CV
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = binomial_lpmf(n_supp[n] | n_visits[n], q[n]);
}
