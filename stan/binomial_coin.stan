data {
  int<lower=0> N;              // number of patients tested
  int<lower=0> n_suppressed;   // number with viral suppression
}

parameters {
  real<lower=0, upper=1> theta;   // probability of viral suppression
}

model {
  theta       ~ beta(2, 2);                   // weakly informative prior
  n_suppressed ~ binomial(N, theta);          // likelihood
}

generated quantities {
  int y_rep = binomial_rng(N, theta);    // posterior predictive replicate
}
