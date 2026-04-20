//
// Medical application: V visit-level observations across HIV patients
//   Sex         = patient sex (0 = female, 1 = male)
//   cd4_init    = CD4 count at ART initiation (cells/100 uL), pass as cd4_init/100
//   counselling = whether patient received adherence counselling at that visit
//                 (0 = no, 1 = yes)
//   suppressed  = viral suppression at that visit (1 = VL < 1000, 0 = not)
//
// Model:  q[v]         = inv_logit(b[1] + b[2]*Sex[v] + b[3]*cd4_init[v]
//                                       + b[4]*counselling[v])
//         suppressed[v] ~ Bernoulli(q[v])

data {
  int V;
  vector<lower=0, upper=1>[V] Sex;          // 0 = female, 1 = male
  vector<lower=0>[V] cd4_init;              // CD4/100 at ART initiation
  vector<lower=0, upper=1>[V] counselling;  // adherence counselling at visit (0/1)
  array[V] int<lower=0, upper=1> suppressed; // viral suppression per visit (0/1)
}

parameters {
  vector[4] b;   // b[1]=intercept, b[2]=sex, b[3]=CD4, b[4]=counselling
}

transformed parameters {
  // Per-visit viral suppression probability via logistic link
  vector[V] q = inv_logit(
    b[1] + b[2]*Sex[1:V] + b[3]*cd4_init[1:V] + b[4]*counselling[1:V]);
}

model {
  // Noninformative priors (Matsuura Ch. 5 default)
  suppressed[1:V] ~ bernoulli(q[1:V]);
}

generated quantities {
  // Posterior predictive replications
  array[V] int yp = bernoulli_rng(q[1:V]);

  // Pointwise log-likelihood for LOO-CV
  vector[V] log_lik;
  for (v in 1:V)
    log_lik[v] = bernoulli_lpmf(suppressed[v] | q[v]);
}
