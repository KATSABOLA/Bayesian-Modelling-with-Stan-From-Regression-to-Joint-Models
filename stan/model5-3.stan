// model5-3.stan
// N = 50 HIV patients; sex (0=female,1=male); cd4_init (CD4/100 at ART initiation);
// adherence_prop = ART adherence proportion (0-1)

data {
  int N;
  vector<lower=0, upper=1>[N] sex;          // 0 = female, 1 = male
  vector<lower=0>[N] cd4_init;              // CD4 at ART initiation / 100 (per 100 cells/uL)
  vector<lower=0, upper=1>[N] adherence_prop;  // ART adherence proportion (0-1)
}

parameters {
  vector[3] b;          // b[1]=intercept, b[2]=sex effect, b[3]=CD4 effect
  real<lower=0> sigma;  // residual standard deviation
}

transformed parameters {
  vector[N] mu = b[1] + b[2] * sex[1:N] + b[3] * cd4_init[1:N];
}

model {
  // Weakly informative priors (Matsuura 2022, Ch. 5)
  b     ~ normal(0, 100);
  sigma ~ uniform(0, 100);

  adherence_prop[1:N] ~ normal(mu[1:N], sigma);
}

generated quantities {
  array[N] real y_rep = normal_rng(mu[1:N], sigma);

  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = normal_lpdf(adherence_prop[n] | mu[n], sigma);
}









