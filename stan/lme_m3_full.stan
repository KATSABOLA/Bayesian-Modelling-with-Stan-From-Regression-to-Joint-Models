// =============================================================================
// lme_m3_full.stan  —  Session 1, Model 3  (PRIMARY longitudinal model)
// Varying intercepts AND slopes with correlation (Sorensen et al. 2016)
// Adapted to PBC log(bilirubin) data
//
// Model:
//   Y_ij = (beta0 + u_{0i}) + (beta1 + u_{1i}) * t_ij + eps_ij
//   [u_{0i}, u_{1i}]' ~ MVN(0, Sigma_u)
//   Sigma_u = diag(sigma_u) * Omega_u * diag(sigma_u)
//   eps_ij  ~ Normal(0, sigma^2)
//
// Cholesky parameterisation (Sorensen notation):
//   L_u  ~ lkj_corr_cholesky(2)              [Cholesky of correlation matrix]
//   z_u  ~ Normal(0, I)                      [standard normal raw effects]
//   u    = diag_pre_multiply(sigma_u, L_u) * z_u   [actual random effects]
//
// This is the model used throughout the joint model sessions.
// =============================================================================

data {
  int<lower=1> N;                               // total observations
  int<lower=1> n_subj;                          // number of subjects
  vector[N] y;                                  // log(bilirubin)
  vector[N] t;                                  // time in years
  array[N] int<lower=1, upper=n_subj> subj_id; // subject indicator
}

parameters {
  // ── Fixed effects ──────────────────────────────────────────────────────────
  real beta0;   // population intercept (mean log-bili at t=0)
  real beta1;   // population slope    (mean change per year)

  // ── Random effects — Sorensen / Cholesky parameterisation ─────────────────
  vector<lower=0>[2] sigma_u;          // [sigma_{u0}, sigma_{u1}]
  cholesky_factor_corr[2] L_u;         // Cholesky factor of correlation matrix
  matrix[2, n_subj] z_u;              // standard normal raw effects (non-centred)

  // ── Residual ───────────────────────────────────────────────────────────────
  real<lower=0> sigma;
}

transformed parameters {
  // u[1,i] = random intercept for subject i  (u_{0i})
  // u[2,i] = random slope    for subject i  (u_{1i})
  // This follows Sorensen et al. exactly:
  //   u = diag_pre_multiply(sigma_u, L_u) * z_u
  matrix[2, n_subj] u = diag_pre_multiply(sigma_u, L_u) * z_u;

  // Linear predictor
  vector[N] mu;
  for (n in 1:N)
    mu[n] = (beta0 + u[1, subj_id[n]])
          + (beta1 + u[2, subj_id[n]]) * t[n];
}

model {
  // ── Priors — fixed effects ─────────────────────────────────────────────────
  beta0 ~ normal(0, 5);
  beta1 ~ normal(0, 2);

  // ── Priors — random effect variance and correlation ────────────────────────
  sigma_u ~ student_t(4, 0, 1);     // half-t on SDs (positive by constraint)
  L_u     ~ lkj_corr_cholesky(2);   // LKJ(eta=2): weakly regularises toward 0

  // ── Non-centred random effects ─────────────────────────────────────────────
  to_vector(z_u) ~ std_normal();

  // ── Residual SD ────────────────────────────────────────────────────────────
  sigma ~ student_t(4, 0, 1);

  // ── Likelihood ────────────────────────────────────────────────────────────
  y ~ normal(mu, sigma);
}

generated quantities {
  // ── Posterior predictive replication ──────────────────────────────────────
  vector[N] y_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n]   = normal_rng(mu[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }

  // ── Random effects covariance and correlation matrices ─────────────────────
  // Sigma_u = diag(sigma_u) * Omega_u * diag(sigma_u)
  corr_matrix[2] Omega_u = multiply_lower_tri_self_transpose(L_u);
  cov_matrix[2]  Sigma_u;
  Sigma_u[1,1] = square(sigma_u[1]);
  Sigma_u[2,2] = square(sigma_u[2]);
  Sigma_u[1,2] = sigma_u[1] * sigma_u[2] * Omega_u[1,2];
  Sigma_u[2,1] = Sigma_u[1,2];

  // ── Intercept-slope correlation (scalar, for easy reporting) ──────────────
  real rho_u = Omega_u[1,2];
}
