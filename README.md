# Bayesian Modelling with Stan: From Regression to Joint Models

A 4-session hands-on short course in applied Bayesian modelling using Stan and cmdstanr.
Covers regression, longitudinal mixed-effects models, survival models, and joint models
with clinical applications to HIV/TB data. Practicals in Quarto (R).

**Institution:** Division of Epidemiology & Biostatistics, Stellenbosch University  
**Year:** 2026

---

## Course Overview

This short course introduces the full Bayesian modelling workflow using
[Stan](https://mc-stan.org) and the `cmdstanr` R interface. Starting from
simple regression models and building progressively to joint longitudinal-survival
models, each session follows the same 8-step Bayesian workflow:

> Choose priors → Prior predictive check → Fit (HMC/NUTS) → MCMC diagnostics →
> Posterior summaries → Posterior predictive check → LOO-CV → Interpret

All models are motivated by clinical HIV/TB data and framed around real biostatistical
questions: viral suppression, ART adherence, CD4 recovery, and retention in care.

---

## Session Structure

| Session | Topic | Models | Key concepts |
|---------|-------|--------|--------------|
| **0** | Introduction to Stan & cmdstanr | Binomial, Normal, Simple LR, Multiple LR, Binomial Logistic, Logistic, Poisson | Stan blocks, HMC/NUTS, R-hat, ESS, LOO-CV |
| **1** | Longitudinal mixed-effects models | LM, LME (random intercept), LME (random slope + intercept) | Random effects, Cholesky parameterisation, posterior trajectories |
| **2** | Survival models | Exponential, Weibull, Gompertz | Hazard functions, censoring, baseline hazard |
| **3** | Joint longitudinal-survival models | Joint Gompertz | Association structures, shared random effects, dynamic predictions |

---

## Repository Structure








---

## Prerequisites

**Software:**
- R ≥ 4.4 and RStudio (or VS Code)
- [cmdstanr](https://mc-stan.org/cmdstanr/) and CmdStan ≥ 2.34
- A C++ toolchain: RTools44+ on Windows
- Quarto ≥ 1.4 (to render practicals)

**R packages:**
```r
install.packages(c("cmdstanr", "posterior", "bayesplot", "loo",
                   "tidyverse", "ggplot2"),
  repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
cmdstanr::install_cmdstan()
```


## References

- Matsuura (2022) [^1]
- Carpenter et al. (2017) [^2]
- Vehtari et al. (2017) [^3]

[^1]: Matsuura, K. (2022). *Bayesian Statistical Modeling with Stan, R, and Python*. Springer.
[^2]: Carpenter, B., et al. (2017). Stan: A probabilistic programming language. *Journal of Statistical Software*, 76(1), 1–32.
[^3]: Vehtari, A., et al. (2017). Practical Bayesian model evaluation. *Statistics and Computing*, 27(5), 1413–1432.
