# Introduction to PyMC and Bayesian Modeling

## Overview

This 5-day short course, offered through the Summer Institute of the Center for Quantitative Sciences at Vanderbilt University Medical Center, provides a comprehensive introduction to Bayesian statistical modeling using PyMC. Participants will progress from foundational concepts through applied modeling techniques, building practical skills through hands-on coding exercises with real-world datasets. Each session combines conceptual instruction with interactive notebook-based exercises.

## Format

- Daily 3-hour sessions over 5 days (15 hours total), July 20–24, 2026
- Interactive instruction with hands-on coding exercises using [marimo](https://marimo.io) reactive notebooks

## Prerequisites

- Working knowledge of Python programming
- Familiarity with basic statistical concepts (distributions, regression concepts)
- No prior Bayesian experience required

## Tools

PyMC, ArviZ, PyTensor, Polars, Matplotlib/Plotly, marimo notebooks

## Setup

This course uses [Pixi](https://pixi.sh) for Python package management. To install Pixi:

On Linux/macOS:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

On Windows:
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

You may need to restart your terminal after installation.

### Getting the Course Materials

The next step is to clone or download the course materials. There are several options:

#### Option A: Using the GitHub CLI (recommended)

The [GitHub CLI](https://cli.github.com/) (`gh`) is the easiest way to clone repositories. First, install it:

- **macOS**: `brew install gh`
- **Windows**: `winget install GitHub.cli`
- **Linux**: See [installation instructions](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)

Then authenticate and clone:

```bash
gh auth login          # Follow prompts to authenticate (one-time setup)
gh repo clone fonnesbeck/cqs-pymc-course
cd cqs-pymc-course
```

#### Option B: Using Git with HTTPS

```bash
git clone https://github.com/fonnesbeck/cqs-pymc-course.git
cd cqs-pymc-course
```

If you get authentication errors, you may need to use a [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) instead of your password.

#### Option C: Using Git with SSH

If you have [SSH keys configured](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) with GitHub:

```bash
git clone git@github.com:fonnesbeck/cqs-pymc-course.git
cd cqs-pymc-course
```

#### Option D: Download as ZIP

If you prefer not to use Git, you can [download a zip file](https://github.com/fonnesbeck/cqs-pymc-course/archive/main.zip) of the materials and unzip it on your computer.


### Setting up the Environment

The repository contains a `pixi.toml` file. From the main course directory, install the environment:

    pixi install

Then open the setup notebook to verify your environment:

    pixi run marimo edit notebooks/Session_0-Setup_and_Pre-work.py

This will launch the marimo editor in your browser. The Session 0 notebook checks that all required packages are installed and provides a refresher on prerequisite skills.

To open any session notebook for interactive editing:

    pixi run marimo edit notebooks/Session_N-*.py

To run a notebook as a read-only app (code hidden by default):

    pixi run marimo run notebooks/Session_N-*.py

## Schedule

### Day 1 (July 20): **Introduction to Bayesian Inference and Prior Selection** (3 hours)

**Part A — Bayesian Inference (90 min)** — `Session_1.1-Bayesian_Inference.py`
- Probability as uncertainty quantification and Bayes' theorem (worked example: testing for a rare disease)
- The Beta-Binomial model and how posteriors evolve with data
- A/B testing: comparing two variants
- Continuous likelihoods: Bayesian estimation of rates and means
- Mini case studies: hierarchical shrinkage on baseball batting averages, Bayesian bandits

**Part B — Prior and Likelihood Selection (90 min)** — `Session_1.2-Prior_and_Likelihood_Selection.py`
- Distribution families: continuous, discrete, bounded
- Choosing likelihoods for different data types (counts, proportions, continuous)
- Choosing priors for location, scale, and rate parameters
- Prior predictive simulation as a sanity check
- Interactive prior elicitation with PreliZ; LKJ priors for correlation matrices

### Day 2 (July 21): **Building Models with PyMC** (3 hours)

**Part A — PyTensor and the PyMC API (90 min)** — `Session_2.1-PyTensor_and_PyMC_API.py`
- PyTensor: symbolic variables, computational graphs, automatic differentiation
- The PyMC `Model` context and random-variable API
- Distribution classes, observed data, and deterministics
- Custom distributions

**Part B — Building Models with PyMC (90 min)** — `Session_2.2-Building_Models_with_PyMC.py`
- Building a complete model from scratch
- Factor potentials and parameter transformations
- Prior predictive checks inside the workflow
- Prediction and forecasting with `pm.set_data` / `pm.sample_posterior_predictive`
- Worked exercise: smart-drug clinical trial

### Day 3 (July 22): **Model Fitting and Diagnostics** (3 hours)

**Part A — MCMC and Convergence Diagnostics (90 min)** — `Session_3.1-MCMC_and_Convergence_Diagnostics.py`
- Warm-up: comparing two sampling results (`DataTree` objects) in the wild
- What `pm.sample()` actually returns
- MCMC fundamentals: from Metropolis to HMC to NUTS
- Convergence diagnostics: R-hat, ESS, MCSE, divergences, energy plots

**Part B — When Sampling Fails (90 min)** — `Session_3.2-When_Sampling_Fails.py`
- Non-identifiability, divergences, and inefficient sampling — diagnosis and fixes
- Sampler configuration tips (`target_accept`, initialization, adaptation)
- Posterior predictive checks and PSIS-LOO model comparison
- Exercise: change-point analysis on coal-mining disasters

### Day 4 (July 23): **Regression and Hierarchical Models** (3 hours)

**Part A — Linear Regression (90 min)** — `Session_4.1-Linear_Regression.py`
- Bayesian linear regression: specification, fitting, interpretation
- Intercept-only baseline → linear → species-stratified models
- Model comparison with PSIS-LOO; robust regression with Student-t likelihoods
- Out-of-sample prediction with posterior predictive distributions
- Case study: fish-weight prediction, with decision analysis on the posterior predictive

**Part B — Hierarchical Models (90 min)** — `Session_4.2-Hierarchical_Models.py`
- The pooling problem: complete pooling vs. no pooling vs. partial pooling
- Shrinkage and borrowing strength across groups
- Varying-intercept and varying-slope models
- Centered vs. non-centered parameterizations
- Group-level predictors; correlated intercepts and slopes
- Case study: radon levels in Minnesota homes

### Day 5 (July 24): **Time Series and Gaussian Processes** (3 hours)

**Part A — State-Space Time Series (90 min)** — `Session_5.1-State_Space_Time_Series.py`
- Motivation: latent-state and observation equations; Kalman filter intuition
- The smallest non-trivial state-space model: position and velocity
- Structural components — the lego blocks: LevelTrend, Seasonality, Cycle, Autoregressive, Regression, MeasurementError
- Component decomposition: filtered, predicted, and smoothed states
- Scenario forecasting with structural time-series models
- Multivariate extension: restricted Bayesian VAR

**Part B — Gaussian Processes (90 min)** — `Session_5.2-Gaussian_Processes.py`
- From basis functions and random walks to priors over functions
- From the multivariate Normal to the Gaussian process
- Covariance functions and the kernel zoo (ExpQuad, Matérn, Periodic, Linear); combining kernels
- GP regression with `pm.gp.Marginal`
- Non-Gaussian observations with latent GPs; sparse approximations
- Hilbert Space GP (HSGP) approximation and 2-D spatial GPs
- Decision guide: which GP method to use when
