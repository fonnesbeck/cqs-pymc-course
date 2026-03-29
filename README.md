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

### Day 1 (July 20): **Introduction to Bayesian Inference and PyMC** (3 hours)

**Part A — Bayesian Inference (90 min):**
- Why Bayesian? Probability as uncertainty quantification
- Bayes' theorem: priors, likelihoods, and posteriors
- The Beta-Binomial model as a worked example
- Overview of common distribution families (continuous, discrete, bounded)
- Choosing likelihoods for different data types (continuous, counts, proportions)
- Choosing priors for different parameter types (location, scale, rates)
- Prior predictive simulation — checking that your model makes sense before seeing data

**Part B — Building Models with PyMC (90 min):**
- The PyMC model context and random variable API
- Core building blocks: stochastic variables, deterministic transforms, observed data
- Distribution classes and parameterizations
- Building a complete model from scratch (dose-response example)
- Prior predictive checks as part of the modeling workflow
- Parameter transformations and constraints

### Day 2 (July 21): **Model Fitting and Diagnostics** (3 hours)

**Part A — MCMC and Inference (90 min):**
- MCMC fundamentals: from Metropolis to HMC to NUTS
- Understanding `pm.sample()` output and the InferenceData object
- Convergence diagnostics: R-hat, ESS, MCSE, divergences
- Diagnosing common problems: non-identifiability, funnels, multimodality

**Part B — Model Checking and Repair (90 min):**
- Posterior predictive checks and calibration (LOO-PIT)
- Fixing problematic models: reparameterization, stronger priors
- Centered vs. non-centered parameterizations

### Day 3 (July 22): **Bayesian Regression and Workflow** (3 hours)

**Part A — Linear Regression (90 min):**
- Bayesian linear regression: specification, fitting, and interpretation
- Prior specification for regression coefficients
- Using `preliz` for principled prior selection
- Model comparison with LOO cross-validation
- Out-of-sample prediction with posterior predictive distributions

**Part B — Bayesian Workflow Case Study (90 min):**
- The iterative Bayesian workflow: explore → build → check priors → fit → diagnose → check posteriors → iterate
- Case study: COVID-19 excess deaths — counterfactual forecasting with model iteration and comparison

### Day 4 (July 23): **Generalized Linear and Hierarchical Models** (3 hours)

**Part A — Generalized Linear Models (90 min):**
- Why linear regression isn't enough: counts, proportions, skewed data
- The GLM framework: linear predictor + link function + distribution family
- Poisson regression for count data
- Handling overdispersion with Negative Binomial regression
- Logistic regression for binary outcomes

**Part B — Hierarchical Models (90 min):**
- The pooling problem: complete pooling vs. no pooling vs. partial pooling
- Shrinkage estimation and borrowing strength across groups
- Building hierarchical models: varying intercepts and varying slopes
- Diagnosing and fixing hierarchical model pathologies (funnels, divergences)
- Model comparison: pooled vs. unpooled vs. hierarchical

### Day 5 (July 24): **Time Series and Gaussian Processes** (3 hours)

**Part A — Time Series (90 min):**
- Temporal dependence and autocorrelation
- Autoregressive (AR) models in a Bayesian framework
- State space models and Gaussian random walks
- Seasonal decomposition and trend modeling (Prophet-style: trend + Fourier seasonality)
- Stochastic volatility models

**Part B — Gaussian Processes (90 min):**
- Distributions over functions: the Gaussian process
- Kernel/covariance functions: ExpQuad, Matérn, Periodic, Linear
- GP regression in PyMC with `pm.gp.Marginal`
- Hilbert Space GP (HSGP) approximation for scalability
- GPs as a flexible approach to time series and spatial data
