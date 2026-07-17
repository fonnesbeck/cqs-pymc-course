import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


with app.setup:
    import marimo as mo
    import numpy as np
    import pymc as pm
    import arviz as az
    import matplotlib.pyplot as plt
    import polars as pl
    import warnings
    from pathlib import Path
    import inspect

    az.style.use("arviz-variat")

    def fig_kwargs(cols=1, rows=1):
        """Compute reasonable figure_kwargs for arviz plots."""
        w = min(max(10, 4.5 * cols), 14)
        h = 3.5 * rows
        return {"figsize": (w, h)}

    data_path = Path(__file__).parent / "data"
    RANDOM_SEED = 20090425
    RNG = np.random.default_rng(RANDOM_SEED)
    warnings.filterwarnings("ignore", module="mkl_fft")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


@app.cell(hide_code=True)
def _():
    mo.md("""

    # Session 3.2: When Sampling Fails

    In Session 3.1 we built a well-specified model and confirmed that the sampler worked correctly. Now we'll see what happens when things go wrong — as they do — and learn to recognize, diagnose, and fix the problems. We'll also cover posterior predictive checks and model comparison.
    """)
    return


@app.cell(hide_code=True)
def _():
    penguins = (
        pl.read_csv(data_path / "penguins.csv")
        .filter(pl.col("bill_length_mm") != "NA")
        .with_columns(
            pl.col("bill_length_mm").cast(pl.Float64),
            pl.col("bill_depth_mm").cast(pl.Float64),
            pl.col("flipper_length_mm").cast(pl.Float64),
            pl.col("body_mass_g").cast(pl.Float64),
        )
    )

    flipper_length = penguins.get_column("flipper_length_mm").to_numpy()
    body_mass = penguins.get_column("body_mass_g").to_numpy()

    # Standardize predictors
    flipper_length_std = (flipper_length - flipper_length.mean()) / flipper_length.std()
    body_mass_kg = body_mass / 1000  # Work in kg for nicer numbers
    return body_mass_kg, flipper_length_std, penguins


@app.cell
def _(body_mass_kg, flipper_length_std):
    def build_baseline():
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=4, sigma=2)
            beta = pm.Normal("beta", mu=0, sigma=2)
            sigma = pm.HalfNormal("sigma", sigma=2)
            mu = alpha + beta * flipper_length_std
            pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)
        return model

    baseline_model = build_baseline()
    with baseline_model:
        baseline_trace = pm.sample(random_seed=RANDOM_SEED)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## When Sampling Fails

    Session 3.1's baseline model converged cleanly because it was well-specified. Now let's see what happens when it isn't — and learn to recognize and fix the problems.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Case 1: Non-identifiability — Poor Mixing, High R-hat

    *What happens when the model has redundant parameters?*

    Let's predict penguin body mass using species indicators using an **overparameterized** model.
    """)
    return


@app.cell(hide_code=True)
def _(penguins):
    # Create species dummy variables (keeping all three — this is the problem)
    species = penguins.get_column("species").to_numpy()
    adelie = (species == "Adelie").astype(float)
    chinstrap = (species == "Chinstrap").astype(float)
    gentoo = (species == "Gentoo").astype(float)
    return adelie, chinstrap, gentoo


@app.cell
def _(adelie, body_mass_kg, chinstrap, gentoo):
    def build_overparam():
        with pm.Model() as model:
            # Intercept
            beta_0 = pm.Normal("intercept", mu=0, sigma=100_000)

            # Species fixed effects
            beta_adelie = pm.Normal("beta_adelie", mu=0, sigma=100_000)
            beta_chinstrap = pm.Normal("beta_chinstrap", mu=0, sigma=100_000)
            beta_gentoo = pm.Normal("beta_gentoo", mu=0, sigma=100_000)

            # Sampling SD
            sigma = pm.HalfNormal("sigma", sigma=2)

            mu = (
                beta_0
                + beta_adelie * adelie
                + beta_chinstrap * chinstrap
                + beta_gentoo * gentoo
            )

            pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)
        return model

    overparam_model = build_overparam()
    overparam_model
    return (overparam_model,)


@app.cell
def _(overparam_model):
    with overparam_model:
        overparam_trace = pm.sample(2000, random_seed=RANDOM_SEED)
    return (overparam_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Diagnosing the problem

    Let's look at the summary table first.
    """)
    return


@app.cell
def _(overparam_trace):
    az.summary(overparam_trace)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Look at the R-hat values for the intercept and species coefficients — they're well above 1.01, and the ESS values have collapsed.

    But `sigma` looks fine. What's going on?

    The trace plots make the problem visible.
    """)
    return


@app.cell
def _(overparam_trace):
    az.plot_trace_dist(
        overparam_trace,
        var_names=["intercept", "beta_adelie", "beta_chinstrap", "beta_gentoo"],
        figure_kwargs=fig_kwargs(cols=4, rows=2),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The chains are wandering without settling — they explore different regions of parameter space and never agree on where to stop. This is exactly the scenario **R-hat** is designed to detect.

    Recall that R-hat compares the variance *between* chains to the variance *within* each chain. When chains are exploring different regions (as here), the between-chain variance is large relative to within-chain variance, and R-hat rises above 1.

    **Split R-hat** goes further by also splitting each chain in half, catching non-stationarity *within* a single chain — if a chain drifts over the course of sampling, the two halves will disagree.

    The rank plot makes this even clearer — well-mixed chains would show flat Δ-ECDF lines near zero within the gray envelope, but here the chains' rank distributions diverge substantially.

    Some of the samples are so extreme, you cannot even see the grey envelope!
    """)
    return


@app.cell
def _(overparam_trace):
    az.plot_rank(
        overparam_trace,
        var_names=["intercept", "beta_adelie", "beta_chinstrap"],
        figure_kwargs=fig_kwargs(cols=3),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A pair plot reveals the geometry of the problem.
    """)
    return


@app.cell
def _(overparam_trace):
    az.plot_pair(
        overparam_trace,
        var_names=["intercept", "beta_adelie", "beta_chinstrap"],
        figure_kwargs=fig_kwargs(cols=3, rows=3),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Why this happens

    The model has infinitely many parameter combinations that produce the same predictions. For example, shifting the intercept up by 1 and all species effects down by 1 gives identical fitted values. The posterior is a *ridge* (a flat direction in parameter space), not a peak. The sampler slides along this ridge without converging — the chains never agree on where on the ridge to settle.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Fix 1: Use reference coding

    Drop one species indicator so the intercept absorbs that species' mean. This removes the redundancy.
    """)
    return


@app.cell
def _(body_mass_kg, chinstrap, gentoo):
    def build_reference():
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=100_000)

            # No Adelie effect
            beta_chinstrap = pm.Normal("beta_chinstrap", mu=0, sigma=100_000)
            beta_gentoo = pm.Normal("beta_gentoo", mu=0, sigma=100_000)

            sigma = pm.HalfNormal("sigma", sigma=2)

            mu = alpha + beta_chinstrap * chinstrap + beta_gentoo * gentoo

            pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)
        return model

    with build_reference():
        reference_trace = pm.sample(random_seed=RANDOM_SEED)
    return (reference_trace,)


@app.cell
def _(reference_trace):
    az.summary(reference_trace)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Fix 2: Use informative priors

    Alternatively, keep all indicators but use priors that regularize. Informative priors reshape the ridge into a proper peak by "pulling" parameters toward specific values.
    """)
    return


@app.cell
def _(adelie, body_mass_kg, chinstrap, gentoo):
    def build_regularized():
        with pm.Model() as model:
            beta_0 = pm.Normal("intercept", mu=4, sigma=2)
            beta_adelie = pm.Normal("beta_adelie", mu=0, sigma=2)
            beta_chinstrap = pm.Normal("beta_chinstrap", mu=0, sigma=2)
            beta_gentoo = pm.Normal("beta_gentoo", mu=0, sigma=2)
            sigma = pm.HalfNormal("sigma", sigma=2)

            mu = (
                beta_0
                + beta_adelie * adelie
                + beta_chinstrap * chinstrap
                + beta_gentoo * gentoo
            )

            pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)
        return model

    with build_regularized():
        regularized_trace = pm.sample(random_seed=RANDOM_SEED)
    return (regularized_trace,)


@app.cell
def _(regularized_trace):
    az.summary(
        regularized_trace,
        var_names=[
            "intercept",
            "beta_adelie",
            "beta_chinstrap",
            "beta_gentoo",
            "sigma",
        ],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ### Case 2: Divergences — Posterior Geometry Problems

    Divergences are NUTS telling you that the sampler is systematically *under-exploring* some region of the posterior, which potentially **biases your inference** — not just makes it noisier.

    For example: Our baseline model used a Normal likelihood, which assumes symmetric, light-tailed noise. What if the data has outliers? A **Student-t likelihood** (as we saw in our discussion of likelihoods) is more robust — but it introduces a degrees-of-freedom parameter `nu` that can create difficult **posterior geometry**.
    """)
    return


@app.cell
def _(body_mass_kg, flipper_length_std):
    def build_robust():
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=4, sigma=2)
            beta = pm.Normal("beta", mu=0, sigma=2)
            sigma = pm.HalfNormal("sigma", sigma=2)
            nu = pm.Exponential("nu", lam=1 / 30)

            mu = alpha + beta * flipper_length_std

            pm.StudentT("mass", nu=nu, mu=mu, sigma=sigma, observed=body_mass_kg)
        return model

    with build_robust():
        robust_trace = pm.sample(1000, random_seed=RANDOM_SEED, target_accept=0.8)
    return (robust_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Diagnosing the problem

    The sampling output should report divergent transitions. Let's extract and visualize them.
    """)
    return


@app.cell
def _(robust_trace):
    robust_trace.sample_stats["diverging"]
    return


@app.cell
def _(robust_trace):
    az.summary(robust_trace)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The regression parameters (`alpha`, `beta`, `sigma`) look healthy — good R-hat and ESS. But look at `nu`: the ESS may be lower, and the divergences suggest the sampler struggled in some region of the posterior.

    A pair plot reveals where the trouble is.
    """)
    return


@app.cell
def _(robust_trace):
    az.plot_pair(
        robust_trace,
        var_names=["nu", "sigma"],
        visuals={"divergence": {"color": "red"}},
        figure_kwargs=fig_kwargs(cols=2, rows=2),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Why this happens

    When `nu` is small (near 1–2), the Student-t distribution has very heavy tails, and the likelihood surface changes rapidly — small shifts in `nu` have outsized effects. The sampler's step size is tuned for the bulk of the posterior where `nu` is moderate, but in the high-curvature region near small `nu`, that step size is too large for accurate trajectories.

    The sampler systematically under-explores the high-curvature region, which can bias your estimate if severe enough.

    You'll see a more extreme version of this geometry in Session 4.2 when we build hierarchical models.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Fix: Increase target acceptance rate

    Increasing `target_accept` forces the sampler to use a smaller step size when simulating the trajectory for a new sample, navigating high-curvature regions more carefully — at the cost of slightly slower sampling.
    """)
    return


@app.cell
def _(body_mass_kg, flipper_length_std):
    def build_robust_fixed():
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=4, sigma=2)
            beta = pm.Normal("beta", mu=0, sigma=2)
            sigma = pm.HalfNormal("sigma", sigma=2)
            nu = pm.Exponential("nu", lam=1 / 30)

            mu = alpha + beta * flipper_length_std

            pm.StudentT("mass", nu=nu, mu=mu, sigma=sigma, observed=body_mass_kg)
        return model

    with build_robust_fixed():
        robust_trace_fixed = pm.sample(
            1000, random_seed=RANDOM_SEED, target_accept=0.95
        )
    return (robust_trace_fixed,)


@app.cell
def _(robust_trace_fixed):
    _divergences_fixed = robust_trace_fixed.sample_stats["diverging"].values.sum()
    mo.md(f"Divergences with `target_accept=0.95`: **{_divergences_fixed}**")
    az.summary(robust_trace_fixed, var_names=["nu"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ### Case 3: Inefficient Sampling — Low ESS, High Autocorrelation

    *What happens when the sampler can't explore the posterior efficiently?*

    Case 1 was a model specification problem, while Case 2 was a hyperparameter misspecification issue that the sampler exposed. But sometimes the sampler itself is the bottleneck — either because a less efficient algorithm is being used, or because the model structure forces it.

    We already saw a preview of this when we compared NUTS to Metropolis in the previous session. Now let's see a more realistic scenario: a model where **PyMC automatically uses Metropolis for some parameters** because they're discrete.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### A model with discrete and continuous parameters

    Let's predict body mass using a latent group indicator — a discrete variable that PyMC can't sample with NUTS.
    """)
    return


@app.cell
def _(body_mass_kg):
    def build_mixed_sampler():
        with pm.Model() as model:
            # Discrete parameter — PyMC uses Metropolis sampling
            group = pm.Categorical("group", p=np.ones(3) / 3, shape=len(body_mass_kg))

            # Continuous parameters — sampled with NUTS
            mu_groups = pm.Normal("mu_groups", mu=4, sigma=2, shape=3)
            sigma = pm.HalfNormal("sigma", sigma=2)

            pm.Normal("mass", mu=mu_groups[group], sigma=sigma, observed=body_mass_kg)
        return model

    with build_mixed_sampler():
        mixed_trace = pm.sample(2000, random_seed=RANDOM_SEED)
    return (mixed_trace,)


@app.cell
def _(mixed_trace):
    az.summary(
        mixed_trace, var_names=["mu_groups", "sigma", "group"], filter_vars="like"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `sigma` parameter has high ESS — NUTS handles it efficiently. But the discrete `group` parameter has dramatically lower ESS, and the `mu_groups` parameters are dragged down with it — their values depend on which observations are assigned to which group, so poor mixing in `group` propagates to the group means.

    This is the **ESS in action**. When you see ESS much lower than your nominal draw count, it means the sampler is producing correlated draws in that region of the model.

    **When might you encounter this in practice?**

    - Models with discrete latent variables (mixture models, change-point models)
    - Models where PyMC falls back to Metropolis for some parameters
    - Models with highly correlated posteriors even for continuous parameters

    The fix depends on the cause: for discrete parameters, consider whether you can marginalize them out; for correlated posteriors, try reparameterization.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sampler Configuration Tips

    Now that you've seen what can go wrong, here are the practical `pm.sample()` settings that help:

    - **`target_accept`**: Default 0.8. Increase to 0.95+ when you see divergences. This uses a smaller step size, trading speed for accuracy.
    - **`tune`**: Default 1000. The warmup period where the sampler adapts its step size and mass matrix. Increase for complex models.
    - **`draws`**: Default 1000. How many posterior samples per chain. Increase if ESS is too low.
    - **`chains`** and **`cores`**: Default 4 chains on 4 cores. Multiple chains are required for R-hat. Never use just 1 chain.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Does the Model Fit?

    Everything so far has been about the **sampler**: did it converge, did it explore efficiently, can we trust the Monte Carlo estimates? These are necessary checks, but they answer a fundamentally different question from what comes next.

    A perfectly sampled posterior from a bad model is still a bad model. A model can converge beautifully, show healthy R-hat and ESS across the board, and still make terrible predictions. Sampler diagnostics tell you the *computation* worked; they say nothing about whether the *model* is any good.

    Now we ask: **could this model have generated the observed data?**

    We'll use two complementary approaches:

    1. **Posterior predictive checks** — absolute model quality: "Does this model describe the data?"
    2. **LOO cross-validation and model comparison** — relative model quality: "Which model describes the data better?"
    """)
    return


@app.cell
def _(body_mass_kg, chinstrap, flipper_length_std, gentoo):
    def build_species():
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=4, sigma=2)
            beta_flipper = pm.Normal("beta_flipper", mu=0, sigma=2)
            beta_chinstrap = pm.Normal("beta_chinstrap", mu=0, sigma=2)
            beta_gentoo = pm.Normal("beta_gentoo", mu=0, sigma=2)
            sigma = pm.HalfNormal("sigma", sigma=2)

            mu = (
                alpha
                + beta_flipper * flipper_length_std
                + beta_chinstrap * chinstrap
                + beta_gentoo * gentoo
            )
            pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)
        return model

    species_model = build_species()
    species_model
    return (species_model,)


@app.cell
def _(species_model):
    with species_model:
        species_trace = pm.sample(random_seed=RANDOM_SEED)
    return (species_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Posterior Predictive Checks

    The idea is simple: for each posterior draw of the *fitted* parameters, simulate a new dataset from the likelihood. This gives a distribution of datasets the model considers plausible. If the real data look nothing like these simulated datasets, the model is missing something important.

    We already checked the *prior* predictive distribution previously — that demonstrated whether our priors were reasonable. Now we check the *posterior* predictive — whether the fitted model captures the patterns in the data.
    """)
    return


@app.cell
def _(body_mass_kg):
    def build_pooled():
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=4, sigma=2)
            sigma = pm.HalfNormal("sigma", sigma=2)

            pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)
        return model

    pooled_model = build_pooled()
    with pooled_model:
        pooled_trace = pm.sample(random_seed=RANDOM_SEED)
        pm.sample_posterior_predictive(
            pooled_trace, extend_inferencedata=True, random_seed=RANDOM_SEED
        )
    return pooled_model, pooled_trace


@app.cell
def _(pooled_trace):
    az.plot_ppc_dist(pooled_trace, num_samples=100, figure_kwargs=fig_kwargs())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The pooled model's predictive distribution is unimodal — it can only produce a single bell curve. But the observed data has two clear modes corresponding to different species. The posterior predictive check reveals this misfit immediately.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now let's compare with a model that accounts for species differences. Adding species indicators (reference-coded) alongside flipper length should capture the bimodal structure.
    """)
    return


@app.cell
def _(species_model, species_trace):
    with species_model:
        pm.sample_posterior_predictive(species_trace, extend_inferencedata=True)
    return


@app.cell
def _(species_trace):
    az.plot_ppc_dist(species_trace, num_samples=100, figure_kwargs=fig_kwargs())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The species-aware model captures the bimodal structure — the two peaks correspond to smaller species (Adélie, Chinstrap) and the larger Gentoo. The posterior predictive draws track both modes well, a clear improvement over the pooled model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Bayesian p-values

    `az.plot_ppc_tstat()` provides a more formal check by computing a **test statistic** (by default, the median) across the posterior predictive samples and comparing it to the same statistic computed on the observed data.

    The plot shows the distribution of the test statistic computed from each posterior predictive draw, with the observed value marked. If the observed statistic falls well within the posterior predictive distribution, the model captures that aspect of the data. If it falls in the tails, the model is systematically missing something.

    You can change the test statistic using `t_stat=` — try `"mean"`, `"std"`, `"var"`, `"min"`, `"max"`, `"iqr"`, or `"mad"` to probe different aspects of the data.
    """)
    return


@app.cell(hide_code=True)
def _(pooled_trace, species_trace):
    p_pooled = az.plot_ppc_tstat(pooled_trace, figure_kwargs=fig_kwargs())
    p_species = az.plot_ppc_tstat(species_trace, figure_kwargs=fig_kwargs())

    _figs = [plt.figure(n) for n in plt.get_fignums()[-2:]]
    _xlims = [ax.get_xlim() for fig in _figs for ax in fig.axes]
    _shared = (min(x[0] for x in _xlims), max(x[1] for x in _xlims))
    for _fig in _figs:
        for _ax in _fig.axes:
            _ax.set_xlim(*_shared)
    _figs[0].axes[0].set_title("mass (median) — pooled model", fontsize=12)
    _figs[1].axes[0].set_title("mass (median) — species model", fontsize=12)

    mo.vstack([_figs[0], _figs[1]])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ### Leave-One-Out Cross-Validation

    **PSIS-LOO-CV** (Pareto Smoothed Importance Sampling Leave-One-Out Cross-Validation) estimates how well the model predicts each observation when that observation is held out. Crucially, it approximates this efficiently without refitting the model.

    A model can "win" a LOO comparison and still be a bad model in absolute terms. LOO tells you which model is *less wrong*, not which is *right* — always combine it with posterior predictive checks.
    """)
    return


@app.cell
def _(species_model, species_trace):
    with species_model:
        pm.compute_log_likelihood(species_trace)
    return


@app.cell
def _(pooled_model, pooled_trace):
    with pooled_model:
        pm.compute_log_likelihood(pooled_trace)
    return


@app.cell
def _(species_trace):
    species_loo = az.loo(species_trace)
    species_loo
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Key outputs:
    - **elpd_loo**: Expected log pointwise predictive density. Higher (less negative) is better.
    - **p_loo**: Effective number of parameters. A rough measure of model complexity.
    - **Pareto k diagnostic**: Values > 0.7 indicate observations where the LOO approximation is unreliable.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### LOO Probability Integral Transform (LOO-PIT)

    LOO-PIT is a calibration check. If the model is well-calibrated, the LOO-PIT values should follow a **uniform distribution**.

    The plot shows the **Δ-ECDF** — the difference between the empirical CDF of the LOO-PIT values and the expected uniform CDF. A well-calibrated model produces a flat line near zero, staying within the gray simultaneous confidence envelope. Deviations reveal systematic misfit:

    - **Positive hump then negative**: The model is overconfident — predictive intervals are too narrow
    - **Negative dip then positive**: The model is underconfident — predictive intervals are too wide
    - **Consistently above or below zero**: Systematic bias — the model consistently over- or under-predicts
    """)
    return


@app.cell
def _(species_trace):
    az.plot_loo_pit(species_trace, var_names=["mass"], figure_kwargs=fig_kwargs())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The species-aware model's LOO-PIT stays within the confidence envelope — the model is well-calibrated.
    """)
    return


@app.cell(hide_code=True)
def _(pooled_trace):
    az.plot_loo_pit(pooled_trace, var_names=["mass"], figure_kwargs=fig_kwargs())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The pooled model shows clear deviations outside the envelope — its predictive intervals are systematically miscalibrated.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ### Model Comparison

    `az.compare()` ranks models by their expected log pointwise predictive density (ELPD). The model with the highest ELPD (least negative) is ranked first. The `weight` column gives **stacking weights** — an estimate of how much each model should contribute to an optimal predictive mixture.
    """)
    return


@app.cell
def _(pooled_trace, species_trace):
    model_comparison = az.compare(
        {
            "species + flipper": species_trace,
            "pooled (no predictors)": pooled_trace,
        },
    )
    model_comparison
    return (model_comparison,)


@app.cell
def _(model_comparison):
    az.plot_compare(model_comparison, figure_kwargs=fig_kwargs())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## The Evaluation Workflow

    Here's a systematic workflow you can follow for every model you build:

    ### Step 0: Check Priors
    ```python
    with model:
        prior = pm.sample_prior_predictive()
    az.plot_ppc_dist(prior, group="prior_predictive")  # Plausible data?
    ```

    ### Step 1: Sample
    ```python
    with model:
        trace = pm.sample(random_seed=RANDOM_SEED)
    ```
    Note any warnings (divergences, BFMI, treedepth).

    ### Step 2: Check Convergence
    ```python
    az.summary(trace)          # R-hat < 1.01? ESS > 400?
    az.plot_trace_dist(trace)  # Mixing well?
    az.plot_rank(trace)        # Flat Δ-ECDF lines?
    az.plot_energy(trace)      # Good BFMI?
    ```

    **If problems:**

    | Symptom | Diagnostic | Likely cause | Fix |
    |---------|-----------|-------------|-----|
    | High R-hat, low ESS | `az.plot_pair()` | Non-identifiability | Reparameterize, add constraints |
    | Divergences | `az.plot_pair()` | Funnel geometry | Non-centered parameterization, increase `target_accept` |
    | Low BFMI (< 0.3) | `az.plot_energy()` | Poor prior scaling | Rescale priors, reparameterize |

    ### Step 3: Check Monte Carlo Error
    ```python
    az.mcse(trace)             # MCSE small relative to posterior SD?
    az.plot_mcse(trace)        # Stable across quantiles?
    ```
    **Rule of thumb:** MCSE / posterior SD should be < 0.1. MCSE tells you how many decimal places you can trust — if MCSE ≈ 0.002, reporting to two decimal places is justified. If MCSE is too high: increase draws (or fix sampling first).

    ### Step 4: Check Model Fit
    ```python
    pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    az.plot_ppc_dist(trace)    # Predictions match data?
    az.loo(trace)              # Reasonable elpd, no high Pareto k?
    az.plot_khat(loo_result)   # Identify problematic observations
    az.plot_loo_pit(trace, var_names=["y"])     # Calibration check
    ```

    ### Step 5: Compare Models (if applicable)
    ```python
    az.compare({"model_a": trace_a, "model_b": trace_b})
    az.plot_compare(comparison)
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Exercise

    The British coal mining disasters dataset records the number of coal mining disasters per year in the UK from 1851 to 1961. It's widely believed that safety regulations introduced in the late 19th century led to a reduction in the disaster rate at some unknown **change point**.

    The model assumes disaster counts follow a Poisson distribution with a rate that switches at an unknown year:

    $$
    \begin{align}
    \lambda_{\text{early}} &\sim \text{Exponential}(\theta_1) \\
    \lambda_{\text{late}} &\sim \text{Exponential}(\theta_2) \\
    \tau &\sim \text{Uniform}(1851, 1962) \\
    y_t &\sim \text{Poisson}\!\left(\lambda_t\right), \quad \lambda_t = \begin{cases} \lambda_{\text{early}} & \text{if } t \leq \tau \\ \lambda_{\text{late}} & \text{if } t > \tau \end{cases}
    \end{align}
    $$

    This model is implemented below. Your task is to apply the full evaluation workflow:

    1. Run the model and identify all problems using diagnostics
    2. Fix the sampling issues
    3. Evaluate model fit with posterior predictive checks
    4. Compare against an improved model using LOO
    """)
    return


@app.cell(hide_code=True)
def _():
    # Coal mining disasters data
    disasters_array = np.array(
        [
            4,
            5,
            4,
            0,
            1,
            4,
            3,
            4,
            0,
            6,
            3,
            3,
            4,
            0,
            2,
            6,
            3,
            3,
            5,
            4,
            5,
            3,
            1,
            4,
            4,
            1,
            5,
            5,
            3,
            4,
            2,
            5,
            2,
            2,
            3,
            4,
            2,
            1,
            3,
            2,
            2,
            1,
            1,
            1,
            1,
            3,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            3,
            1,
            0,
            3,
            2,
            2,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            2,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            2,
            3,
            3,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            2,
            4,
            2,
            0,
            0,
            1,
            4,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
        ]
    )
    years = np.arange(1851, 1962, dtype=int)

    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.vlines(years, 0, disasters_array, lw=4)
    _ax.set(xlabel="Year", ylabel="Disasters", xlim=(1850, 1963), ylim=(0, None))
    _fig
    return disasters_array, years


@app.cell
def _(disasters_array, years):
    def build_exercise():
        with pm.Model() as model:
            early_lambda = pm.Exponential("early_lambda", lam=100)
            late_lambda = pm.Exponential("late_lambda", lam=100)
            change_point = pm.Uniform("change_point", lower=1851, upper=1962)

            lam = pm.math.switch(years > change_point, late_lambda, early_lambda)

            pm.Poisson("disasters", mu=lam, observed=disasters_array)
        return model

    with build_exercise():
        exercise_trace = pm.sample(1000, random_seed=RANDOM_SEED)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Your workflow:

    **Step 1:** Run convergence diagnostics on `exercise_trace` — `az.summary()` (R-hat, ESS), `az.plot_trace_dist()` (mixing), `az.plot_energy()` (BFMI), and check `sample_stats` for divergences.

    What problems do you see? (Hint: there are at least two distinct issues.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Step 2:** Fix the priors so they are on an appropriate scale for count data ranging from 0 to ~6 per year.

    Hint: disaster counts range from 0–6 per year. What prior means are implied by `Exponential(lam=100)`?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Step 3:** Evaluate model fit with posterior predictive checks:

    ```python
    with build_single_cp_fixed():
        pm.sample_posterior_predictive(single_cp_trace, extend_inferencedata=True)
    az.plot_ppc_dist(single_cp_trace)
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Step 4:** Build an improved model with **two change points** (allowing for an intermediate-rate period between two change points), and compare against the single-change-point model using LOO:

    ```python
    with build_single_cp_fixed():
        pm.compute_log_likelihood(single_cp_trace)
    az.compare({"single-changepoint": ..., "two-changepoint": ...})
    ```
    """)
    return


@app.cell
def _(disasters_array, years):
    # YOUR CODE HERE — fix the priors, then build a two-change-point model.
    def build_single_cp_fixed():
        with pm.Model() as model:
            early_lambda = ...
            late_lambda = ...
            change_point = ...
            lam = pm.math.switch(years > change_point, late_lambda, early_lambda)
            pm.Poisson("disasters", mu=lam, observed=disasters_array)
        return model

    def build_two_cp():
        with pm.Model() as model:
            early_lambda = ...
            mid_lambda = ...
            late_lambda = ...
            cp1 = ...
            cp2 = ...
            lam = pm.math.switch(
                years < cp1,
                early_lambda,
                pm.math.switch(years < cp2, mid_lambda, late_lambda),
            )
            pm.Poisson("disasters", mu=lam, observed=disasters_array)
        return model

    with build_single_cp_fixed():
        single_cp_trace = pm.sample(1000, random_seed=RANDOM_SEED)
    with build_two_cp():
        two_cp_trace = pm.sample(1000, random_seed=RANDOM_SEED)
    single_cp_trace, two_cp_trace
    return


@app.cell(hide_code=True)
def _(disasters_array, years):
    def solution_change_point():
        # Step 2: fix the priors on the single change-point model.
        with pm.Model() as exercise_fixed:
            early_lambda = pm.Exponential("early_lambda", lam=0.3)
            late_lambda = pm.Exponential("late_lambda", lam=0.5)
            change_point = pm.Uniform("change_point", lower=1851, upper=1962)
            lam = pm.math.switch(years > change_point, late_lambda, early_lambda)
            pm.Poisson("disasters", mu=lam, observed=disasters_array)
            exercise_trace_fixed = pm.sample(1000, random_seed=RANDOM_SEED)

        # Step 3: posterior predictive check.
        with exercise_fixed:
            pm.sample_posterior_predictive(
                exercise_trace_fixed, extend_inferencedata=True, random_seed=RANDOM_SEED
            )
        ppc_plot = az.plot_ppc_dist(exercise_trace_fixed)

        # Step 4: two change-point model and LOO comparison.
        with pm.Model() as two_cp_model:
            early_lambda_2 = pm.Exponential("early_lambda", lam=0.3)
            mid_lambda_2 = pm.Exponential("mid_lambda", lam=0.3)
            late_lambda_2 = pm.Exponential("late_lambda", lam=0.5)
            cp1 = pm.Uniform("change_point_1", lower=1851, upper=1962)
            # cp2's lower bound is cp1: without this ordering constraint the two change
            # points are exchangeable and the posterior is bimodal by construction —
            # exactly the identifiability failure this session is about.
            cp2 = pm.Uniform("change_point_2", lower=cp1, upper=1962)
            lam_2 = pm.math.switch(
                years < cp1,
                early_lambda_2,
                pm.math.switch(years < cp2, mid_lambda_2, late_lambda_2),
            )
            pm.Poisson("disasters", mu=lam_2, observed=disasters_array)
            two_cp_trace = pm.sample(1000, random_seed=RANDOM_SEED)

        with exercise_fixed:
            pm.compute_log_likelihood(exercise_trace_fixed)
        with two_cp_model:
            pm.compute_log_likelihood(two_cp_trace)

        comparison = az.compare(
            {
                "single-changepoint": exercise_trace_fixed,
                "two-changepoint": two_cp_trace,
            }
        )
        return mo.vstack(
            [
                mo.md(
                    "**Posterior predictive check (fixed single change-point model):**"
                ),
                ppc_plot,
                mo.md("**LOO comparison:**"),
                comparison,
            ]
        )

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_change_point)}\n```"
                    ),
                    mo.lazy(solution_change_point, show_loading_indicator=True),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Key Takeaways

    **Sampler magic.** `pm.sample()` constructs a Markov chain that *approximates* the posterior. NUTS does this remarkably well for continuous parameters — using gradients to take large, informed steps instead of Metropolis's blind random walk.

    **Diagnostics answer two different questions.** R-hat, ESS, divergences, and BFMI tell you whether the *computation* worked — did the *sampler* converge and explore efficiently? Posterior predictive checks and LOO tell you whether the *model* is any good. A perfectly sampled posterior from a bad model is still a bad model.

    **Most sampling failures are model problems, not sampler problems.**

    - Non-identifiable parameters (redundant intercept + all dummies) create ridges the sampler slides along forever. The fix is in the model: drop the redundancy or regularize with informative priors.
    - Divergences mean the sampler's trajectory was *geometrically challenging* in some region — your posterior estimates are biased, not just noisy. The classic cause is funnel geometry (hierarchical models where group variance can approach zero). Non-centered parameterization breaks the funnel.
    - Low ESS usually means either the wrong sampler (Metropolis for discrete parameters) or a posterior the sampler can't navigate efficiently. Marginalize, reparameterize, or increase draws.

    Posterior predictive checks compare simulated data to real data — if they don't match, the model is missing something. LOO cross-validation tells you which model predicts better, but "better" is relative — the winner can still be bad.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## References

    - Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
    - Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
    - Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.
    - Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in Bayesian workflow. *Journal of the Royal Statistical Society: Series A*, 182(2), 389-402.
    """)
    return


if __name__ == "__main__":
    app.run()
