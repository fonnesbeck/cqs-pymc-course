import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pymc as pm
    import arviz as az
    import matplotlib.pyplot as plt
    import polars as pl
    import plotly.graph_objects as go
    import plotly.io as pio
    import base64
    import warnings
    from pathlib import Path
    from pymc_extras import as_model

    PYMC_BLUE = "#154A72"
    PYMC_GREEN = "#81C240"
    PYMC_LIGHT_BLUE = "#4A9EDE"
    PYMC_DARK_GREEN = "#40611F"

    data_path = Path(__file__).parent / "data"

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="The figure layout has changed")
    RANDOM_SEED = 42
    rng = np.random.default_rng(RANDOM_SEED)
    return (
        PYMC_BLUE,
        Path,
        RANDOM_SEED,
        as_model,
        az,
        base64,
        data_path,
        go,
        np,
        pl,
        plt,
        pm,
    )


@app.cell(hide_code=True)
def header(Path, base64, mo):
    logo_path = Path(__file__).parent / "images" / "pymc-labs-logo.png"
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="300" style="margin-bottom: 0.5rem;">'
    else:
        logo_html = ""

    mo.md(f"""
    {logo_html}

    # Model Fitting and Checking
    """)
    return

@app.cell(hide_code=True)
def _(data_path, pl):
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

    flipper_length_std = (flipper_length - flipper_length.mean()) / flipper_length.std()
    body_mass_kg = body_mass / 1000
    penguins
    return (
        body_mass,
        body_mass_kg,
        flipper_length,
        flipper_length_std,
        penguins,
    )

@app.cell
def _(as_model, body_mass_kg, flipper_length_std, pm):
    @as_model()
    def _baseline():
        alpha = pm.Normal("alpha", mu=4, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=2)
        mu = alpha + beta * flipper_length_std
        pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)

    baseline_model = _baseline()
    baseline_model
    return (baseline_model,)

@app.cell
def _(RANDOM_SEED, baseline_model, pm):
    with baseline_model:
        baseline_trace = pm.sample(random_seed=RANDOM_SEED)
    baseline_trace
    return (baseline_trace,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Does the Model Fit?

    Everything so far has been about the **sampler**: did it converge, did it explore efficiently, can we trust the Monte Carlo estimates? These are necessary checks, but they answer a fundamentally different question from what comes next.

    A perfectly sampled posterior from a bad model is still a bad model. A model can converge beautifully, show healthy R-hat and ESS across the board, and still make terrible predictions. Sampler diagnostics tell you the *computation* worked; they say nothing about whether the *model* is any good.

    Now we ask: **does the model actually describe the data?**

    We'll use two complementary approaches:

    1. **Posterior predictive checks** — absolute model quality: "Does this model describe the data?"
    2. **LOO cross-validation and model comparison** — relative model quality: "Which model describes the data better?"
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posterior Predictive Checks

    Posterior predictive checks (PPCs) assess **absolute model quality**: does this model produce data that looks like the real data?

    The idea is simple. For each posterior draw of the parameters, simulate a new dataset from the likelihood. This gives a distribution of datasets the model considers plausible. If the real data look nothing like these simulated datasets, the model is missing something important — regardless of how it compares to other models.

    We already checked the *prior* predictive distribution earlier — that told us whether our priors were reasonable. Now we check the *posterior* predictive — whether the fitted model captures the patterns in the data.
    """)
    return


@app.cell
def _(RANDOM_SEED, baseline_model, baseline_trace, pm):
    with baseline_model:
        pm.sample_posterior_predictive(
            baseline_trace, extend_inferencedata=True, random_seed=RANDOM_SEED
        )
    return


@app.cell(hide_code=True)
def _(az, baseline_trace):
    az.plot_ppc_dist(baseline_trace, num_samples=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dark line is the observed data distribution, and the light lines are individual posterior predictive draws. When they overlap well, the model captures the data-generating process.

    The distributional overlay is one lens. Another is to look at predictions **per observation** — `az.plot_ppc_interval` draws a posterior predictive interval at each data point, overlaid on the observed value. This surfaces *where* the model misfits, not just whether it does overall.
    """)
    return


@app.cell
def _(az, baseline_trace):
    az.plot_ppc_interval(baseline_trace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's compare with a model that *doesn't* fit well. A simple pooled model (ignoring species differences) will miss the multimodal structure of penguin body mass.
    """)
    return


@app.cell
def _(RANDOM_SEED, as_model, body_mass_kg, pm):
    @as_model()
    def _pooled():
        mu = pm.Normal("mu", mu=4, sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=2)
        pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)

    pooled_model = _pooled()
    with pooled_model:
        pooled_trace = pm.sample(random_seed=RANDOM_SEED)
        pm.sample_posterior_predictive(
            pooled_trace, extend_inferencedata=True, random_seed=RANDOM_SEED
        )
    pooled_model
    return pooled_model, pooled_trace


@app.cell(hide_code=True)
def _(az, pooled_trace):
    az.plot_ppc_dist(pooled_trace, num_samples=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The pooled model's predictive distribution is unimodal but the data has multiple modes (corresponding to different species). The posterior predictive check reveals the misfit clearly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Bayesian p-values

    `az.plot_ppc_tstat()` provides a more formal check by computing a **test statistic** (by default, the median) across the posterior predictive samples and comparing it to the same statistic computed on the observed data.

    The plot shows the distribution of the test statistic computed from each posterior predictive draw, with the observed value marked. If the observed statistic falls well within the posterior predictive distribution, the model captures that aspect of the data. If it falls in the tails, the model is systematically missing something.

    You can change the test statistic using `t_stat=` — try `"mean"`, `"std"`, `"var"`, `"min"`, `"max"`, `"iqr"`, or `"mad"` to probe different aspects of the data.
    """)
    return


@app.cell(hide_code=True)
def _(az, baseline_trace, pooled_trace):
    az.plot_ppc_tstat(baseline_trace)
    az.plot_ppc_tstat(pooled_trace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The median is the default, but the right `t_stat` depends on what you suspect the model gets wrong. The mean should hug observed reality for a well-fit linear model; standard deviation probes spread; interquartile range probes the middle; the max probes the tail — linear-Gaussian models rarely capture extremes well. Below: the same `baseline_trace`, four statistics.
    """)
    return


@app.cell
def _(az, baseline_trace, mo):
    tstat_panels = []
    for tstat_name in ("mean", "std", "iqr", "max"):
        tstat_panels.append(mo.md(f'**`t_stat = "{tstat_name}"`**'))
        tstat_panels.append(az.plot_ppc_tstat(baseline_trace, t_stat=tstat_name))
    mo.vstack(tstat_panels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Leave-One-Out Cross-Validation

    Posterior predictive checks tell you whether a model fits the data in absolute terms. But when comparing models, we need a different tool — one that measures **relative predictive performance**.

    **PSIS-LOO-CV** (Pareto Smoothed Importance Sampling Leave-One-Out Cross-Validation) estimates how well the model predicts each observation when that observation is held out. Crucially, it approximates this efficiently without refitting the model.

    Note: a model can "win" a LOO comparison and still be a bad model in absolute terms. LOO tells you which model is *less wrong*, not which is *right*. Always combine LOO with posterior predictive checks.
    """)
    return


@app.cell
def _(baseline_model, baseline_trace, pm):
    with baseline_model:
        pm.compute_log_likelihood(baseline_trace)
    return


@app.cell
def _(pm, pooled_model, pooled_trace):
    with pooled_model:
        pm.compute_log_likelihood(pooled_trace)
    return


@app.cell
def _(az, baseline_trace):
    baseline_loo = az.loo(baseline_trace)
    baseline_loo
    return (baseline_loo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Key outputs:
    - **elpd_loo**: Expected log pointwise predictive density. Higher (less negative) is better.
    - **p_loo**: Effective number of parameters. A rough measure of model complexity.
    - **Pareto k diagnostic**: Values > 0.7 indicate observations where the LOO approximation is unreliable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### LOO Probability Integral Transform (LOO-PIT)

    LOO-PIT is a calibration check. If the model is well-calibrated, the LOO-PIT values should follow a **uniform distribution**.

    The plot shows the **Δ-ECDF** — the difference between the empirical CDF of the LOO-PIT values and the expected uniform CDF. A well-calibrated model produces a flat line near zero, staying within the gray simultaneous confidence envelope. Deviations reveal systematic misfit:

    - **Positive hump then negative**: The model is overconfident — predictive intervals are too narrow
    - **Negative dip then positive**: The model is underconfident — predictive intervals are too wide
    - **Consistently above or below zero**: Systematic bias — the model consistently over- or under-predicts
    """)
    return


@app.cell(hide_code=True)
def _(az, baseline_trace, pooled_trace):
    az.plot_loo_pit(baseline_trace, var_names=["mass"])
    az.plot_loo_pit(pooled_trace, var_names=["mass"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In-sample vs out-of-sample PIT.** `plot_loo_pit` (above) uses leave-one-out cross-validation — each observation is scored against a posterior that *didn't see it*. The in-sample sibling, `plot_ppc_pit`, uses the raw posterior predictive without holding anything out. Both should look uniform for a well-specified model; the LOO version is more honest because the model has been "shown" each point during fitting in the in-sample version. **A telling diagnostic: when the two PIT plots disagree (especially when LOO looks worse), the model is overfitting.**
    """)
    return


@app.cell
def _(az, baseline_trace, pooled_trace):
    az.plot_ppc_pit(baseline_trace, var_names=["mass"])
    az.plot_ppc_pit(pooled_trace, var_names=["mass"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Model Comparison

    `az.compare()` ranks models by their expected log pointwise predictive density (ELPD). The model with the highest ELPD (least negative) is ranked first. The `weight` column gives **stacking weights** — an estimate of how much each model should contribute to an optimal predictive mixture.
    """)
    return


@app.cell
def _(az, baseline_trace, pooled_trace):
    model_comparison = az.compare(
        {
            "baseline (flipper length)": baseline_trace,
            "pooled (no predictors)": pooled_trace,
        },
    )
    model_comparison
    return (model_comparison,)


@app.cell(hide_code=True)
def _(az, model_comparison):
    az.plot_compare(model_comparison)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Pointwise Pareto k Diagnostics

    We can examine the Pareto k diagnostics for each model to identify influential observations — points where the model's predictions are heavily influenced by individual data points.

    Each point is one observation. Points above the 0.7 threshold are highly influential — they disproportionately affect the model's predictions. These are often outliers or observations the model struggles with. Investigate them!
    """)
    return


@app.cell
def _(az, baseline_loo, pooled_trace):
    pooled_loo = az.loo(pooled_trace)

    az.plot_khat(baseline_loo)
    az.plot_khat(pooled_loo)
    return


@app.cell(hide_code=True)
def _(mo):
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
    az.plot_rank_dist(trace)   # Flat Δ-ECDF lines?
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
    az.plot_ppc_dist(trace)         # Predictions match data overall?
    az.plot_ppc_interval(trace)     # Predictions match data per observation?
    az.plot_ppc_tstat(trace, t_stat="mean")  # Tail-sensitive: try "std", "iqr", "max"
    az.plot_ppc_pit(trace, var_names=["y"])  # In-sample calibration
    az.plot_loo_pit(trace, var_names=["y"])  # Out-of-sample calibration
    az.plot_ppc_rootogram(trace)    # Use for count data (Poisson, NegBinomial)
    az.loo(trace)                   # Reasonable elpd, no high Pareto k?
    az.plot_khat(loo_result)        # Identify problematic observations
    ```

    ### Step 5: Compare Models (if applicable)
    ```python
    az.compare({"model_a": trace_a, "model_b": trace_b})
    az.plot_compare(comparison)
    ```
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Exercise: Coal Mining Change Point

    The British coal mining disasters dataset records the number of coal mining disasters per year in the UK from 1851 to 1962. It's widely believed that safety regulations introduced in the late 19th century led to a reduction in the disaster rate at some unknown **change point**.

    Below is a change-point model with **deliberate problems**. Your task is to apply the full evaluation workflow:

    1. Run the model and identify all problems using diagnostics.
    2. Fix the prior scaling so the sampler converges.
    3. Evaluate model fit with posterior predictive checks.
    4. Build an improved **two-changepoint** model and compare against the single-changepoint model using LOO.
    """)
    return


@app.cell
def _(np, plt):
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

    def make_disasters_plot():
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.vlines(years, 0, disasters_array, lw=4)
        ax.set(xlabel="Year", ylabel="Disasters", xlim=(1850, 1963), ylim=(0, None))
        return fig

    make_disasters_plot()
    return disasters_array, years


@app.cell
def _(RANDOM_SEED, as_model, disasters_array, pm, years):
    @as_model()
    def _capstone():
        early_lambda = pm.Exponential("early_lambda", lam=100)
        late_lambda = pm.Exponential("late_lambda", lam=100)
        change_point = pm.Uniform("change_point", lower=1851, upper=1962)
        lam = pm.math.switch(years > change_point, late_lambda, early_lambda)
        pm.Poisson("disasters", mu=lam, observed=disasters_array)

    with _capstone():
        capstone_trace = pm.sample(1000, random_seed=RANDOM_SEED)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Hint": mo.md(r"""
        Disaster counts range from 0–6 per year. What posterior rate is `Exponential(lam=100)` implying? (The mean of `Exponential(lam)` is `1/lam`, so the prior is squeezing the rate toward 0.01 — far below the observed scale.) Pick `lam` values that place the prior mean near the observed rate (roughly 1–3 disasters/year).
        """)
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Solution": mo.md(r"""
        ```python
        # Step 1+2: fix the priors and re-sample
        with pm.Model() as capstone_fixed:
            early_lambda = pm.Exponential("early_lambda", lam=0.3)
            late_lambda = pm.Exponential("late_lambda", lam=0.5)
            change_point = pm.Uniform("change_point", lower=1851, upper=1962)

            lam = pm.math.switch(years > change_point, late_lambda, early_lambda)
            pm.Poisson("disasters", mu=lam, observed=disasters_array)

            capstone_trace_fixed = pm.sample(1000, random_seed=RANDOM_SEED)

        az.summary(capstone_trace_fixed)

        # Step 3: posterior predictive check
        with capstone_fixed:
            pm.sample_posterior_predictive(
                capstone_trace_fixed, extend_inferencedata=True, random_seed=RANDOM_SEED
            )
        az.plot_ppc_dist(capstone_trace_fixed)
        # Rootogram is the canonical count-data PPC — bar heights compare
        # expected vs observed counts on a sqrt scale. For a well-fit Poisson
        # model the bars stay close to zero deviation.
        az.plot_ppc_rootogram(capstone_trace_fixed)

        # Step 4: two-changepoint model + LOO comparison
        with pm.Model() as two_cp_model:
            early_lambda_2 = pm.Exponential("early_lambda", lam=0.3)
            mid_lambda_2 = pm.Exponential("mid_lambda", lam=0.3)
            late_lambda_2 = pm.Exponential("late_lambda", lam=0.5)
            cp1 = pm.Uniform("change_point_1", lower=1851, upper=1962)
            cp2 = pm.Uniform("change_point_2", lower=1851, upper=1962)

            lam_2 = np.where(
                years < cp1,
                early_lambda_2,
                np.where(years < cp2, mid_lambda_2, late_lambda_2),
            )
            pm.Poisson("disasters", mu=lam_2, observed=disasters_array)

            two_cp_trace = pm.sample(1000, random_seed=RANDOM_SEED)

        with capstone_fixed:
            pm.compute_log_likelihood(capstone_trace_fixed)
        with two_cp_model:
            pm.compute_log_likelihood(two_cp_trace)

        az.compare(
            {"single-changepoint": capstone_trace_fixed, "two-changepoint": two_cp_trace}
        )
        ```

        After fixing the priors, the sampler converges (R-hat ≈ 1.00, healthy ESS). The PPC shows the single-changepoint model captures the bulk of the disaster pattern. LOO will typically prefer the two-changepoint model by a modest margin, reflecting the gradual transition in disaster rates.
        """)
        }
    )
    return


if __name__ == "__main__":
    app.run()
