import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


with app.setup:
    import marimo as mo
    import inspect
    import pymc as pm
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import polars as pl
    import warnings
    from pathlib import Path
    import scipy
    from scipy.stats import norm
    import base64
    import arviz as az

    az.style.use("arviz-variat")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 3),
            "figure.dpi": 90,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.2,
        }
    )

    def fig_kwargs(cols=1, rows=1):
        """Compute reasonable figure_kwargs for arviz plots."""
        w = min(max(7, 3.2 * cols), 11)
        h = 2.4 * rows
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

    # Session 3.1: MCMC and Convergence Diagnostics
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 0. Warm-up: two `idata` objects in the wild

    Before we build anything, here are two `idata` objects handed to you by a colleague. They came from two different models fit to the penguin data. Your job, working in pairs: figure out **which one you would trust**, and explain why — using only the objects themselves.

    You have not yet been told what to look for. That is the point. Explore.
    """)
    return


@app.cell(hide_code=True)
def _():
    idata_a = az.from_netcdf(data_path / "s3a_idata_a.nc")
    idata_b = az.from_netcdf(data_path / "s3a_idata_b.nc")

    # The two idatas were pre-sampled with the code below and saved to disk
    # so students can start the exercise immediately. Uncomment to rebuild.
    #
    # species_idx = (
    #     penguins.get_column("species").cast(pl.Categorical).to_physical().to_numpy()
    # )
    # n_species = int(penguins.get_column("species").n_unique())
    #
    # with pm.Model() as model_a:
    #     alpha = pm.Normal("alpha", mu=4, sigma=2, shape=n_species)
    #     beta = pm.Normal("beta", mu=0, sigma=2)
    #     sigma = pm.HalfNormal("sigma", sigma=2)
    #     mu = alpha[species_idx] + beta * flipper_length_std
    #     pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)
    #     idata_a = pm.sample(random_seed=RANDOM_SEED)
    # idata_a.to_netcdf(data_path / "s3a_idata_a.nc")
    #
    # with pm.Model() as model_b:
    #     mu_alpha = pm.Normal("mu_alpha", mu=4, sigma=2)
    #     tau_alpha = pm.HalfNormal("tau_alpha", sigma=1)
    #     alpha = pm.Normal(
    #         "alpha", mu=mu_alpha, sigma=tau_alpha, shape=len(body_mass_kg)
    #     )
    #     beta = pm.Normal("beta", mu=0, sigma=2)
    #     sigma = pm.HalfNormal("sigma", sigma=0.5)
    #     mu = alpha + beta * flipper_length_std
    #     pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)
    #     idata_b = pm.sample(random_seed=RANDOM_SEED, target_accept=0.8)
    # idata_b.to_netcdf(data_path / "s3a_idata_b.nc")
    return idata_a, idata_b


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Stage 1 — Meet the objects

    Get your bearings. For each prompt, write a one-liner in the empty cell below and look at the output:

    1. Type `idata_a` and `idata_b` on their own lines. What groups does each one contain? Are they the same?
    2. How many `chain`s and `draw`s are in each `posterior`?
    3. What are the parameter names in each? Which parameters do they share?
    4. Find where the observed penguin mass lives — the same penguin measurements
       you met in Session 1.2, now used in a regression of body mass on flipper length.
    5. Look at `idata_a.sample_stats` and `idata_b.sample_stats`. List three variables you see in each.
    """)
    return


@app.cell
def _(idata_a, idata_b):
    # your code here
    type(idata_a), type(idata_b)
    return


@app.cell(hide_code=True)
def _(idata_a, idata_b):
    def solution_warmup_stage1():
        groups_a, groups_b = list(idata_a.children), list(idata_b.children)
        post_a, post_b = idata_a["posterior"], idata_b["posterior"]
        shared = sorted(set(post_a.data_vars) & set(post_b.data_vars))
        return mo.md(
            f"""
- **Groups**: `idata_a` has `{groups_a}`, `idata_b` has `{groups_b}` — the same four.
- **Chains × draws**: `idata_a`: {post_a.sizes["chain"]} × {post_a.sizes["draw"]}; `idata_b`: {post_b.sizes["chain"]} × {post_b.sizes["draw"]}.
- **Parameters**: `idata_a` has `{sorted(post_a.data_vars)}`; `idata_b` has `{sorted(post_b.data_vars)}`; shared: `{shared}`.
- **Observed data**: `idata_a["observed_data"]["mass"]` (same location in `idata_b`).
- **`sample_stats`** holds per-draw sampler information such as `diverging`, `energy`, and `depth`.
"""
        )

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_warmup_stage1)}\n```"
                    ),
                    mo.lazy(solution_warmup_stage1, show_loading_indicator=True),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Stage 2 — Ask the summary

    1. Run `az.summary(...)` on each idata. Compare `ess_bulk`, `ess_tail`, and `r_hat` side by side. Which parameters look problematic in `idata_b`?
    2. `sample_stats` contains a boolean flag `diverging` — draws where the sampler broke down. Check both runs: how many divergences does each have? (`.sum()` on the DataArray works.) Careful: a clean divergence count does **not** mean a clean run — as you're about to see, the trouble here shows up elsewhere in the summary.
    """)
    return


@app.cell
def _():
    # your code here
    return


@app.cell(hide_code=True)
def _(idata_a, idata_b):
    def solution_warmup_stage2():
        summary_a = az.summary(idata_a)
        summary_b = az.summary(
            idata_b, var_names=["mu_alpha", "tau_alpha", "beta", "sigma"]
        )
        div_a = int(idata_a["sample_stats"]["diverging"].sum())
        div_b = int(idata_b["sample_stats"]["diverging"].sum())
        return mo.vstack(
            [
                mo.md("**`idata_a` summary:**"),
                summary_a,
                mo.md("**`idata_b` summary (top-level parameters):**"),
                summary_b,
                mo.md(
                    f"Divergence flags: **{div_a}** in `idata_a`, **{div_b}** in `idata_b`. "
                    "Here the trouble shows up in the summary instead: `tau_alpha` and `sigma` "
                    "in `idata_b` have `ess_bulk` around 10 and `r_hat` around 1.3 — "
                    "the chains never mixed."
                ),
            ]
        )

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_warmup_stage2)}\n```"
                    ),
                    mo.lazy(solution_warmup_stage2, show_loading_indicator=True),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Stage 3 — Look at it

    Call `az.plot_trace_dist(...)` on each idata. Minimal usage: `az.plot_trace_dist(idata_a)` (add `var_names=[...]` to focus it). A few things to know before you start:

    - The two models have **different parameters**, so you can't use the same `var_names` for both — look at each `posterior` group to see what's there.
    - `idata_b` has *hundreds* of `alpha` entries (one per penguin). Restrict `var_names` to the top-level parameters or the plot will be unreadable.

    Then:

    1. What visual difference do you see between the two traces? Look at how each chain moves over iterations and whether the chains agree.
    2. Based only on what you have seen so far, write one sentence: *which `idata` would you trust, and what specifically convinced you?* Keep that sentence — over the next hour we will name every piece of evidence you used.
    """)
    return


@app.cell
def _():
    # your code here — start with idata_a
    return


@app.cell
def _():
    # ...then idata_b. It has hundreds of `alpha` entries, so restrict to the
    # top-level parameters with var_names=["mu_alpha", "tau_alpha", "beta", "sigma"]
    return


@app.cell(hide_code=True)
def _(idata_a, idata_b):
    def solution_warmup_stage3():
        plot_a = az.plot_trace_dist(idata_a)
        plot_b = az.plot_trace_dist(
            idata_b, var_names=["mu_alpha", "tau_alpha", "beta", "sigma"]
        )
        return mo.vstack(
            [
                mo.md(
                    "**`idata_a`** — every chain wanders freely over the same region "
                    "and the per-chain densities overlap (the 'fuzzy caterpillar'):"
                ),
                plot_a,
                mo.md(
                    "**`idata_b`** — the `tau_alpha` and `sigma` chains disagree and "
                    "get stuck for long stretches; each chain explores a different region:"
                ),
                plot_b,
                mo.md(
                    "**Verdict:** trust `idata_a`. The evidence: `ess_bulk` in the "
                    "hundreds-to-thousands with `r_hat` ≈ 1 for every parameter, and "
                    "trace plots where all four chains agree."
                ),
            ]
        )

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_warmup_stage3)}\n```"
                    ),
                    mo.lazy(solution_warmup_stage3, show_loading_indicator=True),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The rest of this session names the things you just noticed: what `posterior` and `sample_stats` actually hold, what R-hat and ESS measure, what a divergence is, and how trace, rank, and energy plots turn all of this into pictures. When something feels familiar, it is because you already saw it in `idata_b`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. What Does `pm.sample()` Give You?

    You've been calling `pm.sample()` since Session 1. You know how to specify models and check traceplots. But when something goes wrong — and it will — your ability to fix it depends on understanding what the sampler is doing and what the diagnostics are telling you has occurred, relative to what was expected.

    We'll organize around three questions:

    1. **What does `pm.sample()` give you**, and how do you read the output?
    2. **Did the sampler work?** How do you diagnose and fix problems?
    3. **Does the model fit the data?**

    Along the way, we'll build up your understanding of how the sampler works — not to implement it from scratch, but so you can reason about what's happening when diagnostics flag a problem.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### A well-specified model

    Let's start with a simple model we know works — predicting penguin body mass from flipper length. Before building a model, we should always look at the data.
    """)
    return


@app.cell
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
    return body_mass, body_mass_kg, flipper_length, flipper_length_std


@app.cell
def _(body_mass, flipper_length):
    _fig, _ax = plt.subplots(**fig_kwargs())
    _ax.scatter(flipper_length, body_mass / 1000, alpha=0.5, s=20)
    _ax.set_xlabel("Flipper length (mm)")
    _ax.set_ylabel("Body mass (kg)")
    _ax.set_title("Penguin body mass vs. flipper length")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    There's a clear positive linear relationship — penguins with longer flippers tend to be heavier. There's also meaningful spread around the trend, which our model needs to capture.

    Before specifying the model, two practical choices:

    - **Standardize the predictor** (flipper length): centering and scaling makes the intercept interpretable as the mean body mass at an average flipper length, and puts the slope on a "per standard deviation" scale. This also makes it easier to choose sensible priors.
    - **Work in kilograms**: the raw data is in grams (values around 3000–6000), which would require priors on a large scale. Dividing by 1000 gives values around 3–6 kg, which are easier to reason about.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior choices

    With standardized flipper length and mass in kg, our priors become intuitive:

    - **`alpha ~ Normal(4, 2)`**: The intercept is the expected mass at average flipper length. Penguins weigh roughly 3–6 kg, so a prior centered at 4 kg with SD of 2 covers the plausible range generously.
    - **`beta ~ Normal(0, 2)`**: The slope (effect of a 1-SD change in flipper length). A prior centered at zero expresses no prior directional preference; SD of 2 allows for substantial effects.
    - **`sigma ~ HalfNormal(2)`**: The residual standard deviation. HalfNormal(2) puts most prior mass below ~4 kg of residual spread — generous for a variable that ranges over ~3 kg total.
    """)
    return


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
    return (baseline_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior predictive check

    Before fitting the model, we can check whether our priors generate plausible data. A **prior predictive check** draws parameter values from the priors and simulates datasets — if these simulated datasets look nothing like real penguin data, our priors are poorly calibrated.
    """)
    return


@app.cell
def _(baseline_model):
    with baseline_model:
        prior_pred = pm.sample_prior_predictive(random_seed=RANDOM_SEED)

    az.plot_ppc_dist(
        prior_pred,
        group="prior_predictive",
        num_samples=100,
        figure_kwargs=fig_kwargs(),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The prior predictive distribution covers a wide range of body masses — some implausible (negative masses, masses above 10 kg) but with most of the density in a reasonable range. This tells us our priors are weakly informative: they don't force the model toward any particular answer, but they don't generate data that are off by orders of magnitude. That's exactly what we want for a first pass.

    Now let's fit the model.
    """)
    return


@app.cell
def _(baseline_model):
    with baseline_model:
        nuts_idata = pm.sample(random_seed=RANDOM_SEED)
    return (nuts_idata,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The InferenceData object

    The result of `pm.sample()` is an `InferenceData` object — a container built on top of xarray that holds everything about the sampling run, organized into groups like `posterior`, `sample_stats`, and `observed_data`.
    """)
    return


@app.cell
def _(nuts_idata):
    nuts_idata
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `posterior` group contains the actual parameter draws — this is what you'll use for inference. It's an xarray Dataset with dimensions `(chain, draw)` for each parameter.
    """)
    return


@app.cell
def _(nuts_idata):
    nuts_idata.posterior
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `sample_stats` group contains per-draw sampler diagnostics: step sizes, divergence flags, energy values, tree depth, and more. This is what convergence diagnostics examine.
    """)
    return


@app.cell
def _(nuts_idata):
    nuts_idata.sample_stats
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `observed_data` group stores the data you conditioned on. ArviZ uses this for posterior predictive checks and leave-one-out cross-validation.
    """)
    return


@app.cell
def _(nuts_idata):
    nuts_idata.observed_data
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Summarizing the Posterior Samples

    The `az.summary()` function gives you the most important information in one table. Let's walk through each column.
    """)
    return


@app.cell
def _(nuts_idata):
    az.summary(nuts_idata)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Reading the summary table

    | Column | What it tells you |
    |--------|-------------------|
    | **mean** | Point estimate (posterior mean) |
    | **sd** | Posterior standard deviation — uncertainty in the parameter |
    | **eti89_lb** / **eti89_ub** | Bounds of the 89% Equal-Tailed Interval — equal probability (5.5%) excluded from each tail |
    | **mcse_mean** | Monte Carlo Standard Error of the mean — how much the *estimate* of the mean would change with different random draws |
    | **mcse_sd** | MCSE for the standard deviation |
    | **ess_bulk** | Effective sample size for the bulk (center) of the posterior |
    | **ess_tail** | Effective sample size for the tails (5% and 95% quantiles) |
    | **r_hat** | Split R-hat convergence diagnostic — values near 1.0 indicate convergence |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Equal-Tailed Intervals

    Two columns in the summary table deserve a closer look: `eti89_lb` and `eti89_ub`. The **Equal-Tailed Interval** (ETI) is computed by cutting off equal probability in each tail of the posterior distribution. For an 89% ETI, we exclude 5.5% from each tail.

    The ETI is the default in ArviZ because it is straightforward to interpret and compute. An alternative is the **Highest Density Interval** (HDI), which finds the *narrowest* interval containing the specified probability mass. For **skewed posteriors**, the HDI may be preferable since it always contains the most probable values — but for symmetric posteriors (like the ones here), the two are nearly identical.

    The 89% default probability follows a convention that avoids the false precision of "95%" while providing a useful credible interval. It also has the practical benefit of lower variability in summary statistics compared to wider intervals.

    `az.plot_dist()` shows the posterior density, making it easy to see the shape of the distribution at a glance.
    """)
    return


@app.cell(hide_code=True)
def _(nuts_idata):
    az.plot_dist(
        nuts_idata,
        var_names=["alpha", "beta", "sigma"],
        figure_kwargs=fig_kwargs(cols=3),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can also extract ETI values programmatically:
    """)
    return


@app.cell
def _(nuts_idata):
    az.eti(nuts_idata, prob=0.89)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## 2. Did the Sampler Work?

    Having seen what the output looks like, let's now ask: can we trust it? The summary table showed healthy-looking numbers, but to understand *why* they're healthy — and to recognize when they're not — we need to understand what the sampler is doing.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How the Sampler Works: From Random Walks to NUTS

    When you call `pm.sample()`, PyMC doesn't draw independent samples from the posterior — it can't, because the posterior is a complex, high-dimensional distribution we only know up to a normalizing constant.

    Instead, it constructs a **Markov chain**: a sequence of samples where each draw depends on the previous one, particularly designed so that after enough steps, the samples approximate draws from the posterior. At least, this is what theory guarantees.

    This is the core idea behind **Markov chain Monte Carlo (MCMC)**. To understand the diagnostics we'll use to evaluate sampler output, let's take a look at a specific MCMC algorithm.
    """)
    return


@app.cell(hide_code=True)
def _():
    _metro_path = Path(__file__).parent / "images" / "Metropolis.png"
    if _metro_path.exists():
        _metro_b64 = base64.b64encode(_metro_path.read_bytes()).decode()
        _metro_img = (
            f'<img src="data:image/png;base64,{_metro_b64}" style="max-width:100%;">'
        )
    else:
        _metro_img = ""

    mo.md(f"""
    #### Random Walk Metropolis

    The simplest MCMC algorithm works like this:

    1. Start at some position in parameter space
    2. **Propose** a small random step from the current position
    3. **Accept** the proposal if it lands in a region of higher (or comparable) posterior density; otherwise, **reject** it and stay put
    4. Repeat

    {_metro_img}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Metropolis from scratch, in 1D

    Before watching an animation of it, let's build it. We'll target a **Beta(3, 2)** distribution — a nice 1D example with bounded support on $[0, 1]$ and a closed-form pdf we can plot for comparison. Here's the target:
    """)
    return


@app.cell(hide_code=True)
def _():
    def pdf(x, a=3, b=2):
        return scipy.stats.beta.pdf(x, a, b)

    x_range = np.linspace(-0.2, 1.2, 1000)
    x_range_pdf = pdf(x_range)

    _fig, _ax = plt.subplots(figsize=(7, 2.5))
    _ax.plot(x_range, x_range_pdf, color="k")
    _ax.set_xlabel("x")
    _ax.set_ylabel("pdf(x)")
    _ax.set_title("Target: Beta(3, 2)")
    _fig
    return pdf, x_range, x_range_pdf


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The four-step algorithm above translates directly into code. The only subtlety is the accept/reject rule: if the proposed density is *higher* than the current one, always accept; otherwise accept with probability `p_new / p_old`.
    """)
    return


@app.cell
def _():
    def metropolis(pdf, x0, rng=None, n_draws=1000, step_size=0.5):
        rng = np.random.default_rng(rng)

        x_old = x0
        p_old = pdf(x_old)
        assert p_old > 0

        draws = []
        accepted_draws = 0
        p_0_draws = 0
        for _ in range(n_draws):
            x_new = rng.normal(scale=step_size) + x_old
            p_new = pdf(x_new)
            if p_new == 0:
                p_0_draws += 1

            p_ratio = p_new / p_old
            if p_ratio > 1:
                accept = True
            else:
                p_accept = p_ratio
                accept = rng.uniform(0, 1) <= p_accept

            if accept:
                accepted_draws += 1
                x_old = x_new
                p_old = p_new

            draws.append(x_old)

        print(f"Accepted draws: {accepted_draws / n_draws:.2%}")
        print(f"P == 0 draws:   {p_0_draws / n_draws:.2%}")
        return draws

    return (metropolis,)


@app.cell
def _(metropolis, pdf):
    samples = metropolis(pdf, x0=0.5, n_draws=10_000, step_size=0.5)
    return (samples,)


@app.cell
def _(samples, x_range, x_range_pdf):
    _fig, _ax = plt.subplots(figsize=(7, 2.5))
    _ax.hist(samples, ec="k", bins=30, density=True)
    _ax.plot(x_range, x_range_pdf, color="k", lw=2)
    _ax.set_xlabel("x")
    _ax.set_ylabel("density")
    _ax.set_title("10,000 Metropolis draws vs. true Beta(3, 2) density")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The histogram of draws lines up with the target density. Forty-some lines of code, and we have a working sampler.

    To build intuition for what's happening draw by draw, here's a step-by-step view of the first 12 proposals. Black dots are the current state; green arrows are accepted proposals, red crosses are rejected ones. Move the slider to step through.
    """)
    return


@app.cell(hide_code=True)
def _():
    STEP_STYLES = {
        "accepted": ("green", "-"),
        "rejected": ("red", "-"),
        "rejected_adjustment": ("purple", "-"),
        "rejected_zero": ("red", "--"),
    }

    def walk_trace(pdf_fn, x0, n_steps=12, step_size=0.3, seed=0, pdf_unadjusted=None):
        rng = np.random.default_rng(seed)
        x_old = x0
        p_old = pdf_fn(x_old)
        p_old_unadj = pdf_unadjusted(x_old) if pdf_unadjusted is not None else None
        steps = []
        draws = []
        for _ in range(n_steps):
            x_new = x_old + rng.normal(scale=step_size)
            p_new = pdf_fn(x_new)
            if p_new == 0:
                status = "rejected_zero"
                accept = False
            else:
                ratio = p_new / p_old
                u = rng.uniform()
                accept = ratio >= 1 or u <= ratio
                if accept:
                    status = "accepted"
                elif pdf_unadjusted is not None:
                    p_new_unadj = pdf_unadjusted(x_new)
                    ratio_unadj = p_new_unadj / p_old_unadj
                    if ratio_unadj >= 1 or u <= ratio_unadj:
                        status = "rejected_adjustment"
                    else:
                        status = "rejected"
                else:
                    status = "rejected"
            steps.append((x_old, x_new, status))
            if accept:
                x_old, p_old = x_new, p_new
                if pdf_unadjusted is not None:
                    p_old_unadj = pdf_unadjusted(x_old)
            draws.append(x_old)
        return steps, draws

    def draw_step(ax, i, a, b, status):
        c, ls = STEP_STYLES[status]
        arrowstyle = "->" if status == "accepted" else "-"
        ax.annotate(
            "",
            xy=(b, i),
            xytext=(a, i),
            arrowprops=dict(arrowstyle=arrowstyle, color=c, lw=1.2, linestyle=ls),
        )
        ax.scatter([a], [i], color="black", s=15, zorder=3)
        if status != "accepted":
            ax.scatter([b], [i], marker="x", color=c, s=40, zorder=3)

    return STEP_STYLES, draw_step, walk_trace


@app.cell(hide_code=True)
def _(STEP_STYLES, draw_step, x_range, x_range_pdf):
    def plot_logit_walk(walk, k):
        steps_u, steps_nat, draws_nat = walk

        fig, (ax_logit, ax_nat, ax_pdf) = plt.subplots(
            3,
            1,
            figsize=(7, 6),
            gridspec_kw={"height_ratios": [2, 2, 1]},
        )

        _logit_xs = [v for a, b, _ in steps_u for v in (a, b)]
        logit_x_min = min(_logit_xs) - 0.5
        logit_x_max = max(_logit_xs) + 0.5
        ax_logit.set_xlim(logit_x_min, logit_x_max)
        ax_logit.set_ylim(len(steps_u) - 0.5, -0.5)
        ax_logit.set_xlabel("logit(x)")
        ax_logit.set_ylabel("step")
        ax_logit.set_title("Logit (unconstrained) scale")
        present = {s for _, _, s in steps_u}
        for label, (c, ls) in STEP_STYLES.items():
            if label in present:
                ax_logit.plot([], [], color=c, linestyle=ls, label=label)
        ax_logit.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
            handlelength=1.5,
            borderpad=0.3,
            labelspacing=0.3,
        )

        ax_nat.set_xlim(x_range.min(), x_range.max())
        ax_nat.set_ylim(len(steps_u) - 0.5, -0.5)
        ax_nat.set_xlabel("x (natural scale)")
        ax_nat.set_ylabel("step")
        ax_nat.set_title("Natural scale")

        x0_logit = steps_u[0][0]
        x0_nat = steps_nat[0][0]
        ax_logit.scatter([x0_logit], [0], color="black", s=15, zorder=3)
        ax_nat.scatter([x0_nat], [0], color="black", s=15, zorder=3)
        for i in range(k):
            a_l, b_l, status = steps_u[i]
            a_n, b_n, _ = steps_nat[i]
            draw_step(ax_logit, i, a_l, b_l, status)
            draw_step(ax_nat, i, a_n, b_n, status)

        ax_pdf.plot(x_range, x_range_pdf, color="k", lw=2)
        rug_x = [x0_nat] + list(draws_nat[:k])
        ax_pdf.scatter(rug_x, np.zeros(len(rug_x)), marker="|", s=200, color="k")
        ax_pdf.set_xlabel("x")
        ax_pdf.set_ylabel("pdf(x)")
        ax_pdf.set_xlim(x_range.min(), x_range.max())
        ax_nat.sharex(ax_pdf)
        fig.tight_layout()
        return fig

    return (plot_logit_walk,)


@app.cell(hide_code=True)
def _(pdf, walk_trace):
    nat_walk = walk_trace(pdf, x0=0.5, n_steps=12, step_size=0.5, seed=1)
    return (nat_walk,)


@app.cell(hide_code=True)
def _(nat_walk):
    nat_k = mo.ui.slider(start=0, stop=len(nat_walk[0]), step=1, value=0, label="steps")
    nat_k
    return (nat_k,)


@app.cell(hide_code=True)
def _(STEP_STYLES, draw_step, nat_k, nat_walk, x_range, x_range_pdf):
    def _plot_natural_walk():
        steps, draws = nat_walk
        k = nat_k.value

        fig, (ax_steps, ax_pdf) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(7, 4.5),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        ax_steps.set_xlim(x_range.min(), x_range.max())
        ax_steps.set_ylim(len(steps) - 0.5, -0.5)
        ax_steps.set_ylabel("step")
        present = {s for _, _, s in steps}
        for label, (c, ls) in STEP_STYLES.items():
            if label in present:
                ax_steps.plot([], [], color=c, linestyle=ls, label=label)
        ax_steps.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
            handlelength=1.5,
            borderpad=0.3,
            labelspacing=0.3,
        )

        x0 = steps[0][0]
        ax_steps.scatter([x0], [0], color="black", s=15, zorder=3)
        for i in range(k):
            a, b, status = steps[i]
            draw_step(ax_steps, i, a, b, status)

        ax_pdf.plot(x_range, x_range_pdf, color="k", lw=2)
        rug_x = [x0] + list(draws[:k])
        ax_pdf.scatter(rug_x, np.zeros(len(rug_x)), marker="|", s=200, color="k")
        ax_pdf.set_xlabel("x")
        ax_pdf.set_ylabel("pdf(x)")
        return fig

    _plot_natural_walk()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Bounded parameters and the unconstrained space

    A Beta(3, 2) is supported on $[0, 1]$ — the pdf is zero outside. Our proposal (a Gaussian centred on the current state) regularly suggests values like $x = -0.1$ or $x = 1.2$, which get rejected outright. This is wasted work, and it gets much worse for distributions like a `HalfNormal` or a `Dirichlet` where most Gaussian proposals land out of bounds.

    PyMC solves this by sampling in **unconstrained space**. For a variable bounded on $[0, 1]$, it applies the logit transform $y = \log(x / (1 - x))$, samples in $y \in (-\infty, \infty)$ where any Gaussian proposal is valid, and transforms draws back. Let's try that naively:
    """)
    return


@app.cell
def _(metropolis, pdf):
    def pdf_unconstrained(logit_x, a=3, b=2):
        x = scipy.special.expit(logit_x)
        return pdf(x, a=a, b=b)

    samples_logit_x = metropolis(
        pdf_unconstrained, x0=0.0, n_draws=10_000, step_size=0.5
    )
    samples2 = scipy.special.expit(samples_logit_x)
    return (samples2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's the same step-by-step walk, this time on a *biased* version of the unconstrained sampler. The top panel shows steps in logit space; the middle panel shows the same steps after transforming back to the natural scale.
    """)
    return


@app.cell(hide_code=True)
def _(pdf, walk_trace):
    def _biased_walk():
        def pdf_unconstrained_biased(logit_x, a=3, b=2):
            x = scipy.special.expit(logit_x)
            return pdf(x, a=a, b=b)

        steps_u, draws_u = walk_trace(
            pdf_unconstrained_biased,
            x0=0.0,
            n_steps=12,
            step_size=1.5,
            seed=5,
        )
        steps_nat = [
            (scipy.special.expit(a), scipy.special.expit(b), s) for a, b, s in steps_u
        ]
        draws_nat = scipy.special.expit(np.array(draws_u))
        return steps_u, steps_nat, draws_nat

    biased_walk = _biased_walk()
    return (biased_walk,)


@app.cell(hide_code=True)
def _(biased_walk):
    biased_k = mo.ui.slider(
        start=0, stop=len(biased_walk[0]), step=1, value=0, label="steps"
    )
    biased_k
    return (biased_k,)


@app.cell(hide_code=True)
def _(biased_k, biased_walk, plot_logit_walk):
    plot_logit_walk(biased_walk, biased_k.value)
    return


@app.cell
def _(samples2, x_range, x_range_pdf):
    _fig, _ax = plt.subplots(figsize=(7, 2.5))
    _ax.hist(samples2, ec="k", bins=30, density=True)
    _ax.plot(x_range, x_range_pdf, color="k", lw=2)
    _ax.set_xlabel("x")
    _ax.set_ylabel("density")
    _ax.set_title("Naive unconstrained draws — biased toward the boundaries")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The histogram is **skewed toward the boundaries** compared to the true Beta(3, 2) — too much mass piled up near $x = 1$, not enough in the middle. Something has gone wrong.

    #### Why the bias — and the change of variables

    Equal-sized steps in logit space are *not* equal-sized steps in natural space. Near $y = 0$ (i.e. $x = 0.5$) the logit map is nearly linear — $dx/dy = x(1-x) = 0.25$, its maximum — so a step of $\Delta y = 0.1$ moves $x$ by about $0.025$. Near the boundaries (say $y = 4$, $x \approx 0.982$) the same $\Delta y = 0.1$ only moves $x$ by about $0.0018$ — fifteen times less.

    The logit map stretches the edges of $[0, 1]$ out to infinity, so a huge chunk of $y$-space maps to a thin sliver of $x$-space near the boundaries. The naive sampler doesn't know this: it treats $\text{Beta}(\text{expit}(y); 3, 2)$ as if it were a density in $y$ and samples from it directly. When we transform the resulting draws back via $x = \text{expit}(y)$, all the draws made out in the tails of $y$-space cram into those tiny slivers near $x = 0$ and $x = 1$ — so those regions end up *over-represented* in the histogram, and the middle gets starved.
    """)
    return


@app.cell(hide_code=True)
def _(x_range):
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(x_range, scipy.special.logit(x_range), color="k")
    for _transformed_sample in np.linspace(-6, 6, 12):
        _natural_sample = scipy.special.expit(_transformed_sample)
        _ax.plot(
            [0, _natural_sample],
            [_transformed_sample] * 2,
            color="green",
            ls="--",
            lw=0.5,
        )
        _ax.plot(
            [_natural_sample] * 2,
            [0, _transformed_sample],
            color="blue",
            ls="--",
            lw=0.5,
        )
        _ax.scatter([_natural_sample], [0], color="blue", s=20)
    _ax.axhline(0, color="k")
    _ax.set_xlabel("Natural space")
    _ax.set_ylabel("Logit space")
    _ax.plot(
        [],
        [],
        color="green",
        ls="--",
        label="Evenly spaced in\nlogit space",
    )
    _ax.plot(
        [],
        [],
        color="blue",
        ls="--",
        label="Where they land in\nnatural space",
    )
    _ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The fix is the **change of variables formula**: when we sample in $y = \mathrm{logit}(x)$ but want draws from a density defined on $x$, we have to multiply the target by the Jacobian $\left|\frac{dx}{dy}\right|$. For the logit transform, $\frac{dx}{dy} = x(1-x)$, so the adjusted density is

    $$
    p_Y(y) = p_X(x) \cdot x(1 - x)
    $$

    With that correction the sampler is unbiased again:
    """)
    return


@app.cell
def _(metropolis, pdf):
    def pdf_unconstrained_adjusted(logit_x, a=3, b=2):
        x = scipy.special.expit(logit_x)
        change_of_vars_adjustment = x * (1 - x)
        return pdf(x, a=a, b=b) * change_of_vars_adjustment

    samples_logit_x_adjusted = metropolis(
        pdf_unconstrained_adjusted, x0=0.0, n_draws=10_000, step_size=0.5
    )
    samples3 = scipy.special.expit(samples_logit_x_adjusted)
    return (samples3,)


@app.cell(hide_code=True)
def _(pdf, walk_trace):
    def _adjusted_walk():
        def pdf_unconstrained_adjusted_local(logit_x, a=3, b=2):
            x = scipy.special.expit(logit_x)
            return pdf(x, a=a, b=b) * x * (1 - x)

        def pdf_unconstrained_biased(logit_x, a=3, b=2):
            x = scipy.special.expit(logit_x)
            return pdf(x, a=a, b=b)

        steps_u, draws_u = walk_trace(
            pdf_unconstrained_adjusted_local,
            x0=0.0,
            n_steps=12,
            step_size=1.5,
            seed=5,
            pdf_unadjusted=pdf_unconstrained_biased,
        )
        steps_nat = [
            (scipy.special.expit(a), scipy.special.expit(b), s) for a, b, s in steps_u
        ]
        draws_nat = scipy.special.expit(np.array(draws_u))
        return steps_u, steps_nat, draws_nat

    adjusted_walk = _adjusted_walk()
    return (adjusted_walk,)


@app.cell(hide_code=True)
def _(adjusted_walk):
    adjusted_k = mo.ui.slider(
        start=0, stop=len(adjusted_walk[0]), step=1, value=0, label="steps"
    )
    adjusted_k
    return (adjusted_k,)


@app.cell(hide_code=True)
def _(adjusted_k, adjusted_walk, plot_logit_walk):
    plot_logit_walk(adjusted_walk, adjusted_k.value)
    return


@app.cell
def _(samples3, x_range, x_range_pdf):
    _fig, _ax = plt.subplots(figsize=(7, 2.5))
    _ax.hist(samples3, ec="k", bins=30, density=True)
    _ax.plot(x_range, x_range_pdf, color="k", lw=2)
    _ax.set_xlabel("x")
    _ax.set_ylabel("density")
    _ax.set_title("Unconstrained draws with Jacobian adjustment — unbiased")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Purple arrows** in the step-by-step plot above mark steps that *would* have been accepted by the naive sampler but were rejected by the adjusted one — that's the Jacobian doing its job, pushing the sampler away from regions that the unconstrained proposal over-represents.

    This is exactly what PyMC does under the hood for every bounded variable in your model: it transforms to an unconstrained space, samples there, and tacks the log-Jacobian onto the log-posterior. You normally never see it — but now you know what's happening when the sampler trace lives on some transformed scale.

    ---

    #### Back to two dimensions

    The 1D walk shows the algorithm clearly, but it hides the hard part: in high dimensions and with correlated posteriors, the step size is a tightrope between "never accept anything" and "never go anywhere". Let's rerun Metropolis on a slightly harder target — a **two-parameter posterior**: a bivariate normal over $(x_1, x_2)$ with correlation $\rho = 0.9$. The grey ellipses in the animation below are its density contours — the "ridge" where most of the probability mass lives.

    The left panel shows the sampler moving through parameter space —
    <span style="color:#2ca02c">green</span> lines are accepted proposals,
    <span style="color:#d62728">red</span> lines are rejected ones.

    Watch how the **trace** (center) and **marginal distribution** (right)
    build up sample by sample. By the end, the bottom panels look just like what `az.plot_trace_dist()` produces.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Correlated bivariate normal target (rho=0.9) for MCMC demos
    _demo_rho = 0.9
    _demo_cov = np.array([[1.0, _demo_rho], [_demo_rho, 1.0]])
    demo_cov_inv = np.linalg.inv(_demo_cov)

    def demo_logp(x):
        return -0.5 * x @ demo_cov_inv @ x

    def demo_grad_logp(x):
        return -demo_cov_inv @ x

    def demo_metropolis(n_steps, step_size, rng):
        """Random walk Metropolis-Hastings."""
        pos = np.zeros(2)
        samples = np.empty((n_steps + 1, 2))
        samples[0] = pos
        proposals = np.empty((n_steps, 2))
        accepted = np.empty(n_steps, dtype=bool)
        for i in range(n_steps):
            prop = pos + rng.normal(size=2) * step_size
            proposals[i] = prop
            if np.log(rng.random()) < demo_logp(prop) - demo_logp(pos):
                pos = prop
                accepted[i] = True
            else:
                accepted[i] = False
            samples[i + 1] = pos
        return samples, proposals, accepted

    def demo_hmc(n_steps, step_size, n_leapfrog, rng):
        """Hamiltonian Monte Carlo with leapfrog integration."""
        pos = np.zeros(2)
        samples = np.empty((n_steps + 1, 2))
        samples[0] = pos
        trajectories = []
        accepted = np.empty(n_steps, dtype=bool)
        for i in range(n_steps):
            q, p0 = pos.copy(), rng.normal(size=2)
            p = p0.copy()
            traj = [q.copy()]
            p += 0.5 * step_size * demo_grad_logp(q)
            for _ in range(n_leapfrog - 1):
                q += step_size * p
                p += step_size * demo_grad_logp(q)
                traj.append(q.copy())
            q += step_size * p
            p += 0.5 * step_size * demo_grad_logp(q)
            traj.append(q.copy())
            H_current = -demo_logp(pos) + 0.5 * p0 @ p0
            H_proposed = -demo_logp(q) + 0.5 * p @ p
            if np.log(rng.random()) < H_current - H_proposed:
                pos = q
                accepted[i] = True
            else:
                accepted[i] = False
            trajectories.append(np.array(traj))
            samples[i + 1] = pos
        return samples, trajectories, accepted

    # Pre-compute contour grid
    _demo_grid = np.linspace(-3.5, 3.5, 100)
    demo_X, demo_Y = np.meshgrid(_demo_grid, _demo_grid)
    demo_Z = np.exp(
        -0.5
        * (
            demo_cov_inv[0, 0] * demo_X**2
            + 2 * demo_cov_inv[0, 1] * demo_X * demo_Y
            + demo_cov_inv[1, 1] * demo_Y**2
        )
    )
    return demo_X, demo_Y, demo_Z, demo_hmc, demo_metropolis


@app.cell(hide_code=True)
def _(demo_metropolis):
    # Pre-compute Metropolis samples for the interactive explorer
    _rng = np.random.default_rng(42)
    metro_anim_samples, metro_anim_proposals, metro_anim_accepted = demo_metropolis(
        200, step_size=0.5, rng=_rng
    )

    # Pre-build line segments and per-step colors for fast LineCollection rendering
    _n = len(metro_anim_accepted)
    metro_anim_segments = np.empty((_n, 2, 2))
    metro_anim_segments[:, 0] = metro_anim_samples[:_n]
    metro_anim_segments[:, 1] = metro_anim_proposals
    metro_anim_seg_colors = np.where(metro_anim_accepted, "#2ca02c", "#d62728")

    # True marginal PDF for histogram overlay
    _y_grid = np.linspace(-3.5, 3.5, 200)
    metro_true_pdf_y = _y_grid
    metro_true_pdf_x = norm.pdf(_y_grid, 0, 1)
    return (
        metro_anim_accepted,
        metro_anim_proposals,
        metro_anim_samples,
        metro_anim_seg_colors,
        metro_anim_segments,
        metro_true_pdf_x,
        metro_true_pdf_y,
    )


@app.cell(hide_code=True)
def _():
    metro_step_slider = mo.ui.slider(
        1, 200, value=200, label="Steps to show", full_width=True
    )
    metro_step_slider
    return (metro_step_slider,)


@app.cell(hide_code=True)
def _(
    demo_X,
    demo_Y,
    demo_Z,
    metro_anim_accepted,
    metro_anim_proposals,
    metro_anim_samples,
    metro_anim_seg_colors,
    metro_anim_segments,
    metro_step_slider,
    metro_true_pdf_x,
    metro_true_pdf_y,
):
    def _draw_metropolis_animation(n_show):
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.5, 0.4]}
        )

        samples = metro_anim_samples
        proposals = metro_anim_proposals
        accepted = metro_anim_accepted

        # Panel 1: 2D sampling view
        ax1.contour(
            demo_X, demo_Y, demo_Z, levels=6, colors="gray", alpha=0.4, linewidths=0.8
        )
        if n_show > 0:
            ax1.add_collection(
                LineCollection(
                    metro_anim_segments[:n_show],
                    colors=metro_anim_seg_colors[:n_show],
                    linewidths=1.5,
                    alpha=0.6,
                )
            )
            ax1.scatter(
                proposals[:n_show, 0],
                proposals[:n_show, 1],
                c=metro_anim_seg_colors[:n_show],
                s=8,
                alpha=0.5,
                zorder=3,
            )
        ax1.scatter(
            samples[: n_show + 1, 0],
            samples[: n_show + 1, 1],
            color="#1f77b4",
            s=4,
            alpha=0.3,
            zorder=2,
        )
        if n_show <= len(accepted):
            ax1.scatter(
                samples[n_show, 0],
                samples[n_show, 1],
                color="#1f77b4",
                s=40,
                zorder=5,
                edgecolors="white",
                linewidths=1.5,
            )
        ax1.set_xlim(-3.5, 3.5)
        ax1.set_ylim(-3.5, 3.5)
        ax1.set_xlabel("x₁")
        ax1.set_ylabel("x₂")
        ax1.set_title("Metropolis Sampling")
        acc_rate = accepted[:n_show].sum() / n_show if n_show > 0 else 0
        ax1.text(
            0.05,
            0.95,
            f"Step {n_show}/200\nAccept: {acc_rate:.0%}",
            transform=ax1.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        # Panel 2: Trace of x₁
        ax2.plot(range(n_show + 1), samples[: n_show + 1, 0], color="#1f77b4", lw=1)
        ax2.axhline(0, color="#bbb", lw=0.5, ls="--")
        ax2.set_xlim(0, 200)
        ax2.set_ylim(-3.5, 3.5)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("x₁")
        ax2.set_title("Trace of x₁")

        # Panel 3: Marginal histogram
        if n_show >= 3:
            vals = samples[1 : n_show + 1, 0]
            bins = np.linspace(-3.5, 3.5, 30)
            ax3.hist(
                vals,
                bins=bins,
                orientation="horizontal",
                density=True,
                color="#1f77b4",
                alpha=0.45,
                edgecolor="none",
            )
            ax3.plot(
                metro_true_pdf_x,
                metro_true_pdf_y,
                color="gray",
                ls="--",
                lw=1.5,
                alpha=0.5,
            )
        ax3.set_ylim(-3.5, 3.5)
        ax3.set_xlabel("Density")
        ax3.set_title("Marginal")
        ax3.set_yticklabels([])

        fig.tight_layout()
        return fig

    _draw_metropolis_animation(metro_step_slider.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The critical tuning parameter is the **step size** — how far each proposal jumps. This single number controls the fundamental tradeoff of the algorithm:

    - **Too small**: Every proposal is accepted (it barely moved), but the chain crawls — it takes thousands of steps to cross the posterior. The trace looks like a slow, smooth random walk.
    - **Too large**: Most proposals land in low-density regions and get rejected. The chain gets stuck for long stretches, jumping only occasionally.
    - **Just right**: A mix of accepted and rejected proposals. The trace looks like a "fuzzy caterpillar" — exactly what we want.

    Try manually adjusting the step size below:
    """)
    return


@app.cell(hide_code=True)
def _(demo_metropolis):
    # Pre-compute traces for a range of step sizes
    _step_sizes = [round(0.01 + i * 0.01, 2) for i in range(300)]
    step_size_traces = {}
    for _ss in _step_sizes:
        _s, _, _a = demo_metropolis(500, _ss, np.random.default_rng(42))
        step_size_traces[f"{_ss:.2f}"] = {
            "x1": _s[1:, 0],
            "accept_rate": float(_a.mean()),
        }
    return (step_size_traces,)


@app.cell(hide_code=True)
def _():
    step_size_slider = mo.ui.slider(
        0.01, 3.0, step=0.01, value=0.5, label="Step size", full_width=True
    )
    step_size_slider
    return (step_size_slider,)


@app.cell(hide_code=True)
def _(step_size_slider, step_size_traces):
    def _draw_step_size_explorer(step_size):
        key = f"{step_size:.2f}"
        tr = step_size_traces.get(key)
        if tr is None:
            return

        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.plot(tr["x1"], color="#1f77b4", lw=1, alpha=0.85)
        ax.axhline(0, color="#bbb", lw=0.5, ls="--")
        ax.set_xlim(0, 500)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("x₁")
        ax.set_title(f"step size = {key}     accept rate = {tr['accept_rate']:.0%}")
        fig.tight_layout()
        return fig

    _draw_step_size_explorer(step_size_slider.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice how the step size controls everything about the sampler's behavior — the acceptance rate, the autocorrelation, and how quickly the chain explores the target distribution. This is what PyMC's **warmup phase** automates: finding the step size that produces that well-mixed caterpillar trace. When you call `pm.sample()`, the first `tune` draws (default: 1000) are a **warmup phase** where the sampler adapts to the posterior geometry.

    `pm.sample(tune=500)`

    These draws are discarded and never appear in your trace, because the sampler's behavior is non-stationary while it's still learning.

    If you see poor mixing in the early post-warmup draws, the warmup may not have been long enough. Try increasing `tune` (e.g., `tune=2000`). Complex models — hierarchical structures, many parameters, difficult geometry — often need more warmup to get both the step size and mass matrix right. We'll return to these settings in the sampler configuration tips in Session 3.2.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### The problem with Metropolis

    But even with optimal tuning, Metropolis has a fundamental limitation: because each step is a *random* perturbation, successive samples are highly correlated. The chain takes many small steps to traverse the posterior, so you need a very long chain to get a modest number of effectively independent samples. And in high dimensions, this problem gets dramatically worse.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Hamiltonian Monte Carlo and NUTS

    **Hamiltonian Monte Carlo (HMC)** solves this by using the *gradient* of the log-posterior to make informed proposals. Instead of random steps, HMC simulates a physical system: imagine placing a ball on a surface shaped like the posterior density and giving it a random push. The ball rolls along the surface following Hamiltonian dynamics, naturally staying in high-density regions while covering large distances.

    **NUTS** (the No-U-Turn Sampler) extends HMC by automatically choosing how far to "roll" — it stops the trajectory when it starts doubling back, which is the "no U-turn" criterion. This eliminates HMC's most sensitive tuning parameter (trajectory length).

    NUTS adapts two things during warmup:

    - **Step size**, via dual averaging — targeting the acceptance rate set by `target_accept` (default 0.8). This is the automated version of what the slider above let you do manually.
    - **Mass matrix** (also called the inverse metric) — an estimate of the posterior covariance that lets the sampler take appropriately-scaled steps in each direction. Without it, a parameter with SD = 0.01 and one with SD = 100 would need very different step sizes. The mass matrix handles this automatically.

    The result: proposals that are *distant* from the current position but still in high-density regions, producing nearly independent samples. Let's see the contrast directly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The animation below shows both samplers targeting the same correlated bivariate normal distribution ($\rho=0.9$). Watch how Metropolis proposes small random steps (many rejected in <span style="color:#d62728">red</span>), while HMC follows curved trajectories along the density surface — covering far more ground per step.
    """)
    return


@app.cell(hide_code=True)
def _(demo_hmc, demo_metropolis):
    # Pre-compute Metropolis and HMC samples for comparison
    _rng1 = np.random.default_rng(42)
    cmp_metro_s, cmp_metro_p, cmp_metro_a = demo_metropolis(
        80, step_size=0.5, rng=_rng1
    )
    _rng2 = np.random.default_rng(42)
    cmp_hmc_s, cmp_hmc_t, cmp_hmc_a = demo_hmc(
        80, step_size=0.15, n_leapfrog=20, rng=_rng2
    )

    # Pre-build LineCollection-ready segments and colors so slider redraws are cheap
    _n_metro = len(cmp_metro_a)
    cmp_metro_segments = np.empty((_n_metro, 2, 2))
    cmp_metro_segments[:, 0] = cmp_metro_s[:_n_metro]
    cmp_metro_segments[:, 1] = cmp_metro_p
    cmp_metro_seg_colors = np.where(cmp_metro_a, "#2ca02c", "#d62728")

    cmp_hmc_seg_colors = np.where(cmp_hmc_a, "#2ca02c", "#d62728")
    return (
        cmp_hmc_a,
        cmp_hmc_s,
        cmp_hmc_seg_colors,
        cmp_hmc_t,
        cmp_metro_a,
        cmp_metro_p,
        cmp_metro_s,
        cmp_metro_seg_colors,
        cmp_metro_segments,
    )


@app.cell(hide_code=True)
def _():
    cmp_step_slider = mo.ui.slider(
        1, 80, value=80, label="Steps to show", full_width=True
    )
    cmp_step_slider
    return (cmp_step_slider,)


@app.cell(hide_code=True)
def _(
    cmp_hmc_a,
    cmp_hmc_s,
    cmp_hmc_seg_colors,
    cmp_hmc_t,
    cmp_metro_a,
    cmp_metro_p,
    cmp_metro_s,
    cmp_metro_seg_colors,
    cmp_metro_segments,
    cmp_step_slider,
    demo_X,
    demo_Y,
    demo_Z,
):
    def _draw_mcmc_comparison(n_show):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for ax, title in [
            (ax1, "Random Walk Metropolis"),
            (ax2, "Hamiltonian Monte Carlo"),
        ]:
            ax.contour(
                demo_X,
                demo_Y,
                demo_Z,
                levels=6,
                colors="gray",
                alpha=0.4,
                linewidths=0.8,
            )
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_title(title)

        # Metropolis panel
        if n_show > 0:
            ax1.add_collection(
                LineCollection(
                    cmp_metro_segments[:n_show],
                    colors=cmp_metro_seg_colors[:n_show],
                    linewidths=1.5,
                    alpha=0.6,
                )
            )
            ax1.scatter(
                cmp_metro_p[:n_show, 0],
                cmp_metro_p[:n_show, 1],
                c=cmp_metro_seg_colors[:n_show],
                s=8,
                alpha=0.5,
                zorder=3,
            )
        ax1.scatter(
            cmp_metro_s[: n_show + 1, 0],
            cmp_metro_s[: n_show + 1, 1],
            color="#1f77b4",
            s=6,
            alpha=0.3,
            zorder=2,
        )
        ax1.scatter(
            cmp_metro_s[min(n_show, 80), 0],
            cmp_metro_s[min(n_show, 80), 1],
            color="#1f77b4",
            s=40,
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
        )
        m_acc = cmp_metro_a[:n_show].sum() / n_show if n_show > 0 else 0
        ax1.text(
            0.05,
            0.95,
            f"Step {n_show}/80\nAccept: {m_acc:.0%}",
            transform=ax1.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        # HMC panel — variable-length trajectories, use LineCollection with list input
        if n_show > 0:
            _trajs = [cmp_hmc_t[k] for k in range(min(n_show, len(cmp_hmc_a)))]
            ax2.add_collection(
                LineCollection(
                    _trajs,
                    colors=cmp_hmc_seg_colors[:n_show],
                    linewidths=1.0,
                    alpha=0.5,
                )
            )
        ax2.scatter(
            cmp_hmc_s[: n_show + 1, 0],
            cmp_hmc_s[: n_show + 1, 1],
            color="#1f77b4",
            s=6,
            alpha=0.3,
            zorder=2,
        )
        ax2.scatter(
            cmp_hmc_s[min(n_show, 80), 0],
            cmp_hmc_s[min(n_show, 80), 1],
            color="#1f77b4",
            s=40,
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
        )
        h_acc = cmp_hmc_a[:n_show].sum() / n_show if n_show > 0 else 0
        ax2.text(
            0.05,
            0.95,
            f"Step {n_show}/80\nAccept: {h_acc:.0%}",
            transform=ax2.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        fig.tight_layout()
        return fig

    _draw_mcmc_comparison(cmp_step_slider.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This contrast explains why PyMC defaults to NUTS for continuous parameters. Let's verify this on our penguin model by running the same model with Metropolis and comparing directly.
    """)
    return


@app.cell
def _(baseline_model):
    with baseline_model:
        metropolis_idata = pm.sample(
            step=pm.Metropolis(),
            random_seed=RANDOM_SEED,
        )
    return (metropolis_idata,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice that `pm.sample` drew **multiple chains** by default. MCMC is *embarassingly parallel* so we can easily use several computer cores to sample faster, in parallel.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now let's compare the autocorrelation — how correlated successive draws are — for both samplers on the same model.
    """)
    return


@app.cell
def _(metropolis_idata, nuts_idata):
    az.plot_autocorr(
        {"NUTS": nuts_idata, "Metropolis": metropolis_idata},
        var_names=["alpha", "beta", "sigma"],
        col_wrap=2,
        figure_kwargs=fig_kwargs(cols=2, rows=3),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The difference in effective sample size tells the same story numerically. NUTS produces nearly independent draws; Metropolis wastes most of its computation on correlated samples.
    """)
    return


@app.cell(hide_code=True)
def _(metropolis_idata, nuts_idata):
    _nuts_summary = az.summary(nuts_idata, var_names=["alpha", "beta", "sigma"])
    _metro_summary = az.summary(metropolis_idata, var_names=["alpha", "beta", "sigma"])

    mo.md(f"""
    **NUTS ESS:**

    {_nuts_summary[["ess_bulk", "ess_tail"]].to_markdown()}

    **Metropolis ESS:**

    {_metro_summary[["ess_bulk", "ess_tail"]].to_markdown()}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When ESS is much lower than expected, something about the posterior geometry is preventing efficient exploration — we'll see concrete examples in Session 3.2.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Visual Diagnostics

    The visual diagnostic tools in ArviZ let you inspect the sampler's behavior directly. For each one, we'll show NUTS and Metropolis side by side on the same model — so you can see what "healthy" and "inefficient but correct" look like, and practice reading the diagnostics before we meet genuinely broken samplers in Session 3.2.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Trace Plots

    The trace plot shows two things side by side: the posterior distribution (left) and the raw draws over time (right). You want to see "fuzzy caterpillars" — chains that mix well and overlap completely. `plot_trace_dist` combines the trace and distribution density in one view.
    """)
    return


@app.cell
def _(metropolis_idata, nuts_idata):
    az.plot_trace_dist(
        {"NUTS": nuts_idata, "Metropolis": metropolis_idata},
        var_names=["alpha", "beta", "sigma"],
        combined=True,
        figure_kwargs=fig_kwargs(cols=3, rows=2),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Rank Plots

    Rank plots (`plot_rank`) are often better than raw trace plots for detecting convergence problems. They work by ranking *all* draws across all chains (pooling them), converting to fractional ranks, then computing the empirical CDF of those ranks for each chain.

    The plot shows the **Δ-ECDF** — the difference between each chain's observed rank ECDF and the expected uniform CDF. If chains are sampling from the same distribution, the lines should be flat near zero, staying within the gray envelope. Lines that extend outside the envelope or show "squared-off" patterns indicate convergence problems or low ESS.
    """)
    return


@app.cell
def _(metropolis_idata, nuts_idata):
    az.plot_rank(
        {"NUTS": nuts_idata, "Metropolis": metropolis_idata},
        var_names=["alpha", "beta", "sigma"],
        figure_kwargs=fig_kwargs(cols=3, rows=2),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Numerical Diagnostics

    Now let's attach numbers to the visual picture. Again we'll compare NUTS and Metropolis side by side on each diagnostic. Metropolis here isn't *broken* — R-hat will still look fine, for example — it's just wasteful. Session 3.2 is where we'll meet diagnostics that flag genuinely broken models.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### R-hat ($\hat{R}$)

    *How do we know the chains have converged to the same distribution?*

    We ran 4 independent chains. If they've all found the posterior, they should agree. R-hat quantifies this by comparing variance *between* chains to variance *within* chains. Values near 1.0 mean agreement. **Threshold: R-hat < 1.01.**

    We'll see exactly what high R-hat looks like — and why it happens — in Session 3.2.
    """)
    return


@app.cell
def _(nuts_idata):
    az.rhat(nuts_idata, var_names=["alpha", "beta", "sigma"])
    return


@app.cell
def _(metropolis_idata):
    az.rhat(metropolis_idata, var_names=["alpha", "beta", "sigma"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both are near 1.0 — Metropolis is slow but the chains *do* agree on where the posterior is. **R-hat tells you about convergence, not efficiency** — it catches chains that disagree, not chains that are simply inefficient.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Monte Carlo Standard Error (MCSE)

    *How precise are our posterior summaries?*

    Even with good mixing, finite samples mean our estimates of the mean, SD, and quantiles have some Monte Carlo error. MCSE quantifies this. **Rule of thumb:** if MCSE / posterior SD > 0.1, you need more draws before trusting summaries. You can also think of MCSE as the number of decimal places you can trust — with MCSE ≈ 0.002, reporting to two decimal places is justified.
    """)
    return


@app.cell
def _(nuts_idata):
    az.mcse(nuts_idata, var_names=["alpha", "beta", "sigma"])
    return


@app.cell
def _(metropolis_idata):
    az.mcse(metropolis_idata, var_names=["alpha", "beta", "sigma"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Metropolis MCSE values are noticeably larger than NUTS — the same number of draws buys less precision because successive samples are correlated. `plot_mcse` visualises how MCSE changes across quantiles of the posterior.
    """)
    return


@app.cell
def _(metropolis_idata, nuts_idata):
    az.plot_mcse(
        {"NUTS": nuts_idata, "Metropolis": metropolis_idata},
        var_names=["alpha", "beta", "sigma"],
        figure_kwargs=fig_kwargs(cols=3),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### A Note on Thinning

    You may read older advice to "thin" your chains — keeping every $n$th draw to reduce autocorrelation. Modern practice generally favors **collecting more samples** over thinning, since thinning discards information. NUTS already produces low-autocorrelation draws.

    The one case where thinning makes sense is **storage**: if you have very many draws and only need rough summaries. ArviZ provides `az.thin()` for this:
    """)
    return


@app.cell(hide_code=True)
def _(nuts_idata):
    _thinned = az.thin(nuts_idata, factor="auto")
    _original_ess = az.ess(nuts_idata, var_names=["beta"])["beta"].item()
    _thinned_ess = az.ess(_thinned["beta"]).item()

    mo.md(f"""
    Original draws: {nuts_idata.posterior.sizes["draw"]}
    Thinned draws:  {_thinned.sizes["draw"]}

    Original beta ESS: {_original_ess:.0f}
    Thinned beta ESS:  {_thinned_ess:.0f}

    ESS barely changes — thinning doesn't create new information.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Energy and BFMI

    *Is the sampler exploring the full posterior, or stuck in one region?*

    This is an HMC-specific diagnostic. `az.plot_energy()` overlays two distributions — the marginal energy across all samples and the energy transition between successive samples. If the sampler explores efficiently, these should overlap well. BFMI (Bayesian Fraction of Missing Information) quantifies the overlap. **Values below 0.3 are concerning** — they indicate the sampler can't move freely between energy levels.
    """)
    return


@app.cell
def _(nuts_idata):
    _pc = az.plot_energy(
        nuts_idata, visuals={"legend": False}, figure_kwargs={"figsize": (10, 3.5)}
    )

    _fig = plt.gcf()
    _bfmi_ax, _energy_ax = _fig.axes[0], _fig.axes[1]

    _n_chains = nuts_idata.posterior.sizes["chain"]
    _bfmi_ax.set_yticks(range(_n_chains))
    _bfmi_ax.set_ylim(-0.5, _n_chains - 0.5)
    for _coll in _bfmi_ax.collections:
        _coll.set_sizes([80])

    _bfmi_ax.set_position([0.05, 0.15, 0.15, 0.75])
    _energy_ax.set_position([0.28, 0.15, 0.5, 0.75])

    for _line, _label in zip(_energy_ax.get_lines(), ["marginal", "transition"]):
        _line.set_label(_label)
    _energy_ax.legend(loc="upper right")

    _pc
    return


@app.cell
def _(nuts_idata):
    az.bfmi(nuts_idata)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Effective Sample Size (ESS)

    We saw ESS in the NUTS vs. Metropolis comparison above. Two variants matter:

    - **Bulk ESS**: How well the chain explores the center of the distribution
    - **Tail ESS**: How well it explores the extremes (5th and 95th percentiles) — often lower than bulk ESS

    Rule of thumb: you want at least **400 total effective samples** (across all chains) for reliable estimates.

    Two plots give complementary views of ESS:

    - **`plot_ess`** shows ESS across different quantiles of the posterior, helping identify whether the tails are as well-explored as the center.
    - **`plot_ess_evolution`** shows ESS as a function of the number of draws. It plots two lines per variable — **bulk ESS** and **tail ESS** — tracked as if the chain had been stopped earlier. Both lines should grow roughly linearly with the number of draws; if a line flattens, collecting more samples stops buying you independent information.
    """)
    return


@app.cell
def _(metropolis_idata, nuts_idata):
    az.plot_ess(
        {"NUTS": nuts_idata, "Metropolis": metropolis_idata},
        var_names=["alpha", "beta", "sigma"],
        kind="quantile",
        figure_kwargs=fig_kwargs(cols=3),
    )
    return


@app.cell
def _(metropolis_idata, nuts_idata):
    az.plot_ess_evolution(
        {"NUTS": nuts_idata, "Metropolis": metropolis_idata},
        var_names=["alpha", "beta", "sigma"],
        figure_kwargs=fig_kwargs(cols=3),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    All the diagnostics above look healthy — our baseline model is well-specified and the sampler is working correctly. In **Session 3.2**, we'll see what happens when things go wrong: non-identifiable models, divergences, inefficient sampling, and how to diagnose and fix each problem. We'll also cover posterior predictive checks and model comparison.
    """)
    return


if __name__ == "__main__":
    app.run()
