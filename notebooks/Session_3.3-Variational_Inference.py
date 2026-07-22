import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    layout_file="layouts/Session_2.2_Variational_Inference.slides.json",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Variational Inference in PyMC

    This notebook explains how **Automatic Differentiation Variational Inference (ADVI)** works in PyMC, when it is a good approximation to the posterior, when its mean-field assumptions break down, and how **reparameterization** can fix the latter. We also demonstrate **DADVI** (Deterministic ADVI) from `pymc-extras`.

    ---

    ## 1. What is ADVI?

    ADVI approximates the posterior $p(\theta \mid y)$ with a simpler distribution $q(\theta)$ drawn from a family $\mathcal{Q}$. In PyMC's default ADVI:

    1. **Transform** all constrained variables (e.g. HalfNormals, SDs) to the real line.
    2. **Assume a mean-field Gaussian** for $q$: each dimension is an independent univariate Normal.
    3. **Optimize** the ELBO (Evidence Lower Bound) via stochastic gradient ascent to find the best means and standard deviations.

    Because the approximation is **mean-field** (independent across dimensions) and **Gaussian**, it can struggle when the true posterior has strong correlations or non-Gaussian shapes (funnels, multimodality, heavy tails).
    """)
    return


@app.cell
def _():
    import numpy as np
    import pymc as pm
    import arviz as az
    import arviz_plots as azp
    import matplotlib.pyplot as plt
    from pymc_extras import as_model
    from pymc_extras.inference import fit_dadvi
    import time
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="The figure layout has changed")
    warnings.filterwarnings("ignore", message="overflow encountered in dot")
    RANDOM_SEED = 42
    rng = np.random.default_rng(RANDOM_SEED)
    return RANDOM_SEED, as_model, az, azp, fit_dadvi, np, plt, pm, rng, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. When ADVI works well: a simple, well-identified normal model

    When the posterior is roughly Gaussian and dimensions are weakly correlated, mean-field ADVI can be an excellent fast approximation.
    """)
    return


@app.cell
def _(rng):
    # Generate data from a normal with known-ish parameters
    n = 200
    true_mu = 5.0
    true_sigma = 2.0
    y_good = rng.normal(true_mu, true_sigma, size=n)
    return true_mu, true_sigma, y_good


@app.cell
def _(as_model, pm, y_good):
    @as_model()
    def _model_good():
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y_good)

    model_good = _model_good()
    return (model_good,)


@app.cell
def _(RANDOM_SEED, model_good, pm):
    idata_good_nuts = pm.sample(
        model=model_good,
        draws=1000,
        tune=1000,
        chains=4,
        cores=1,
        random_seed=RANDOM_SEED,
        target_accept=0.9,
    )
    return (idata_good_nuts,)


@app.cell
def _(RANDOM_SEED, model_good, np, pm):
    param_history_good = []

    def _callback_good(approx, losses, i):
        param_history_good.append(approx.mean.eval().copy())

    with model_good:
        approx_good = pm.fit(
            n=50000,
            random_seed=RANDOM_SEED,
            callbacks=[_callback_good],
        )
        idata_good_advi = approx_good.sample(1000)

    param_history_good = np.array(param_history_good)
    return approx_good, idata_good_advi, param_history_good


@app.cell(hide_code=True)
def _(azp, idata_good_advi, idata_good_nuts):
    azp.plot_dist(
        {"NUTS": idata_good_nuts, "ADVI": idata_good_advi},
        var_names=["mu", "sigma"],
        kind="kde",
        visuals={"face": {"alpha": 0.2}},
        col_wrap=2,
        figure_kwargs={"figsize": (12, 4)},
    )
    return


@app.cell(hide_code=True)
def _(approx_good, plt):
    fig_elbo, ax_elbo = plt.subplots(figsize=(12, 4))
    ax_elbo.plot(approx_good.hist)
    ax_elbo.set_title("ADVI ELBO (negative loss)")
    ax_elbo.set_xlabel("Iteration")
    ax_elbo.set_ylabel("Loss")
    plt.tight_layout()
    fig_elbo
    return


@app.cell(hide_code=True)
def _(approx_good, np, param_history_good, plt, true_mu, true_sigma):
    fig_good_conv, ax_good = plt.subplots(1, 2, figsize=(14, 4))

    # ELBO with rolling mean smoothing
    elbo_series = np.convolve(approx_good.hist, np.ones(500)/500, mode='valid')
    ax_good[0].plot(approx_good.hist, alpha=0.3, color="gray", lw=0.5)
    ax_good[0].plot(range(499, len(approx_good.hist)), elbo_series, color="steelblue", lw=2)
    ax_good[0].set_title("ELBO convergence")
    ax_good[0].set_xlabel("Iteration")
    ax_good[0].set_ylabel("Loss")
    ax_good[0].set_yscale("log")

    # Parameter means
    ax_good[1].plot(param_history_good[:, 0], label="μ mean", color="C0", lw=1)
    ax_good[1].plot(np.exp(param_history_good[:, 1]), label="σ mean", color="C1", lw=1)
    ax_good[1].axhline(true_mu, color="C0", ls="--", alpha=0.5)
    ax_good[1].axhline(true_sigma, color="C1", ls="--", alpha=0.5)
    ax_good[1].set_title("Parameter mean convergence")
    ax_good[1].set_xlabel("Iteration")
    ax_good[1].set_ylabel("Value")
    ax_good[1].legend()

    plt.tight_layout()
    fig_good_conv
    return


@app.cell(hide_code=True)
def _(az, idata_good_advi, idata_good_nuts, mo, np, true_mu, true_sigma):
    summary_nuts = az.summary(idata_good_nuts, var_names=["mu", "sigma"], kind="stats")[["mean", "sd"]]
    summary_advi = az.summary(idata_good_advi, var_names=["mu", "sigma"], kind="stats")[["mean", "sd"]]

    table_vals = np.column_stack([summary_nuts.to_numpy(), summary_advi.to_numpy()])

    mo.md(
        f"""
        ### Numerical comparison

        | | True | NUTS mean | NUTS sd | ADVI mean | ADVI sd |
        |---|---|---|---|---|---|
        | μ | {true_mu} | {table_vals[0][0]} | {table_vals[0][1]} | {table_vals[0][2]} | {table_vals[0][3]} |
        | σ | {true_sigma} | {table_vals[1][0]} | {table_vals[1][1]} | {table_vals[1][2]} | {table_vals[1][3]} |

        ADVI nails the posterior means and does a reasonable job on the standard deviations. The ELBO converges smoothly. This is the ideal scenario: plenty of data, a simple model, and a posterior that is close to Gaussian with weak correlations.
        """
    )
    return


@app.cell
def _(RANDOM_SEED, mo, model_good, pm, time):
    # Wall-clock timing: NUTS vs ADVI on the same model -- 2,000 iterations each

    _t0_nuts = time.perf_counter()
    _idata_nuts_timed = pm.sample(
        model=model_good,
        draws=1000,
        tune=1000,
        chains=4,
        cores=1,
        random_seed=RANDOM_SEED,
        target_accept=0.9,
    )
    nuts_wall = time.perf_counter() - _t0_nuts
    nuts_logp_evals = int(_idata_nuts_timed["sample_stats"]["n_steps"].sum().values)

    _t0_advi = time.perf_counter()
    with model_good:
        _approx_timed = pm.fit(
            n=2000,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )
        _idata_advi_timed = _approx_timed.sample(1000)
    advi_wall = time.perf_counter() - _t0_advi
    advi_iters = len(_approx_timed.hist)

    _advi_obj_final = _approx_timed.hist[-1]

    mo.md(
        f"""
        ### ⏱ Speed comparison (2,000 iterations each)

        | | NUTS | ADVI | Speedup |
        |---|---|---|---|
        | Iterations | 2,000 (1,000 draw + 1,000 tune) | 2,000 | -- |
        | Wall time | {nuts_wall:.2f}s | {advi_wall:.2f}s | **{nuts_wall / advi_wall:.0f}×** |
        | Logp evaluations | {nuts_logp_evals:,} | {advi_iters:,} gradient steps | -- |

        ADVI is **{nuts_wall / advi_wall:.0f}× faster** than NUTS on this simple normal model
        ({nuts_wall:.1f}s vs {advi_wall:.1f}s) at the same iteration count.
        The ELBO converges to {_advi_obj_final:.1f} in {advi_iters:,} iterations -- no tuning phase,
        no warmup, just straight gradient descent on the variational objective.

        > On larger models, the speed difference grows. While NUTS scales with the number of
        > parameters (each leapfrog step touches every parameter), ADVI's per-iteration cost is
        > similar but it typically needs far fewer iterations to reach a useful approximation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. When ADVI assumptions are a poor fit

    Mean-field ADVI assumes **independent** Gaussian dimensions. When the true posterior has strong correlations, ADVI cannot capture the joint shape and typically **underestimates variances**.

    A classic source of strong posterior correlation is a regression with an **uncentered predictor**: the intercept and slope become tightly coupled because they trade off to fit the same data.
    """)
    return


@app.cell
def _(np, rng):
    def generate_uncentered():
        n = 50
        x_uncentered = np.linspace(0, 10, n)
        true_alpha = 2.0
        true_beta = 0.5
        y_reg = true_alpha + true_beta * x_uncentered + rng.normal(0, 0.5, size=n)
        return (y_reg, x_uncentered)


    y_reg, x_uncentered = generate_uncentered()
    return x_uncentered, y_reg


@app.cell
def _(x_uncentered):
    x_centered = x_uncentered - x_uncentered.mean()
    return (x_centered,)


@app.cell
def _(as_model, pm, x_uncentered, y_reg):
    @as_model()
    def _model_uncentered():
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        pm.Normal("y", mu=alpha + beta * x_uncentered, sigma=sigma, observed=y_reg)

    model_uncentered = _model_uncentered()
    return (model_uncentered,)


@app.cell
def _(RANDOM_SEED, model_uncentered, pm):
    idata_uncentered_nuts = pm.sample(
        model=model_uncentered,
        draws=1000,
        tune=1000,
        chains=4,
        cores=1,
        random_seed=RANDOM_SEED,
        target_accept=0.9,
    )
    return (idata_uncentered_nuts,)


@app.cell
def _(RANDOM_SEED, model_uncentered, np, pm):
    param_history_unc = []

    def _callback_unc(approx, losses, i):
        param_history_unc.append(approx.mean.eval().copy())

    with model_uncentered:
        approx_uncentered = pm.fit(
            n=50000,
            random_seed=RANDOM_SEED,
            callbacks=[_callback_unc],
        )
        idata_uncentered_advi = approx_uncentered.sample(1000)

    param_history_unc = np.array(param_history_unc)
    return approx_uncentered, idata_uncentered_advi, param_history_unc


@app.cell(hide_code=True)
def _(idata_uncentered_advi, idata_uncentered_nuts, np, plt):
    fig_unc, axes_unc = plt.subplots(1, 2, figsize=(14, 5))

    a_n_unc = idata_uncentered_nuts["posterior"]["alpha"].values.flatten()
    b_n_unc = idata_uncentered_nuts["posterior"]["beta"].values.flatten()
    corr_nuts_unc = np.corrcoef(a_n_unc, b_n_unc)[0, 1]
    axes_unc[0].hexbin(a_n_unc, b_n_unc, gridsize=30, cmap="Blues", mincnt=1)
    axes_unc[0].set_xlabel("alpha")
    axes_unc[0].set_ylabel("beta")
    axes_unc[0].set_title(f"NUTS joint posterior\nCorr(α, β) = {corr_nuts_unc:.2f}")

    a_a_unc = idata_uncentered_advi["posterior"]["alpha"].values.flatten()
    b_a_unc = idata_uncentered_advi["posterior"]["beta"].values.flatten()
    corr_advi_unc = np.corrcoef(a_a_unc, b_a_unc)[0, 1]
    axes_unc[1].hexbin(a_a_unc, b_a_unc, gridsize=30, cmap="Oranges", mincnt=1)
    axes_unc[1].set_xlabel("alpha")
    axes_unc[1].set_ylabel("beta")
    axes_unc[1].set_title(f"ADVI mean-field approximation\nCorr(α, β) = {corr_advi_unc:.2f}")

    plt.tight_layout()
    fig_unc
    return


@app.cell(hide_code=True)
def _(approx_uncentered, np, param_history_unc, plt):
    fig_unc_conv, ax_unc = plt.subplots(1, 3, figsize=(18, 4))

    # ELBO
    smooth_unc = np.convolve(approx_uncentered.hist, np.ones(500)/500, mode='valid')
    ax_unc[0].plot(approx_uncentered.hist, alpha=0.3, color="gray", lw=0.5)
    ax_unc[0].plot(range(499, len(approx_uncentered.hist)), smooth_unc, color="steelblue", lw=2)
    ax_unc[0].set_title("ELBO convergence")
    ax_unc[0].set_yscale("log")
    ax_unc[0].set_xlabel("Iteration")
    ax_unc[0].set_ylabel("Loss")

    # Alpha and beta convergence
    ax_unc[1].plot(param_history_unc[:, 0], label="α", color="C0", lw=1)
    ax_unc[1].plot(param_history_unc[:, 1], label="β", color="C1", lw=1)
    ax_unc[1].axhline(2.0, color="C0", ls="--", alpha=0.5)
    ax_unc[1].axhline(0.5, color="C1", ls="--", alpha=0.5)
    ax_unc[1].set_title("α, β convergence (uncentered)")
    ax_unc[1].set_xlabel("Iteration")
    ax_unc[1].set_ylabel("Value")
    ax_unc[1].legend()

    # Sigma convergence
    ax_unc[2].plot(np.exp(param_history_unc[:, 2]), label="σ", color="C2", lw=1)
    ax_unc[2].axhline(0.5, color="C2", ls="--", alpha=0.5)
    ax_unc[2].set_title("σ convergence (uncentered)")
    ax_unc[2].set_xlabel("Iteration")
    ax_unc[2].set_ylabel("Value")

    plt.tight_layout()
    fig_unc_conv
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Take-away
    The true posterior (NUTS) shows a strong negative correlation between $\alpha$ and $\beta$ because the predictor $x$ is far from zero. ADVI's mean-field approximation forces independence, so it misses the correlation entirely. It also **underestimates the marginal variances** because it cannot spread mass along the diagonal direction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Fixing the fit with reparameterization

    The cure is to **reparameterize** the model so that the posterior dimensions are approximately independent. For regression, centering the predictor removes the intercept--slope correlation.

    We rewrite:
    $$ y = \alpha' + \beta \,(x - \bar{x}) + \varepsilon $$

    Now $\alpha'$ represents the expected $y$ at the average $x$, and it is a-posteriori independent of $\beta$.
    """)
    return


@app.cell
def _(as_model, pm, x_centered, x_uncentered, y_reg):
    @as_model()
    def _model_centered():
        # alpha_c is the intercept when x is at its mean (x_centered = 0)
        alpha_c = pm.Normal("alpha_c", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        # Deterministic: recover the intercept in original (uncentered) x-space
        #   mu = alpha_c + beta*(x - x_bar) = (alpha_c - beta*x_bar) + beta*x
        #   So alpha_original = alpha_c - beta*x_bar
        pm.Deterministic("alpha", alpha_c - beta * x_uncentered.mean())
        pm.Normal("y", mu=alpha_c + beta * x_centered, sigma=sigma, observed=y_reg)

    model_centered = _model_centered()
    return (model_centered,)


@app.cell
def _(RANDOM_SEED, model_centered, pm):
    idata_centered_nuts = pm.sample(
        model=model_centered,
        draws=1000,
        tune=1000,
        chains=4,
        cores=1,
        random_seed=RANDOM_SEED,
        target_accept=0.9,
    )
    return (idata_centered_nuts,)


@app.cell
def _(RANDOM_SEED, model_centered, np, pm, time):
    param_history_cen = []
    advi_walltime_cen = []
    _start_cen = time.perf_counter()

    def _cb_cen_timed(approx, losses, i):
        param_history_cen.append(approx.mean.eval().copy())
        advi_walltime_cen.append(time.perf_counter() - _start_cen)

    with model_centered:
        approx_centered = pm.fit(
            n=50000,
            random_seed=RANDOM_SEED,
            callbacks=[_cb_cen_timed],
        )
        idata_centered_advi = approx_centered.sample(1000)

    param_history_cen = np.array(param_history_cen)
    advi_walltime_cen = np.array(advi_walltime_cen)
    return (
        advi_walltime_cen,
        approx_centered,
        idata_centered_advi,
        param_history_cen,
    )


@app.cell(hide_code=True)
def _(idata_centered_advi, idata_centered_nuts, np, plt):
    fig_cen, axes_cen = plt.subplots(1, 2, figsize=(14, 5))

    a_n_cen = idata_centered_nuts["posterior"]["alpha"].values.flatten()
    b_n_cen = idata_centered_nuts["posterior"]["beta"].values.flatten()
    corr_nuts_cen = np.corrcoef(a_n_cen, b_n_cen)[0, 1]
    axes_cen[0].hexbin(a_n_cen, b_n_cen, gridsize=30, cmap="Blues", mincnt=1)
    axes_cen[0].set_xlabel("alpha")
    axes_cen[0].set_ylabel("beta")
    axes_cen[0].set_title(f"NUTS (centered x)\nCorr(α, β) = {corr_nuts_cen:.2f}")

    a_a_cen = idata_centered_advi["posterior"]["alpha"].values.flatten()
    b_a_cen = idata_centered_advi["posterior"]["beta"].values.flatten()
    corr_advi_cen = np.corrcoef(a_a_cen, b_a_cen)[0, 1]
    axes_cen[1].hexbin(a_a_cen, b_a_cen, gridsize=30, cmap="Oranges", mincnt=1)
    axes_cen[1].set_xlabel("alpha")
    axes_cen[1].set_ylabel("beta")
    axes_cen[1].set_title(f"ADVI (centered x)\nCorr(α, β) = {corr_advi_cen:.2f}")

    plt.tight_layout()
    fig_cen
    return


@app.cell(hide_code=True)
def _(approx_centered, np, param_history_cen, plt, x_uncentered):
    x_bar = x_uncentered.mean()
    # param_history_cen columns: alpha_c, beta, sigma (free params)
    # Compute deterministic alpha = alpha_c - beta * x_bar
    alpha_conv = param_history_cen[:, 0] - param_history_cen[:, 1] * x_bar

    fig_cen_conv, ax_cen = plt.subplots(1, 3, figsize=(18, 4))

    # ELBO
    smooth_cen = np.convolve(approx_centered.hist, np.ones(500)/500, mode='valid')
    ax_cen[0].plot(approx_centered.hist, alpha=0.3, color="gray", lw=0.5)
    ax_cen[0].plot(range(499, len(approx_centered.hist)), smooth_cen, color="steelblue", lw=2)
    ax_cen[0].set_title("ELBO convergence (centered x)")
    ax_cen[0].set_xlabel("Iteration")
    ax_cen[0].set_ylabel("Loss")
    ax_cen[0].set_yscale("log")

    # Alpha (Deterministic) and beta convergence
    ax_cen[1].plot(alpha_conv, label="\u03b1 (deterministic)", color="C0", lw=1)
    ax_cen[1].plot(param_history_cen[:, 1], label="\u03b2", color="C1", lw=1)
    ax_cen[1].axhline(2.0, color="C0", ls="--", alpha=0.5)
    ax_cen[1].axhline(0.5, color="C1", ls="--", alpha=0.5)
    ax_cen[1].set_title("\u03b1, \u03b2 convergence (centered)")
    ax_cen[1].set_xlabel("Iteration")
    ax_cen[1].set_ylabel("Value")
    ax_cen[1].legend()

    # Sigma convergence
    ax_cen[2].plot(np.exp(param_history_cen[:, 2]), label="\u03c3", color="C2", lw=1)
    ax_cen[2].axhline(0.5, color="C2", ls="--", alpha=0.5)
    ax_cen[2].set_title("\u03c3 convergence (centered)")
    ax_cen[2].set_xlabel("Iteration")
    ax_cen[2].set_ylabel("Value")

    plt.tight_layout()
    fig_cen_conv
    return (x_bar,)


@app.cell(hide_code=True)
def _(
    az,
    idata_centered_advi,
    idata_centered_nuts,
    idata_uncentered_advi,
    idata_uncentered_nuts,
    mo,
    np,
):
    summary_unc = np.column_stack([
            az.summary(idata_uncentered_nuts, var_names=["alpha", "beta", "sigma"], kind="stats")[["mean", "sd"]].to_numpy(),
            az.summary(idata_uncentered_advi, var_names=["alpha", "beta", "sigma"], kind="stats")[["mean", "sd"]].to_numpy(),
    ])

    summary_cen = np.column_stack([
            az.summary(idata_centered_nuts, var_names=["alpha", "beta", "sigma"], kind="stats")[["mean", "sd"]].to_numpy(),
            az.summary(idata_centered_advi, var_names=["alpha", "beta", "sigma"], kind="stats")[["mean", "sd"]].to_numpy(),
    ])

    mo.md(
        f"""
        ### Uncentered $x$ (correlated posterior, ADVI struggles)

        | Param | True | NUTS mean | NUTS sd | ADVI mean | ADVI sd |
        |-------|------|-----------|---------|-----------|---------|
        | α | 2.0 | {summary_unc[0][0]} | {summary_unc[0][1]} | {summary_unc[0][2]} | {summary_unc[0][3]} |
        | β | 0.5 | {summary_unc[1][0]} | {summary_unc[1][1]} | {summary_unc[1][2]} | {summary_unc[1][3]} |
        | σ | 0.5 | {summary_unc[2][0]} | {summary_unc[2][1]} | {summary_unc[2][2]} | {summary_unc[2][3]} |

        ADVI underestimates the uncertainty (sd too small) because the mean-field assumption cannot capture the strong negative correlation between α and β.

        ### Centered $x$ (reparameterized, ADVI performs well)

        | Param | True | NUTS mean | NUTS sd | ADVI mean | ADVI sd |
        |-------|------|-----------|---------|-----------|---------|
        | α | 2.0 | {summary_cen[0][0]} | {summary_cen[0][1]} | {summary_cen[0][2]} | {summary_cen[0][3]} |
        | β | 0.5 | {summary_cen[1][0]} | {summary_cen[1][1]} | {summary_cen[1][2]} | {summary_cen[1][3]} |
        | σ | 0.5 | {summary_cen[2][0]} | {summary_cen[2][1]} | {summary_cen[2][2]} | {summary_cen[2][3]} |

        After centering $x$, the posterior correlation vanishes and ADVI’s standard deviations line up with NUTS. The Deterministic α recovers the original intercept, making it directly comparable across models.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. ADVI convergence in action: website traffic forecasting

    To see how ADVI converges in practice, we fit a model of daily website visits —
    an RBF-spline trend plus weekly Fourier seasonality — with both **MCMC** (NUTS,
    as a reference) and **ADVI** (stochastic gradient descent).

    The model has 17 free parameters: 12 spline coefficients for the trend, 4
    Fourier coefficients for the weekly pattern, and an observation-noise standard
    deviation.

    Use the slider below the fits to scrub through ADVI iterations and watch the
    fitted curve converge from the prior toward the data.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    # Generate synthetic website traffic data (S7)
    _rng = np.random.default_rng(42)
    n_days_traffic = 70
    day_traffic = np.arange(n_days_traffic, dtype=float)

    # RBF spline basis for the trend
    n_knots_traffic = 12
    _knots = np.linspace(0, n_days_traffic - 1, n_knots_traffic)
    _width = n_days_traffic / n_knots_traffic * 1.5
    basis_traffic = np.exp(
        -0.5 * ((day_traffic[:, None] - _knots[None, :]) / _width) ** 2
    )

    # True parameters
    _true_beta_spline = _rng.normal(0, 30, size=n_knots_traffic)
    _true_trend = basis_traffic @ _true_beta_spline + 100
    _true_seasonal = (
        15.0 * np.cos(2 * np.pi * day_traffic / 7)
        + 8.0 * np.sin(2 * np.pi * day_traffic / 7)
        + -5.0 * np.cos(4 * np.pi * day_traffic / 7)
        + 3.0 * np.sin(4 * np.pi * day_traffic / 7)
    )
    _true_mu = _true_trend + _true_seasonal
    y_traffic = _rng.normal(_true_mu, 6.0)
    return basis_traffic, day_traffic, n_knots_traffic, y_traffic


@app.cell(hide_code=True)
def _(basis_traffic, day_traffic, mo, n_knots_traffic, np, pm, y_traffic):
    # Model: RBF-spline trend + weekly Fourier seasonality (S7)
    import pytensor.tensor as _pt

    with pm.Model(
        coords={"day": day_traffic, "knot": np.arange(n_knots_traffic)}
    ) as model_traffic:
        # Trend via RBF spline coefficients
        beta_spline = pm.Normal("beta_spline", 0, 50, dims="knot")

        # Weekly seasonality — two harmonics
        beta_cos1 = pm.Normal("beta_cos1", 0, 30)
        beta_sin1 = pm.Normal("beta_sin1", 0, 30)
        beta_cos2 = pm.Normal("beta_cos2", 0, 20)
        beta_sin2 = pm.Normal("beta_sin2", 0, 20)

        # Observation noise
        sigma = pm.HalfNormal("sigma", sigma=15)

        # Construct trend and seasonal components
        trend = pm.Deterministic(
            "trend", _pt.dot(basis_traffic.astype("float64"), beta_spline), dims="day"
        )
        seasonal = pm.Deterministic(
            "seasonal",
            beta_cos1 * np.cos(2 * np.pi * day_traffic / 7)
            + beta_sin1 * np.sin(2 * np.pi * day_traffic / 7)
            + beta_cos2 * np.cos(4 * np.pi * day_traffic / 7)
            + beta_sin2 * np.sin(4 * np.pi * day_traffic / 7),
            dims="day",
        )
        mu = pm.Deterministic("mu", trend + seasonal, dims="day")
        y = pm.Normal("y", mu, sigma, observed=y_traffic, dims="day")

    # Custom-styled mermaid diagram
    mo.mermaid("""
    %%{init: {'theme': 'base', 'themeVariables': {
    'primaryColor': '#e8f0fe',
    'primaryTextColor': '#1a3a5c',
    'primaryBorderColor': '#4a9ede',
    'lineColor': '#6b8aaa',
    'tertiaryColor': '#f0f4f8'
    }}}%%
    graph TD
    subgraph priors["<b>Priors</b>"]
        bs["<b>&beta;<sub>spline</sub></b><br/>N(0, 50) × 12"]
        bc1["<b>&beta;<sub>cos1</sub></b><br/>N(0, 30)"]
        bs1["<b>&beta;<sub>sin1</sub></b><br/>N(0, 30)"]
        bc2["<b>&beta;<sub>cos2</sub></b><br/>N(0, 20)"]
        bs2["<b>&beta;<sub>sin2</sub></b><br/>N(0, 20)"]
        sig["<b>&sigma;</b><br/>HalfNormal(15)"]
    end

    subgraph det["<b>Deterministic</b>"]
        trend["<b>trend</b><br/>= B &middot; &beta;<sub>spline</sub>"]
        seas["<b>seasonal</b><br/>= &Sigma; Fourier terms"]
        mu["<b>&mu;</b><br/>= trend + seasonal"]
    end

    yobs["<b>y</b> ~ Normal(&mu;, &sigma;)<br/>(observed)"]

    bs --> trend
    bc1 & bs1 & bc2 & bs2 --> seas
    trend --> mu
    seas --> mu
    mu --> yobs
    sig --> yobs
    """)
    return (model_traffic,)


@app.cell
def _(RANDOM_SEED, model_traffic, pm):
    # MCMC reference fit — NUTS (S7)
    idata_traffic_nuts = pm.sample(
        model=model_traffic, draws=500, tune=500,
        random_seed=RANDOM_SEED, chains=2,
    )
    return (idata_traffic_nuts,)


@app.cell
def _(RANDOM_SEED, model_traffic, pm, time):
    # ADVI fit with per-iteration parameter recording
    param_history_traffic = []
    walltime_history_traffic = []
    _t0_traffic = time.perf_counter()

    def _traffic_callback(approx, loss, i):
        param_history_traffic.append(approx.mean.eval().copy())
        walltime_history_traffic.append(time.perf_counter() - _t0_traffic)

    approx_traffic = pm.fit(
        model=model_traffic, method="advi", n=1_000_000,
        callbacks=[_traffic_callback],
        random_seed=RANDOM_SEED,
    )
    return param_history_traffic, walltime_history_traffic


@app.cell(hide_code=True)
def _(
    basis_traffic,
    day_traffic,
    idata_traffic_nuts,
    n_knots_traffic,
    np,
    param_history_traffic,
):
    # Precompute ADVI fitted mu, subsampled every 500th iteration
    _param_hist = np.array(param_history_traffic)
    n_iters_traffic = len(_param_hist)
    _step = 500

    # Vectorized: trend = (n_frames, 12) @ (12, 70) = (n_frames, 70)
    _idx = np.arange(0, n_iters_traffic, _step)
    _all_beta = _param_hist[_idx, :n_knots_traffic].astype(np.float64)
    _all_trend = _all_beta @ basis_traffic.T.astype(np.float64)

    _c1 = np.cos(2 * np.pi * day_traffic / 7)
    _s1 = np.sin(2 * np.pi * day_traffic / 7)
    _c2 = np.cos(4 * np.pi * day_traffic / 7)
    _s2 = np.sin(4 * np.pi * day_traffic / 7)

    _all_seasonal = (
        _param_hist[_idx, n_knots_traffic, None] * _c1
        + _param_hist[_idx, n_knots_traffic + 1, None] * _s1
        + _param_hist[_idx, n_knots_traffic + 2, None] * _c2
        + _param_hist[_idx, n_knots_traffic + 3, None] * _s2
    )

    mu_history_traffic = _all_trend + _all_seasonal

    # MCMC posterior mean mu for reference
    mu_mcmc_mean_traffic = (
        idata_traffic_nuts["/posterior"]["mu"]
        .mean(dim=("chain", "draw"))
        .values
    )
    return mu_history_traffic, mu_mcmc_mean_traffic, n_iters_traffic


@app.cell
def _(mo, n_iters_traffic):
    # Slider to scrub through ADVI iterations
    advi_iter_slider = mo.ui.slider(
        start=1, stop=n_iters_traffic, step=500, value=n_iters_traffic,
        label="ADVI iteration",
        show_value=True,
    )
    advi_iter_slider
    return (advi_iter_slider,)


@app.cell(hide_code=True)
def _(
    advi_iter_slider,
    day_traffic,
    mu_history_traffic,
    mu_mcmc_mean_traffic,
    n_iters_traffic,
    plt,
    y_traffic,
):
    # Plot: observed data, MCMC reference, ADVI fit at current slider position
    _k = advi_iter_slider.value - 1

    _fig, _ax = plt.subplots(figsize=(12, 5))
    _ax.scatter(day_traffic, y_traffic, s=20, alpha=0.5, color="gray", label="Observed traffic", zorder=2)
    _ax.plot(day_traffic, mu_mcmc_mean_traffic, "k--", lw=1.5, alpha=0.6, label="MCMC posterior mean (reference)", zorder=3)
    _ax.plot(day_traffic, mu_history_traffic[_k // 500], color="steelblue", lw=2.5, label=f"ADVI at iteration {_k:,}", zorder=4)
    _ax.set_xlabel("Day")
    _ax.set_ylabel("Daily visits")
    _ax.set_title(f"ADVI fit vs MCMC reference \u2014 iteration {_k+1:,} of {n_iters_traffic:,}")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3)
    _fig
    return


@app.cell
def _(
    RANDOM_SEED,
    mo,
    model_traffic,
    param_history_traffic,
    pm,
    time,
    walltime_history_traffic,
):
    # Wall-clock timing: NUTS vs ADVI — website traffic model
    # ADVI wall time was already captured by the callback in the cell above.
    # We re-time NUTS for a fair comparison.

    _t0_nuts2 = time.perf_counter()
    _idata_traffic_nuts_timed = pm.sample(
        model=model_traffic,
        draws=500,
        tune=500,
        random_seed=RANDOM_SEED,
        chains=2,
    )
    nuts_traffic_wall = time.perf_counter() - _t0_nuts2
    nuts_traffic_steps = int(_idata_traffic_nuts_timed["sample_stats"]["n_steps"].sum().values)

    advi_traffic_wall = walltime_history_traffic[-1]
    advi_traffic_iters = len(param_history_traffic)

    mo.md(
        f"""
        ### ⏱️ Speed comparison — website traffic model

        | | NUTS | ADVI | Speedup |
        |---|---|---|---|
        | Wall time | {nuts_traffic_wall:.1f}s | {advi_traffic_wall:.1f}s | **{nuts_traffic_wall / advi_traffic_wall:.0f}×** |
        | Logp evaluations | {nuts_traffic_steps:,} | {advi_traffic_iters:,} iterations | — |

        On this 11-parameter model with an RBF-spline trend and Fourier seasonality, ADVI is
        **{nuts_traffic_wall / advi_traffic_wall:.0f}× faster** than NUTS
        ({advi_traffic_wall:.1f}s vs {nuts_traffic_wall:.1f}s). The ADVI callback records every
        iteration's ELBO, letting us scrub through the convergence path with the slider below
        — something that's impossible with MCMC.

        > ADVI's speed advantage grows with model complexity. On large hierarchical models or
        > Gaussian processes, ADVI can be **10–100× faster** than NUTS, making it a practical
        > choice for exploratory modelling when exact posterior samples are not strictly required.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. DADVI from pymc-extras

    **DADVI** (Deterministic ADVI) replaces the stochastic gradient optimization of standard ADVI with a **deterministic optimizer** (e.g. `scipy.optimize.minimize`). It fixes a small set of random draws at the start and optimizes the ELBO exactly, so:

    - **Convergence is reliable** — you get a standard optimization stopping criterion instead of the noisy ELBO trace of ADVI.
    - **No step-size tuning** — it uses second-order methods by default (`trust-ncg`).
    - **It is still mean-field**, so it shares ADVI's limitations on correlated posteriors. Reparameterization still matters!

    The API is `fit_dadvi()` (or `pymc_extras.inference.fit(method="dadvi")`).
    """)
    return


@app.cell
def _(RANDOM_SEED, fit_dadvi, model_centered):
    idata_centered_dadvi = fit_dadvi(model=model_centered, random_seed=RANDOM_SEED)
    return (idata_centered_dadvi,)


@app.cell
def _(RANDOM_SEED, fit_dadvi, model_uncentered):
    idata_uncentered_dadvi = fit_dadvi(model=model_uncentered, random_seed=RANDOM_SEED)
    return (idata_uncentered_dadvi,)


@app.cell(hide_code=True)
def _(azp, idata_centered_advi, idata_centered_dadvi, idata_centered_nuts):
    azp.plot_dist(
        {"NUTS": idata_centered_nuts, "ADVI": idata_centered_advi, "DADVI": idata_centered_dadvi},
        var_names=["alpha", "beta"],
        kind="kde",
        visuals={"face": {"alpha": 0.2}},
        col_wrap=2,
        figure_kwargs={"figsize": (12, 4)},
    )
    return


@app.cell(hide_code=True)
def _(
    azp,
    idata_uncentered_advi,
    idata_uncentered_dadvi,
    idata_uncentered_nuts,
):
    azp.plot_dist(
        {"NUTS": idata_uncentered_nuts, "ADVI": idata_uncentered_advi, "DADVI": idata_uncentered_dadvi},
        var_names=["alpha", "beta"],
        kind="kde",
        visuals={"face": {"alpha": 0.2}},
        col_wrap=2,
        figure_kwargs={"figsize": (12, 4)},
    )
    return


@app.cell(hide_code=True)
def _(
    RANDOM_SEED,
    advi_walltime_cen,
    approx_centered,
    model_centered,
    np,
    param_history_cen,
    plt,
    time,
    x_bar,
):
    # Compare ADVI vs DADVI convergence — wall-clock time (centered model)
    # Uses ADVI trace from ecfG; runs DADVI via scipy directly to record per-iteration history

    from scipy.optimize import minimize as scipy_minimize
    from pymc_extras.inference.dadvi.dadvi import create_dadvi_graph
    from pymc_extras.inference.laplace_approx.scipy_interface import (
        scipy_optimize_funcs_from_loss, set_optimizer_function_defaults,
    )
    from pymc.blocking import DictToArrayBijection

    # --- Build and run DADVI — time from graph creation ---
    _t0 = time.perf_counter()

    _n_fixed = 30
    _ip_dict = model_centered.initial_point()
    _ip = DictToArrayBijection.map(_ip_dict)
    _n_params = _ip.data.shape[0]

    _var_params, _objective = create_dadvi_graph(
        model_centered, n_fixed_draws=_n_fixed,
        random_seed=RANDOM_SEED, n_params=_n_params,
    )

    _use_grad, _use_hess, _use_hessp = set_optimizer_function_defaults("trust-ncg", None, None, None)
    _f_fused, _f_hessp = scipy_optimize_funcs_from_loss(
        loss=_objective, inputs=[_var_params], initial_point_dict=None,
        use_grad=_use_grad, use_hessp=_use_hessp, use_hess=False,
        gradient_backend="pytensor", compile_kwargs=None, inputs_are_flat=True,
    )

    _dadvi_ip = {
        f"{vn}_mu": np.asarray(v).ravel() for vn, v in _ip_dict.items()
    }
    _dadvi_ip.update({
        f"{vn}_sigma__log": np.zeros_like(v).ravel() for vn, v in _ip_dict.items()
    })
    _dadvi_ip = DictToArrayBijection.map(_dadvi_ip)

    dadvi_obj_hist = []
    dadvi_x_hist = []

    def _dadvi_cb(xk):
        dadvi_x_hist.append(xk.copy())
        fval, _ = _f_fused(xk)
        dadvi_obj_hist.append(fval)

    _result = scipy_minimize(
        _f_fused, _dadvi_ip.data, method="trust-ncg",
        jac=True, hessp=_f_hessp, callback=_dadvi_cb,
    )

    dadvi_elapsed_total = time.perf_counter() - _t0
    dadvi_x_hist = np.array(dadvi_x_hist)
    dadvi_obj_hist = np.array(dadvi_obj_hist)
    n_dadvi_iters = len(dadvi_x_hist)

    # Evenly distribute total wall time across iterations
    dadvi_times_cen = np.linspace(0, dadvi_elapsed_total, n_dadvi_iters)

    # Map to natural parameters
    alpha_c_dadvi = dadvi_x_hist[:, 0]
    beta_dadvi = dadvi_x_hist[:, 1]
    sigma_dadvi = np.exp(dadvi_x_hist[:, 2])
    alpha_dadvi = alpha_c_dadvi - beta_dadvi * x_bar

    # --- ADVI trace (from ecfG) ---
    advi_alpha_det = param_history_cen[:, 0] - param_history_cen[:, 1] * x_bar
    advi_for_plot = np.column_stack([
        advi_alpha_det,
        param_history_cen[:, 1],
        np.exp(param_history_cen[:, 2]),
    ])

    # Smoothed ADVI ELBO
    advi_elbo_smooth = np.convolve(approx_centered.hist, np.ones(500)/500, mode='valid')
    advi_elbo_smooth_t = advi_walltime_cen[499:]

    # Extend DADVI traces to full x-axis range
    _t_end = advi_walltime_cen[-1]
    _t_dadvi_end = dadvi_times_cen[-1]

    # --- Plot ---
    fig_cmp, ax_cmp = plt.subplots(2, 2, figsize=(14, 8))

    param_names = ["\u03b1", "\u03b2", "\u03c3"]
    true_vals = [2.0, 0.5, 0.5]
    dadvi_vals = [alpha_dadvi, beta_dadvi, sigma_dadvi]
    advi_vals = [advi_for_plot[:, 0], advi_for_plot[:, 1], advi_for_plot[:, 2]]

    for idx, (name, true_val) in enumerate(zip(param_names, true_vals)):
        row, col = divmod(idx, 2)
        axi = ax_cmp[row][col]
        dadvi_final = dadvi_vals[idx][-1]

        axi.plot(advi_walltime_cen, advi_vals[idx], color="steelblue", lw=0.8, alpha=0.7,
                 label=f"ADVI ({advi_walltime_cen[-1]:.1f}s, {len(advi_walltime_cen):,} iters)")

        # DADVI optimization path + horizontal continuation to end of ADVI
        axi.plot(dadvi_times_cen, dadvi_vals[idx], color="darkorange", lw=2, marker=".",
                 markersize=5)
        axi.hlines(dadvi_final, _t_dadvi_end, _t_end, color="darkorange", lw=2, ls="--",
                   label=f"DADVI ({dadvi_elapsed_total:.2f}s, {n_dadvi_iters} iters)")

        axi.axhline(true_val, color="black", ls=":", alpha=0.4, lw=1)
        axi.set_title(f"{name} convergence")
        axi.set_xlabel("Wall-clock time (s)")
        axi.set_ylabel("Value")
        axi.legend(fontsize=8)
        axi.grid(True, alpha=0.3)

    # Objective subplot
    ax_obj = ax_cmp[1][1]
    dadvi_obj_final = dadvi_obj_hist[-1]

    ax_obj.plot(advi_elbo_smooth_t, advi_elbo_smooth, color="steelblue", lw=1.5,
               label=f"ADVI ELBO (smoothed, {advi_walltime_cen[-1]:.1f}s)")
    ax_obj.plot(dadvi_times_cen, dadvi_obj_hist, color="darkorange", lw=2, marker=".",
                markersize=5)
    ax_obj.hlines(dadvi_obj_final, _t_dadvi_end, _t_end, color="darkorange", lw=2, ls="--",
                  label=f"DADVI obj ({dadvi_elapsed_total:.2f}s, {n_dadvi_iters} iters)")
    ax_obj.set_title("Objective value")
    ax_obj.set_xlabel("Wall-clock time (s)")
    ax_obj.set_ylabel("Value")
    ax_obj.set_yscale("log")
    ax_obj.legend(fontsize=8)
    ax_obj.grid(True, alpha=0.3)

    plt.suptitle("ADVI vs DADVI Convergence \u2014 Wall-Clock Time (Centered Model)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_cmp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Take-away

    - **DADVI converges automatically** (no guessing iteration counts) and agrees closely with NUTS on the *centered* model.
    - On the *uncentered* model, DADVI still suffers from the same mean-field bias as ADVI: it underestimates variances because it cannot model the correlation. **Reparameterization is essential for any mean-field VI method.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Practical recommendations

    1. **Start with NUTS** when possible. It is asymptotically exact and has great diagnostics (R-hat, ESS, divergences).
    2. **Use ADVI / DADVI** when speed matters and the model is simple:
       - Large datasets where MCMC is too slow.
       - Exploratory model building.
       - As a warm-start for NUTS (initialize with ADVI means).
    3. **Check the geometry** before trusting VI:
       - Run a short NUTS fit and inspect pair plots. Strong correlations or funnels mean mean-field VI will struggle.
       - Use `pm.sample_posterior_predictive` to check predictions — bad VI can still give reasonable predictions even if posteriors are off.
    4. **Reparameterize**:
       - Center continuous predictors.
       - Use non-centered parameterizations for hierarchical models.
       - Consider full-rank ADVI (`pm.fit(method='fullrank_advi')`) if you need some covariance structure but cannot reparameterize.
    5. **Try DADVI** from `pymc-extras` when ADVI convergence is finicky. It removes the stochastic-optimization headache while keeping the speed advantage.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
