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
def _(mo):
    mo.md(r"""
    ## What Does `pm.sample()` Give You?

    You've been calling `pm.sample()` since Session 1. You know how to specify models and check traceplots. But when something goes wrong — and it will — your ability to fix it depends on understanding what the sampler is doing and what the diagnostics are telling you has occurred, relative to what was expected.

    We'll organize around three questions:

    1. **What does `pm.sample()` give you**, and how do you read the output?
    2. **Did the sampler work?** How do you diagnose and fix problems?
    3. **Does the model fit the data?**

    Along the way, we'll build up your understanding of how the sampler works — not to implement it from scratch, but so you can reason about what's happening when diagnostics flag a problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why Sampling? The Intractable Integral

    Recall Bayes' theorem:

    $$P(\theta \mid x) = \frac{P(x \mid \theta)\, P(\theta)}{P(x)}$$

    The numerator -- likelihood times prior -- is easy to evaluate for any particular $\theta$. The difficulty is the **denominator**, also called the *evidence*:

    $$P(x) = \int_\Theta P(x \mid \theta)\, P(\theta)\, d\theta$$

    This integral sums over *all possible parameter values*. For even moderately complex models, it has no closed-form solution and is too high-dimensional for numerical quadrature. This is the fundamental computational challenge of Bayesian inference.

    **Monte Carlo integration** sidesteps the problem entirely. Instead of computing the posterior analytically, we *draw samples* from it. Any expectation under the posterior:

    $$E[h(\theta)] = \int h(\theta)\, p(\theta \mid x)\, d\theta$$

    can be approximated by a sample average:

    $$\hat{E}[h(\theta)] = \frac{1}{n}\sum_{i=1}^{n} h(\theta_i), \quad \theta_i \sim p(\theta \mid x)$$

    By the law of large numbers, this converges to the true expectation as $n \to \infty$, and the simulation error is measurable:

    $$\text{Var}(\hat{E}) = \frac{1}{n(n-1)}\sum_{i=1}^{n}(h(\theta_i) - \hat{E})^2$$

    This is why `pm.sample()` draws thousands of samples -- each one contributes to more precise estimates of posterior means, quantiles, and credible intervals.

    But this raises the key question: **how do we generate samples from the posterior** when we can't even normalize it?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Curse of Dimensionality: Why Simple Sampling Fails

    The most intuitive approach is **rejection sampling**: propose points from a simple distribution, accept those that fall under the target density. This works beautifully in 1 or 2 dimensions, but acceptance rates collapse exponentially with the number of parameters:

    | Dimensions | Approximate acceptance rate |
    |:----------:|:--------------------------:|
    | 1          | ~95%                       |
    | 10         | ~60%                       |
    | 100        | ~0.7%                      |
    | 1,000      | ~0%                        |

    Even a small mismatch between proposal and target compounds across dimensions, making nearly every proposal fall in a region of negligible density. A typical Bayesian model might have dozens to thousands of parameters, so rejection sampling is hopeless.

    This motivates **Markov chain Monte Carlo (MCMC)**: instead of proposing points independently, we construct a *sequence* of samples where each depends on the previous one. By design, this sequence converges to draws from the posterior -- even in high dimensions. The key insight is that each proposal is informed by the current position, so the sampler stays in regions of reasonable density rather than blindly guessing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A well-specified model

    Let's start with a simple model we know works — predicting penguin body mass from flipper length. Before building a model, we should always look at the data.
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


@app.cell(hide_code=True)
def _(body_mass, flipper_length, plt):
    def make_scatter_plot():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(flipper_length, body_mass / 1000, alpha=0.5, s=20)
        ax.set_xlabel("Flipper length (mm)")
        ax.set_ylabel("Body mass (kg)")
        ax.set_title("Penguin body mass vs. flipper length")
        return fig

    make_scatter_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There's a clear positive linear relationship — penguins with longer flippers tend to be heavier. There's also meaningful spread around the trend, which our model needs to capture.

    Before specifying the model, two practical choices:

    - **Standardize the predictor** (flipper length): centering and scaling makes the intercept interpretable as the mean body mass at an average flipper length, and puts the slope on a "per standard deviation" scale. This also makes it easier to choose sensible priors.
    - **Work in kilograms**: the raw data is in grams (values around 3000–6000), which would require priors on a large scale. Dividing by 1000 gives values around 3–6 kg, which are easier to reason about.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior choices

    With standardized flipper length and mass in kg, our priors become intuitive:

    - **`alpha ~ Normal(4, 2)`**: The intercept is the expected mass at average flipper length. Penguins weigh roughly 3–6 kg, so a prior centered at 4 kg with SD of 2 covers the plausible range generously.
    - **`beta ~ Normal(0, 2)`**: The slope (effect of a 1-SD change in flipper length). A prior centered at zero expresses no prior directional preference; SD of 2 allows for substantial effects.
    - **`sigma ~ HalfNormal(2)`**: The residual standard deviation. HalfNormal(2) puts most prior mass below ~4 kg of residual spread — generous for a variable that ranges over ~3 kg total.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior predictive check

    Before fitting the model, we can check whether our priors generate plausible data. A **prior predictive check** draws parameter values from the priors and simulates datasets — if these simulated datasets look nothing like real penguin data, our priors are poorly calibrated.
    """)
    return


@app.cell
def _(RANDOM_SEED, az, baseline_model, pm):
    with baseline_model:
        prior_pred = pm.sample_prior_predictive(random_seed=RANDOM_SEED)

    az.plot_ppc_dist(prior_pred, group="prior_predictive", num_samples=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The prior predictive distribution covers a wide range of body masses — some implausible (negative masses, masses above 10 kg) but with most of the density in a reasonable range. This tells us our priors are weakly informative: they don't force the model toward any particular answer, but they don't generate data that are off by orders of magnitude. That's exactly what we want for a first pass.

    Now let's fit the model.
    """)
    return


@app.cell
def _(RANDOM_SEED, baseline_model, pm):
    with baseline_model:
        baseline_trace = pm.sample(random_seed=RANDOM_SEED)
    baseline_trace
    return (baseline_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The DataTree object

    Since PyMC 6.0 and ArViz 1.0 we now return `DataTree` instead of `InferenceData`

    The result of `pm.sample()` is an `xarray.DataTree` object — a hierarchical container that holds everything about the sampling run.
    """)
    return


@app.cell(hide_code=True)
def _(baseline_trace):
    baseline_trace
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `posterior` group contains the actual parameter draws — this is what you'll use for inference. It's an xarray Dataset with dimensions `(chain, draw)` for each parameter.
    """)
    return


@app.cell(hide_code=True)
def _(baseline_trace):
    baseline_trace.posterior
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `sample_stats` group contains per-draw sampler diagnostics: step sizes, divergence flags, energy values, tree depth, and more. This is what convergence diagnostics examine.
    """)
    return


@app.cell(hide_code=True)
def _(baseline_trace):
    baseline_trace.sample_stats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each variable in `sample_stats` records something about the sampler's behavior at each draw:

    | Statistic | What it records |
    |-----------|----------------|
    | **step_size** | The leapfrog integration step size used for this draw |
    | **tree_depth** | How many doubling steps NUTS took to build the trajectory (deeper = longer trajectory) |
    | **n_steps** | Total number of leapfrog steps in the trajectory ($2^{\text{tree\_depth}} - 1$) |
    | **diverging** | `True` if the leapfrog trajectory diverged (numerical error exceeded threshold) |
    | **energy** | The Hamiltonian energy at the end of the trajectory (used for BFMI) |
    | **acceptance_rate** | Metropolis acceptance probability for this draw (NUTS targets ~0.8 by default) |

    These become important when diagnosing problems -- divergences, max tree depth warnings, and low acceptance rates all show up here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `observed_data` group stores the data you conditioned on. ArviZ uses this for posterior predictive checks and leave-one-out cross-validation.
    """)
    return


@app.cell(hide_code=True)
def _(baseline_trace):
    baseline_trace.observed_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summarizing the Posterior Samples

    The `az.summary()` function gives you the most important information in one table. Let's walk through each column.
    """)
    return


@app.cell
def _(az, baseline_trace):
    az.summary(baseline_trace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the summary table

    | Column | What it tells you |
    |--------|-------------------|
    | **mean** | Point estimate (posterior mean) |
    | **sd** | Posterior standard deviation — uncertainty in the parameter |
    | **eti_5.5%** / **eti_94.5%** | 89% Equal-Tailed Interval — the interval with equal probability in each tail containing 89% of the posterior |
    | **mcse_mean** | Monte Carlo Standard Error of the mean — how much the *estimate* of the mean would change with different random draws |
    | **mcse_sd** | MCSE for the standard deviation |
    | **ess_bulk** | Effective sample size for the bulk (center) of the posterior |
    | **ess_tail** | Effective sample size for the tails (5% and 95% quantiles) |
    | **r_hat** | Split R-hat convergence diagnostic — values near 1.0 indicate convergence |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Equal-Tailed Intervals

    Two columns in the summary table deserve a closer look: `eti_5.5%` and `eti_94.5%`. The **Equal-Tailed Interval** (ETI) is computed by cutting off equal probability in each tail of the posterior distribution. For an 89% ETI, we exclude 5.5% from each tail.

    The ETI is the default in ArviZ because it is straightforward to interpret and compute. An alternative is the **Highest Density Interval** (HDI), which finds the *narrowest* interval containing the specified probability mass. For **skewed posteriors**, the HDI may be preferable since it always contains the most probable values — but for symmetric posteriors (like the ones here), the two are nearly identical.

    The 89% default probability follows a convention that avoids the false precision of "95%" while providing a useful credible interval. It also has the practical benefit of lower variability in summary statistics compared to wider intervals.

    `az.plot_dist()` shows the posterior density, making it easy to see the shape of the distribution at a glance.
    """)
    return


@app.cell(hide_code=True)
def _(az, baseline_trace):
    az.plot_dist(baseline_trace, var_names=["alpha", "beta", "sigma"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also extract ETI values programmatically:
    """)
    return


@app.cell
def _(az, baseline_trace):
    az.eti(baseline_trace, prob=0.89)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Did the Sampler Work?

    Having seen what the output looks like, let's now ask: can we trust it? The summary table showed healthy-looking numbers, but to understand *why* they're healthy — and to recognize when they're not — we need to understand what the sampler is doing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the Sampler Works: From Random Walks to NUTS

    When you call `pm.sample()`, PyMC doesn't draw independent samples from the posterior — it can't, because the posterior is a complex, high-dimensional distribution we only know up to a normalizing constant.

    Instead, it constructs a **Markov chain**: a sequence of samples where each draw depends on the previous one, particularly designed so that after enough steps, the samples approximate draws from the posterior. At least, this is what theory guarantees.

    This is the core idea behind **Markov chain Monte Carlo (MCMC)**. To understand the diagnostics we'll use to evaluate sampler output, let's take a look at a specific MCMC algorithm.
    """)
    return


@app.cell(hide_code=True)
def _(Path, base64, mo):
    def load_image():
        metro_path = Path(__file__).parent / "images" / "Metropolis.png"
        if metro_path.exists():
            metro_b64 = base64.b64encode(metro_path.read_bytes()).decode()
            return (
                f'<img src="data:image/png;base64,{metro_b64}" style="max-width:100%;">'
            )
        return ""

    metro_img = load_image()

    mo.md(f"""
    #### Random Walk Metropolis

    The simplest MCMC algorithm works like this:

    1. Start at some position in parameter space
    2. **Propose** a small random step from the current position
    3. **Accept** the proposal if it lands in a region of higher (or comparable) posterior density; otherwise, **reject** it and stay put
    4. Repeat

    {metro_img}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The animation below shows this process in action. The left panel shows the sampler moving through parameter space —
    <span style="color:#2ca02c">green</span> lines are accepted proposals,
    <span style="color:#d62728">red</span> lines are rejected ones.

    Watch how the **trace** (center) and **marginal distribution** (right)
    build up sample by sample. By the end, the bottom panels look just like what `az.plot_trace_dist()` produces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The critical tuning parameter is the **step size** — how far each proposal jumps. This single number controls the fundamental tradeoff of the algorithm:

    - **Too small**: Every proposal is accepted (it barely moved), but the chain crawls — it takes thousands of steps to cross the posterior. The trace looks like a slow, smooth random walk.
    - **Too large**: Most proposals land in low-density regions and get rejected. The chain gets stuck for long stretches, jumping only occasionally.
    - **Just right**: A mix of accepted and rejected proposals. The trace looks like a "fuzzy caterpillar" — exactly what we want.

    Try manually adjusting the step size below:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how the step size controls everything about the sampler's behavior — the acceptance rate, the autocorrelation, and how quickly the chain explores the target distribution. This is what PyMC's **warmup phase** automates: finding the step size that produces that well-mixed caterpillar trace. When you call `pm.sample()`, the first `tune` draws (default: 1000) are a **warmup phase** where the sampler adapts to the posterior geometry.

    `pm.sample(tune=500)`

    These draws are discarded and never appear in your trace, because the sampler's behavior is non-stationary while it's still learning.

    If you see poor mixing in the early post-warmup draws, the warmup may not have been long enough. Try increasing `tune` (e.g., `tune=2000`). Complex models — hierarchical structures, many parameters, difficult geometry — often need more warmup to get both the step size and mass matrix right. We'll return to these settings in the sampler configuration tips later, when we look at what to do when sampling fails.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### The problem with Metropolis

    But even with optimal tuning, Metropolis has a fundamental limitation: because each step is a *random* perturbation, successive samples are highly correlated. The chain takes many small steps to traverse the posterior, so you need a very long chain to get a modest number of effectively independent samples. And in high dimensions, this problem gets dramatically worse.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Gibbs Sampling

    An alternative to Metropolis is **Gibbs sampling**, which cycles through parameters one at a time, sampling each from its *conditional* distribution given the current values of all other parameters:

    $$\theta_1^{(j)} \sim \pi(\theta_1 \mid \theta_2^{(j-1)}, \theta_3^{(j-1)}, \ldots, \theta_k^{(j-1)})$$
    $$\theta_2^{(j)} \sim \pi(\theta_2 \mid \theta_1^{(j)}, \theta_3^{(j-1)}, \ldots, \theta_k^{(j-1)})$$
    $$\vdots$$
    $$\theta_k^{(j)} \sim \pi(\theta_k \mid \theta_1^{(j)}, \theta_2^{(j)}, \ldots, \theta_{k-1}^{(j)})$$

    When conjugate priors are available, each conditional has a known closed form -- making Gibbs sampling exact and efficient. The classic example is the **coal mining disasters** change-point model (which we'll use in the capstone exercise), where Gamma-Poisson conjugacy gives closed-form conditionals for each rate parameter.

    However, Gibbs sampling has important limitations:

    - It requires **known conditional distributions** -- hard to automate for arbitrary models
    - It updates one parameter at a time, making it **slow when parameters are correlated** (it can only move along coordinate axes)
    - It doesn't use gradient information, so it shares Metropolis's scaling problems in high dimensions

    PyMC uses a variant called `CategoricalGibbsMetropolis` for discrete variables, but for continuous parameters, gradient-based methods are far more efficient.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hamiltonian Monte Carlo and NUTS

    **Hamiltonian Monte Carlo (HMC)** solves the efficiency problem by using the *gradient* of the log-posterior to make informed proposals. Instead of random steps, HMC simulates a physical system: imagine placing a ball on a surface shaped like the negative log-posterior and giving it a random push. The ball rolls along the surface following Hamiltonian dynamics, naturally staying in high-density regions while covering large distances.

    More precisely, HMC augments the parameter space with auxiliary **momentum** variables $\phi$ and defines a Hamiltonian:

    $$\mathcal{H}(\theta, \phi) = \underbrace{-\log p(\theta \mid x)}_{\text{potential energy } U(\theta)} + \underbrace{\frac{1}{2}\phi^T M^{-1} \phi}_{\text{kinetic energy } K(\phi)}$$

    where $M$ is the **mass matrix** (estimated during warmup). The sampler alternates between:

    1. **Resampling momentum**: draw $\phi \sim \mathcal{N}(0, M)$ (a random "push")
    2. **Simulating dynamics**: follow the Hamiltonian trajectory using the **leapfrog integrator**:

    $$\phi_{t+\epsilon/2} = \phi_t - \frac{\epsilon}{2}\nabla_\theta U(\theta_t)$$
    $$\theta_{t+\epsilon} = \theta_t + \epsilon\, M^{-1}\phi_{t+\epsilon/2}$$
    $$\phi_{t+\epsilon} = \phi_{t+\epsilon/2} - \frac{\epsilon}{2}\nabla_\theta U(\theta_{t+\epsilon})$$

    3. **Metropolis correction**: accept or reject the endpoint with probability $\min(1, \exp(-\mathcal{H}(\theta', \phi') + \mathcal{H}(\theta, \phi)))$, which corrects for discretization error in the leapfrog integrator.

    **NUTS** (the No-U-Turn Sampler) extends HMC by automatically choosing how far to "roll" — it stops the trajectory when it starts doubling back, which is the "no U-turn" criterion. This eliminates HMC's most sensitive tuning parameter (trajectory length).

    NUTS adapts two things during warmup:

    - **Step size** ($\epsilon$), via dual averaging — targeting the acceptance rate set by `target_accept` (default 0.8). This is the automated version of what the slider above let you do manually.
    - **Mass matrix** ($M$, also called the inverse metric) — an estimate of the posterior covariance that lets the sampler take appropriately-scaled steps in each direction. Without it, a parameter with SD = 0.01 and one with SD = 100 would need very different step sizes. The mass matrix handles this automatically.

    The result: proposals that are *distant* from the current position but still in high-density regions, producing nearly independent samples. Let's see the contrast directly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The animations below show how HMC differs from MCMC on various distributions.

    [HMC Animations](https://chi-feng.github.io/mcmc-demo/app.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This contrast explains why PyMC defaults to NUTS for continuous parameters. Let's verify this on our penguin model by running the same model with Metropolis and comparing directly.
    """)
    return


@app.cell
def _(RANDOM_SEED, baseline_model, pm):
    with baseline_model:
        metropolis_trace = pm.sample(
            step=pm.Metropolis(),
            random_seed=RANDOM_SEED,
        )
    metropolis_trace
    return (metropolis_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that `pm.sample` drew **multiple chains** by default. MCMC is *embarassingly parallel* so we can easily use several computer cores to sample faster, in parallel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's compare the autocorrelation — how correlated successive draws are — for both samplers on the same model.
    """)
    return


@app.cell(hide_code=True)
def _(az, baseline_trace, metropolis_trace):
    az.plot_autocorr(
        baseline_trace,
        var_names=["alpha", "beta", "sigma"],
    )
    az.plot_autocorr(
        metropolis_trace,
        var_names=["alpha", "beta", "sigma"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The difference in effective sample size tells the same story numerically. NUTS produces nearly independent draws; Metropolis wastes most of its computation on correlated samples.
    """)
    return


@app.cell
def _(az, baseline_trace, metropolis_trace, mo):
    nuts_summary = az.summary(baseline_trace, var_names=["alpha", "beta", "sigma"])
    metro_summary = az.summary(metropolis_trace, var_names=["alpha", "beta", "sigma"])

    _output = mo.md(f"""
    **NUTS ESS (bulk):**

    {nuts_summary[["ess_bulk", "ess_tail"]].to_markdown()}

    **Metropolis ESS (bulk):**

    {metro_summary[["ess_bulk", "ess_tail"]].to_markdown()}
    """)
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When ESS is much lower than expected, something about the posterior geometry is preventing efficient exploration — we'll see concrete examples once we look at what to do when sampling fails.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visual Diagnostics

    The visual diagnostic tools in ArviZ let you inspect the sampler's behavior directly. We'll look at four complementary views, all applied to our healthy baseline model.
    """)
    return


@app.cell(hide_code=True)
def whats_new_in_arviz_1(mo):
    mo.md(r"""
    > **What changed in ArviZ 1.0**
    >
    > The diagnostics here use the ArviZ 1.0 API. A few things to know if you have older notebooks lying around:
    >
    > - The default credible-interval summary is now the 89% **ETI** (equal-tailed interval), not the 94% HDI. `az.summary` and `az.eti` reflect this.
    > - `plot_trace` is now `plot_trace_dist`, and `plot_rank` is now `plot_rank_dist`. The `_dist` suffix marks the redesigned versions that combine trace and density in one view.
    > - `InferenceData` is now backed by `xarray.DataTree`, which means groups can nest arbitrarily and you can compose multiple traces into one tree.
    > - **WAIC has been removed** in favor of PSIS-LOO-CV. Use `az.loo()` and `az.compare()` for model comparison (we will get to that in the LOO section).
    > - There is a new `arviz-plots` library underneath; you can opt into the plotly or bokeh backend if you prefer those to matplotlib.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Trace Plots

    The trace plot shows two things side by side: the posterior distribution (left) and the raw draws over time (right). You want to see "fuzzy caterpillars" — chains that mix well and overlap completely. `plot_trace_dist` combines the trace and distribution density in one view.
    """)
    return


@app.cell(hide_code=True)
def _(az, baseline_trace):
    az.plot_trace_dist(baseline_trace, var_names=["alpha", "beta", "sigma"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Rank Plots

    Rank plots (`plot_rank`) are often better than raw trace plots for detecting convergence problems. They work by ranking *all* draws across all chains (pooling them), converting to fractional ranks, then computing the empirical CDF of those ranks for each chain.

    The plot shows the **$\Delta$-ECDF** -- the difference between each chain's observed rank ECDF and the expected uniform CDF. If chains are sampling from the same distribution, the lines should be flat near zero, staying within the gray envelope. Lines that extend outside the envelope or show "squared-off" patterns indicate convergence problems or low ESS.

    **Why uniform ranks indicate good mixing:** If all chains are drawing from the same distribution, then pooling and ranking all draws should give each chain an equal share of low, medium, and high ranks. The rank distribution within each chain should be approximately uniform -- no chain should consistently produce higher or lower values than the others. When ranks *are* uniform across chains, the $\Delta$-ECDF stays near zero. When one chain is stuck in a different region, its ranks will cluster (e.g., all high), producing a visible departure from the envelope.
    """)
    return


@app.cell(hide_code=True)
def _(az, baseline_trace):
    az.plot_rank_dist(baseline_trace, var_names=["alpha", "beta", "sigma"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Numerical Diagnostics

    The visual diagnostics above all look healthy. Let's now attach numbers to these observations. We'll introduce each diagnostic briefly here — showing what "good" looks like — and then explore what "bad" looks like later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### R-hat ($\hat{R}$)

    *How do we know the chains have converged to the same distribution?*

    We ran 4 independent chains. If they've all found the posterior, they should agree. R-hat quantifies this by comparing variance *between* chains to variance *within* chains. Values near 1.0 mean agreement. **Threshold: R-hat < 1.01.**

    We'll see exactly what high R-hat looks like — and why it happens — once we look at sampling failures.
    """)
    return


@app.cell
def _(az, baseline_trace):
    az.rhat(baseline_trace, var_names=["alpha", "beta", "sigma"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Monte Carlo Standard Error (MCSE)

    *How precise are our posterior summaries?*

    Even with good mixing, finite samples mean our estimates of the mean, SD, and quantiles have some Monte Carlo error. MCSE quantifies this: it is the standard deviation of the estimate divided by the square root of the effective sample size:

    $$\text{MCSE} = \frac{\text{SD}}{\sqrt{\text{ESS}}}$$

    This tells you how much the *estimate* of the mean would change if you ran the sampler again with a different random seed. **Rule of thumb:** if MCSE / posterior SD > 0.1, you need more draws before trusting summaries. You can also think of MCSE as the number of decimal places you can trust -- with MCSE of about 0.002, reporting to two decimal places is justified.
    """)
    return


@app.cell
def _(az, baseline_trace):
    az.mcse(baseline_trace, var_names=["alpha", "beta", "sigma"])
    return


@app.cell(hide_code=True)
def _(az, baseline_trace):
    az.plot_mcse(baseline_trace, var_names=["alpha", "beta", "sigma"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### A Note on Thinning

    You may read older advice to "thin" your chains — keeping every $n$th draw to reduce autocorrelation. Modern practice generally favors **collecting more samples** over thinning, since thinning discards information. NUTS already produces low-autocorrelation draws.

    The one case where thinning makes sense is **storage**: if you have very many draws and only need rough summaries. ArviZ provides `az.thin()` for this:
    """)
    return


@app.cell
def _(az, baseline_trace, mo):
    thinned = az.thin(baseline_trace, factor="auto")
    original_ess = az.ess(baseline_trace, var_names=["beta"])["beta"].item()
    thinned_ess = az.ess(thinned["beta"]).item()

    # ESS barely changes — thinning doesn't create new information
    _output = mo.md(f"""
    - Original draws: {baseline_trace.posterior.sizes["draw"]}
    - Thinned draws: {thinned.sizes["draw"]}
    - Original beta ESS: {original_ess:.0f}
    - Thinned beta ESS: {thinned_ess:.0f}
    """)
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Energy and BFMI

    *Is the sampler exploring the full posterior, or stuck in one region?*

    This is an HMC-specific diagnostic. `az.plot_energy()` overlays two distributions — the marginal energy across all samples and the energy transition between successive samples. If the sampler explores efficiently, these should overlap well. BFMI (Bayesian Fraction of Missing Information) quantifies the overlap. **Values below 0.3 are concerning** — they indicate the sampler can't move freely between energy levels.
    """)
    return


@app.cell(hide_code=True)
def _(az, baseline_trace, plt):
    def make_energy_plot():
        pc = az.plot_energy(baseline_trace, visuals={"legend": False})
        fig = plt.gcf()
        bfmi_ax, energy_ax = (fig.axes[0], fig.axes[1])
        n_chains = baseline_trace.posterior.sizes["chain"]
        bfmi_ax.set_yticks(range(n_chains))
        bfmi_ax.set_ylim(-0.5, n_chains - 0.5)
        for coll in bfmi_ax.collections:
            coll.set_sizes([80])
        bfmi_ax.set_position([0.05, 0.15, 0.15, 0.75])
        energy_ax.set_position([0.28, 0.15, 0.5, 0.75])
        for line, label in zip(energy_ax.get_lines(), ["marginal", "transition"]):
            line.set_label(label)
        energy_ax.legend(loc="upper right")
        return pc

    make_energy_plot()
    return


@app.cell
def _(az, baseline_trace):
    az.bfmi(baseline_trace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Effective Sample Size (ESS)

    We saw ESS in the NUTS vs. Metropolis comparison above. Two variants matter:

    - **Bulk ESS**: How well the chain explores the center of the distribution
    - **Tail ESS**: How well it explores the extremes (5th and 95th percentiles) — often lower than bulk ESS

    Rule of thumb: you want at least **400 total effective samples** (across all chains) for reliable estimates.

    The `plot_ess` function shows ESS across different quantiles, helping identify whether the tails are as well-explored as the center. The `plot_ess_evolution` function shows ESS as a function of the number of draws.
    """)
    return


@app.cell(hide_code=True)
def _(az, baseline_trace):
    az.plot_ess(baseline_trace, var_names=["alpha", "beta", "sigma"], kind="quantile")
    return


@app.cell(hide_code=True)
def _(az, baseline_trace):
    az.plot_ess_evolution(baseline_trace, var_names=["alpha", "beta", "sigma"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## When Sampling Fails

    The diagnostics above all looked healthy. That's because our baseline model is well-specified. Now let's see what happens when things go wrong — and learn to recognize and fix the problems.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Case 1: Non-identifiability — Poor Mixing, High R-hat

    *What happens when the model has redundant parameters?*

    Let's predict penguin body mass using species indicators — but we'll deliberately overparameterize the model by including an intercept *and* indicators for all three species (no reference category).
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
def _(RANDOM_SEED, adelie, as_model, body_mass_kg, chinstrap, gentoo, pm):
    @as_model()
    def _overparam():
        beta_0 = pm.Normal("intercept", mu=0, sigma=100000)
        beta_adelie = pm.Normal("beta_adelie", mu=0, sigma=100000)
        beta_chinstrap = pm.Normal("beta_chinstrap", mu=0, sigma=100000)
        beta_gentoo = pm.Normal("beta_gentoo", mu=0, sigma=100000)
        sigma = pm.HalfNormal("sigma", sigma=2)
        mu = (
            beta_0
            + beta_adelie * adelie
            + beta_chinstrap * chinstrap
            + beta_gentoo * gentoo
        )
        pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)

    with _overparam():
        overparam_trace = pm.sample(2000, random_seed=RANDOM_SEED)
    overparam_trace
    return (overparam_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Diagnosing the problem

    Let's look at the summary table first.
    """)
    return


@app.cell
def _(az, overparam_trace):
    az.summary(
        overparam_trace,
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
def _(mo):
    mo.md(r"""
    Look at the R-hat values for the intercept and species coefficients — they're well above 1.01, and the ESS values have collapsed. But `sigma` looks fine. What's going on?

    The trace plots make the problem visible.
    """)
    return


@app.cell(hide_code=True)
def _(az, overparam_trace):
    az.plot_trace_dist(
        overparam_trace,
        var_names=["intercept", "beta_adelie", "beta_chinstrap", "beta_gentoo"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The chains are wandering without settling — they explore different regions of parameter space and never agree on where to stop. This is exactly the scenario **R-hat** is designed to detect.

    Recall that R-hat compares the variance *between* chains to the variance *within* each chain. When chains are exploring different regions (as here), the between-chain variance is large relative to within-chain variance, and R-hat rises above 1. **Split R-hat** goes further by also splitting each chain in half, catching non-stationarity *within* a single chain — if a chain drifts over the course of sampling, the two halves will disagree.

    The rank plot makes this even clearer — well-mixed chains would show flat Δ-ECDF lines near zero within the gray envelope, but here the chains' rank distributions diverge substantially.
    """)
    return


@app.cell(hide_code=True)
def _(az, overparam_trace):
    az.plot_rank_dist(
        overparam_trace, var_names=["intercept", "beta_adelie", "beta_chinstrap"]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A pair plot reveals the geometry of the problem.
    """)
    return


@app.cell(hide_code=True)
def _(az, overparam_trace):
    az.plot_pair(
        overparam_trace,
        var_names=["intercept", "beta_adelie", "beta_chinstrap"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Why this happens

    The model has infinitely many parameter combinations that produce the same predictions. For example, shifting the intercept up by 1 and all species effects down by 1 gives identical fitted values. The posterior is a *ridge* (a flat direction in parameter space), not a peak. The sampler slides along this ridge without converging — the chains never agree on where on the ridge to settle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Fix 1: Use reference coding

    Drop one species indicator so the intercept absorbs that species' mean. This removes the redundancy.
    """)
    return


@app.cell
def _(RANDOM_SEED, as_model, body_mass_kg, chinstrap, gentoo, pm):
    @as_model()
    def _reference():
        alpha = pm.Normal("alpha", mu=4, sigma=2)
        beta_chinstrap = pm.Normal("beta_chinstrap", mu=0, sigma=2)
        beta_gentoo = pm.Normal("beta_gentoo", mu=0, sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=2)
        mu = alpha + beta_chinstrap * chinstrap + beta_gentoo * gentoo
        pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)

    with _reference():
        reference_trace = pm.sample(random_seed=RANDOM_SEED)
    reference_trace
    return (reference_trace,)


@app.cell
def _(az, reference_trace):
    az.summary(reference_trace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Fix 2: Use informative priors

    Alternatively, keep all indicators but use priors that regularize. Informative priors constrain the ridge into a proper peak by "pulling" parameters toward specific values.
    """)
    return


@app.cell
def _(RANDOM_SEED, adelie, as_model, body_mass_kg, chinstrap, gentoo, pm):
    @as_model()
    def _regularized():
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

    with _regularized():
        regularized_trace = pm.sample(random_seed=RANDOM_SEED)
    regularized_trace
    return (regularized_trace,)


@app.cell
def _(az, regularized_trace):
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
def _(mo):
    mo.callout(
        mo.md(
            "**Takeaway:** Non-identifiability is a *model specification* problem that manifests as a *sampling* problem. Diagnostics catch it; the fix is in the model, not the sampler."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Case 2: Divergences — Posterior Geometry Problems

    *What happens when the posterior has difficult geometry?*

    Divergences are one of the most important diagnostics in Bayesian modeling. They are NUTS telling you: "my numerical trajectory was inaccurate here." This means the sampler is systematically *under-exploring* some region of the posterior, which **biases your inference** — not just makes it noisier.

    To understand them clearly, we'll demonstrate with Neal's funnel — a textbook example of problematic geometry — then connect it to hierarchical models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Neal's funnel** is a textbook example of problematic posterior geometry.
    The parameter `v` controls the scale, and `x` values are drawn with that scale.
    When `v` is very negative, all `x` values must be near zero — creating a funnel shape.
    """)
    return


@app.cell
def _(RANDOM_SEED, as_model, pm):
    @as_model()
    def _centered_funnel():
        v = pm.Normal("v", mu=0, sigma=3)
        pm.Normal("x", mu=0, sigma=pm.math.exp(v / 2), shape=9)

    with _centered_funnel() as centered_funnel:
        centered_trace = pm.sample(1000, random_seed=RANDOM_SEED, target_accept=0.8)
    centered_trace
    return (centered_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Diagnosing the problem

    The sampling output above should report divergent transitions.
    """)
    return


@app.cell
def _(centered_trace, mo):
    divergences = centered_trace.sample_stats["diverging"].values.sum()
    _output = mo.md(f"Number of divergent transitions: **{divergences}**")
    _output
    return


@app.cell(hide_code=True)
def _(az, centered_trace):
    az.plot_trace_dist(centered_trace, var_names=["v"])
    return


@app.cell(hide_code=True)
def _(az, centered_trace):
    az.plot_pair(
        centered_trace,
        var_names=["v", "x"],
        coords={"x_dim_0": [0, 1]},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To see the funnel clearly, let's plot `v` against one of the `x` values, highlighting divergent samples. We'll also compare against exact samples from the funnel distribution to see what the sampler is missing.
    """)
    return


@app.cell(hide_code=True)
def _(centered_trace, np, plt):
    def make_funnel_plot():
        posterior = centered_trace.posterior
        v_draws = posterior["v"].values.flatten()
        x_0_draws = posterior["x"].sel(x_dim_0=0).values.flatten()
        diverging = centered_trace.sample_stats["diverging"].values.flatten()

        rng = np.random.default_rng(42)
        v_exact = rng.normal(0, 3, size=5000)
        x_exact = rng.normal(0, np.exp(v_exact / 2))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        ax1.scatter(v_exact, x_exact, s=2, alpha=0.15, color="C0")
        ax1.set_xlabel("v")
        ax1.set_ylabel("x[0]")
        ax1.set_ylim(-10, 10)
        ax1.set_xlim(-10, 10)
        ax1.set_title("Exact samples from the funnel")

        ax2.scatter(
            v_draws[~diverging],
            x_0_draws[~diverging],
            s=3,
            alpha=0.2,
            label="Non-divergent",
        )
        ax2.scatter(
            v_draws[diverging],
            x_0_draws[diverging],
            s=5,
            color="red",
            label="Divergent",
        )
        ax2.set_xlabel("v")
        ax2.set_ylim(-10, 10)
        ax2.set_xlim(-10, 10)
        ax2.set_title("MCMC samples (centered parameterization)")
        ax2.legend()

        fig.suptitle(
            "Neal's funnel: MCMC fails to explore the narrow neck (v < -2)", fontsize=12
        )
        fig.tight_layout()
        return fig

    make_funnel_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The **parallel coordinates plot** shows each sample as a line connecting its value across parameters. Lines are colored by divergence status. If divergent draws (highlighted) cluster at extreme values of `v`, this confirms the funnel neck is the problem region.
    """)
    return


@app.cell(hide_code=True)
def _(az, centered_trace):
    def make_parallel_plot():
        pc = az.plot_parallel(
            centered_trace, var_names=["v", "x"], coords={"x_dim_0": [0, 1, 2]}
        )
        pc.get_viz("plot").set_ylim(-100, 100)
        return pc

    make_parallel_plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Why this happens

    When `v` is very negative, `exp(v/2)` is tiny, so all `x` values must be very close to zero. When `v` is large, the `x` values can spread widely. This creates a "funnel" shape — wide at the top, narrow at the bottom.

    The sampler's step size is tuned for the *wide* part of the funnel. In the narrow neck, this step size is too large, causing the numerical simulation of Hamiltonian dynamics to be inaccurate. **Divergences are NUTS telling you: "my trajectory was wrong here."**

    This matters because the sampler systematically *under-explores* the funnel's neck, biasing your estimates. The **energy plot** and **BFMI** capture this from a different angle — when the sampler can't move freely between the wide and narrow parts of the posterior, the energy transition distribution won't match the marginal energy distribution.

    **Connection to hierarchical models:** This is exactly what happens when group-level variance can approach zero. In a hierarchical model, if `sigma_group` is near zero, all group effects must be nearly identical — creating the same funnel geometry. The centered parameterization `alpha ~ Normal(mu, sigma)` creates this coupling; non-centered `alpha = mu + z * sigma` breaks it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Fix: Non-centered parameterization

    Instead of `x ~ Normal(0, exp(v/2))`, we write `x = z * exp(v/2)` where `z ~ Normal(0, 1)`. This decouples `x` from `v`, transforming the funnel into a shape the sampler can navigate easily.
    """)
    return


@app.cell
def _(RANDOM_SEED, as_model, pm):
    @as_model()
    def _noncentered_funnel():
        v = pm.Normal("v", mu=0, sigma=3)
        z = pm.Normal("z", mu=0, sigma=1, shape=9)
        pm.Deterministic("x", z * pm.math.exp(v / 2))

    with _noncentered_funnel():
        noncentered_trace = pm.sample(1000, random_seed=RANDOM_SEED)
    noncentered_trace
    return (noncentered_trace,)


@app.cell
def _(az, mo, noncentered_trace):
    divergences_fixed = noncentered_trace.sample_stats["diverging"].values.sum()
    mo.md(f"Divergences (non-centered): {divergences_fixed}")
    az.summary(noncentered_trace, var_names=["v"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""**Partial fix:** You can also try increasing `target_accept` (e.g., 0.95 or 0.99), which uses a smaller step size. This reduces but may not eliminate divergences:
    ```python
    pm.sample(1000, target_accept=0.99)
    ```"""),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**Takeaway:** Divergences always warrant investigation. They indicate regions of the posterior the sampler can't explore accurately, which means your inference is *biased*, not just noisy."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Case 3: Inefficient Sampling — Low ESS, High Autocorrelation

    *What happens when the sampler can't explore the posterior efficiently?*

    Cases 1 and 2 were model specification problems that the sampler exposed. But sometimes the sampler itself is the bottleneck — either because a less efficient algorithm is being used, or because the model structure forces it.

    We already saw a preview of this when we compared NUTS to Metropolis earlier. Now let's see a more realistic scenario: a model where **PyMC automatically uses Metropolis for some parameters** because they're discrete.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### A model with discrete and continuous parameters

    Let's predict body mass using a latent group indicator — a discrete variable that PyMC can't sample with NUTS. PyMC will automatically assign Metropolis to the discrete parameter and NUTS to the continuous ones.
    """)
    return


@app.cell
def _(RANDOM_SEED, as_model, body_mass_kg, np, pm):
    @as_model()
    def _mixed_sampler():
        group = pm.Categorical("group", p=np.ones(3) / 3, shape=len(body_mass_kg))
        mu_groups = pm.Normal("mu_groups", mu=4, sigma=2, shape=3)
        sigma = pm.HalfNormal("sigma", sigma=2)
        pm.Normal("mass", mu=mu_groups[group], sigma=sigma, observed=body_mass_kg)

    with _mixed_sampler():
        mixed_trace = pm.sample(2000, random_seed=RANDOM_SEED)
    mixed_trace
    return (mixed_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the sampling output — PyMC reports using both **NUTS** (for continuous parameters) and **CategoricalGibbsMetropolis** (for the discrete `group` variable). Let's compare the ESS.
    """)
    return


@app.cell
def _(az, mixed_trace):
    az.summary(
        mixed_trace, var_names=["mu_groups", "sigma", "group"], filter_vars="like"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The continuous parameters (`mu_groups`, `sigma`) have high ESS — NUTS handles them efficiently. But the discrete `group` parameter has dramatically lower ESS — the Metropolis sampler for discrete variables produces highly correlated draws.

    This is the **ESS in action**. When you see ESS much lower than your nominal draw count, it means the sampler is producing correlated draws in that region of the model. The autocorrelation plot tells the same story: the discrete parameter's autocorrelation decays slowly compared to the continuous ones.

    **When might you encounter this in practice?**

    - Models with discrete latent variables (mixture models, change-point models)
    - Models where PyMC falls back to Metropolis for some parameters
    - Models with highly correlated posteriors even for continuous parameters

    The fix depends on the cause: for discrete parameters, consider whether you can marginalize them out; for correlated posteriors, try reparameterization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            "**Takeaway:** The choice of sampler matters. NUTS is dramatically more efficient than Metropolis for continuous parameters. When you see low ESS, check what sampler is being used and whether you can reformulate the model to help it."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Marginalizing discrete latents automatically

    For mixture models, change-point models, and other situations with discrete latent variables, PyMC v6 ships an automatic-marginalization helper in `pymc_extras` that rewrites the model with the discrete variables summed out. The resulting model has only continuous parameters, so NUTS samples it cleanly. The discrete posterior can be recovered afterwards.

    We will demonstrate on a tiny two-component Gaussian mixture.
    """)
    return


@app.cell
def mixture_data(np):
    mixture_rng = np.random.default_rng(0)
    mixture_n = 200
    mixture_true_p = 0.4
    mixture_z_true = mixture_rng.binomial(1, mixture_true_p, size=mixture_n)
    mixture_y = mixture_rng.normal(
        loc=np.where(mixture_z_true == 1, 2.0, -2.0), scale=1.0
    )
    return mixture_n, mixture_y


@app.cell
def mixture_explicit(RANDOM_SEED, as_model, mixture_n, mixture_y, pm):
    @as_model()
    def _mixture_explicit():
        p = pm.Beta("p", 1, 1)
        mu = pm.Normal("mu", mu=[-2, 2], sigma=2, shape=2)
        z = pm.Bernoulli("z", p=p, shape=mixture_n)
        pm.Normal("y", mu=mu[z], sigma=1, observed=mixture_y)

    with _mixture_explicit():
        mixture_explicit_trace = pm.sample(
            draws=500, tune=500, chains=2, progressbar=False, random_seed=RANDOM_SEED
        )
    mixture_explicit_trace
    return (mixture_explicit_trace,)


@app.cell(hide_code=True)
def _(az, mixture_explicit_trace):
    explicit_summary = az.summary(
        mixture_explicit_trace, var_names=["p", "mu"], round_to=3
    )
    explicit_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the **ess_bulk** column for `mu` and `p` is small relative to the 1000 draws we took, and `r_hat` is often noticeably above 1.00. That's the Metropolis-on-`z` step dragging the continuous parameters down with it.

    Now the same model with `z` marginalized out by `pymc_extras.marginalize`. The model loses `z`, and `pm.sample` falls back to pure NUTS:
    """)
    return


@app.cell
def mixture_marginalized(RANDOM_SEED, as_model, mixture_n, mixture_y, pm):
    from pymc_extras import marginalize

    @as_model()
    def _mixture_base():
        p = pm.Beta("p", 1, 1)
        mu = pm.Normal("mu", mu=[-2, 2], sigma=2, shape=2)
        z = pm.Bernoulli("z", p=p, shape=mixture_n)
        pm.Normal("y", mu=mu[z], sigma=1, observed=mixture_y)

    mixture_base_model = _mixture_base()
    mixture_marginal_model = marginalize(mixture_base_model, ["z"])

    with mixture_marginal_model:
        mixture_marginal_trace = pm.sample(
            draws=500, tune=500, chains=2, progressbar=False, random_seed=RANDOM_SEED
        )
    mixture_marginal_trace
    return mixture_marginal_model, mixture_marginal_trace


@app.cell(hide_code=True)
def _(az, mixture_marginal_trace):
    marginal_summary = az.summary(
        mixture_marginal_trace, var_names=["p", "mu"], round_to=3
    )
    marginal_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **ess_bulk** climbs by an order of magnitude or more, and `r_hat` is essentially 1.00. The marginalized model is the same statistical object; the sampler just has a much easier time on it.

    If you need the cluster assignments for downstream analysis, `recover_marginals` reconstructs the discrete-state posterior from the marginalized fit:
    """)
    return


@app.cell
def recover_z(mixture_marginal_model, mixture_marginal_trace):
    from pymc_extras import recover_marginals

    with mixture_marginal_model:
        mixture_recovered = recover_marginals(mixture_marginal_trace)
    z_probs = mixture_recovered.posterior["z"].mean(dim=("chain", "draw")).to_numpy()
    z_probs[:5]
    return (z_probs,)


@app.cell(hide_code=True)
def _(PYMC_BLUE, go, mo, z_probs):
    z_fig = go.Figure()
    z_fig.add_trace(
        go.Histogram(
            x=z_probs,
            nbinsx=30,
            marker_color=PYMC_BLUE,
            opacity=0.85,
        )
    )
    z_fig.update_layout(
        title="Recovered P(z=1 | y) for each observation",
        xaxis_title="Posterior probability of belonging to cluster 1",
        yaxis_title="Number of observations",
        height=350,
    )
    mo.vstack(
        [
            z_fig,
            mo.md(
                "The histogram is bimodal at 0 and 1: most observations are assigned confidently to one cluster, which is the structure we put in when we simulated."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampler Configuration Tips

    Now that you've seen what can go wrong, here are the practical `pm.sample()` settings that help:

    - **`target_accept`**: Default 0.8. Increase to 0.95+ when you see divergences. This uses a smaller step size, trading speed for accuracy.
    - **`tune`**: Default 1000. The warmup period where the sampler adapts its step size and mass matrix. Increase for complex models.
    - **`draws`**: Default 1000. How many posterior samples per chain. Increase if ESS is too low.
    - **`chains`** and **`cores`**: Default 4 chains on 4 cores. Multiple chains are required for R-hat. Never use just 1 chain.
    - **`nuts_sampler`**: PyMC 6 uses nutpie by default when installed. To explicitly select another implementation, e.g. `pm.sample(nuts_sampler="pymc")` or `pm.sample(nuts_sampler="numpyro")`.
    - **`backend`**: PyMC 6 also exposes a `backend=` argument that picks the compute backend independent of the NUTS implementation. The new v6 default is `"numba"`; `"c"` is the previous default and `"jax"` requires `jax` to be installed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Choosing a backend and a NUTS sampler

    PyMC v6 changed two defaults at once: **`nutpie`** is now the default NUTS sampler (it tunes faster and supports low-rank mass-matrix adaptation, which we will meet in **Session 4**), and the **Numba** backend replaces the C backend (no system BLAS or C compiler required). The cells below run the baseline penguin model three ways on the same data so you can see the difference.

    Each cell does one warm-up call (so first-run compile cost is paid before the measured run) and one timed call.
    """)
    return


@app.cell
def benchmark_default(RANDOM_SEED, baseline_model, pm):
    import time

    def _bench_default():
        with baseline_model:
            pm.sample(
                draws=100, tune=200, chains=2, progressbar=False, random_seed=RANDOM_SEED
            )
            t0 = time.perf_counter()
            pm.sample(
                draws=500, tune=500, chains=2, progressbar=False, random_seed=RANDOM_SEED
            )
            return time.perf_counter() - t0

    bench_default_seconds = _bench_default()
    bench_default_seconds
    return bench_default_seconds, time


@app.cell
def benchmark_pymc_nuts(RANDOM_SEED, baseline_model, pm, time):
    def _bench_pymc():
        with baseline_model:
            pm.sample(
                draws=100, tune=200, chains=2, progressbar=False,
                random_seed=RANDOM_SEED, nuts_sampler="pymc",
            )
            t0 = time.perf_counter()
            pm.sample(
                draws=500, tune=500, chains=2, progressbar=False,
                random_seed=RANDOM_SEED, nuts_sampler="pymc",
            )
            return time.perf_counter() - t0

    bench_pymc_seconds = _bench_pymc()
    bench_pymc_seconds
    return (bench_pymc_seconds,)


@app.cell
def benchmark_c_backend(RANDOM_SEED, baseline_model, pm, time):
    def _bench_c():
        with baseline_model:
            pm.sample(
                draws=100, tune=200, chains=2, progressbar=False,
                random_seed=RANDOM_SEED, backend="c",
            )
            t0 = time.perf_counter()
            pm.sample(
                draws=500, tune=500, chains=2, progressbar=False,
                random_seed=RANDOM_SEED, backend="c",
            )
            return time.perf_counter() - t0

    bench_c_seconds = _bench_c()
    bench_c_seconds
    return (bench_c_seconds,)


@app.cell(hide_code=True)
def benchmark_summary(
    bench_c_seconds,
    bench_default_seconds,
    bench_pymc_seconds,
    mo,
    pl,
):
    backend_summary = pl.DataFrame(
        {
            "configuration": [
                "v6 default (nutpie + numba)",
                "pre-v6 NUTS (nuts_sampler='pymc')",
                "pre-v6 backend (backend='c')",
            ],
            "wall_clock_s": [
                round(bench_default_seconds, 2),
                round(bench_pymc_seconds, 2),
                round(bench_c_seconds, 2),
            ],
        }
    )
    mo.vstack(
        [
            mo.md(
                "**Sampling wall-clock on the baseline penguin model (500 draws, 500 tune, 2 chains, post-warmup):**"
            ),
            backend_summary,
            mo.md(
                "If you have `jax` and `numpyro` installed, `pm.sample(nuts_sampler='numpyro', backend='jax')` is another option worth timing. Don't generalize from a tiny model: the relative ordering can flip for hierarchical or high-dim models where adaptation matters more than per-iteration cost."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise: Diagnose and Fix

    The following model has deliberate problems. Run the diagnostics, identify what's wrong, and fix it.
    """)
    return


@app.cell
def _(RANDOM_SEED, as_model, body_mass_kg, penguins, pm):
    @as_model()
    def _exercise():
        bill_length = penguins.get_column("bill_length_mm").to_numpy()
        bill_length_std = (bill_length - bill_length.mean()) / bill_length.std()
        alpha = pm.Normal("alpha", mu=0, sigma=0.01)
        beta = pm.Normal("beta", mu=0, sigma=0.01)
        sigma = pm.HalfNormal("sigma", sigma=0.01)
        mu = alpha + beta * bill_length_std
        pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)

    with _exercise():
        exercise_trace = pm.sample(1000, random_seed=RANDOM_SEED)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Hint": mo.md(r"""
        The priors are far too tight: `Normal(0, 0.01)` forces the intercept and slope toward zero with vanishing wiggle room, and `HalfNormal(0.01)` constrains residual scale to ~0.01 kg — orders of magnitude tighter than the data. Loosen them to scales the data can actually inhabit.
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
        bill_length = penguins.get_column("bill_length_mm").to_numpy()
        bill_length_std = (bill_length - bill_length.mean()) / bill_length.std()

        with pm.Model() as fixed_model:
            alpha = pm.Normal("alpha", mu=4, sigma=2)
            beta = pm.Normal("beta", mu=0, sigma=2)
            sigma = pm.HalfNormal("sigma", sigma=2)
            mu = alpha + beta * bill_length_std
            pm.Normal("mass", mu=mu, sigma=sigma, observed=body_mass_kg)
            fixed_trace = pm.sample(1000, random_seed=RANDOM_SEED)

        az.summary(fixed_trace)
        ```

        With priors scaled to the data (kg-scale mass, unit-scale standardized covariate), R-hat returns to 1.00 and ESS recovers. The original priors were so tight that they fought the likelihood, producing divergences and a posterior that didn't represent the data.
        """)
        }
    )
    return



if __name__ == "__main__":
    app.run()
