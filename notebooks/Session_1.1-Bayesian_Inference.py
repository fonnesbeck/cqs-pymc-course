import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import base64
    from pathlib import Path

    import arviz as az
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import pymc as pm
    from scipy import stats

    PYMC_BLUE = "#154A72"
    PYMC_GREEN = "#81C240"
    PYMC_LIGHT_BLUE = "#4A9EDE"
    PYMC_DARK_GREEN = "#40611F"

    pymc_template = go.layout.Template()
    pymc_template.layout = go.Layout(
        colorway=[
            PYMC_BLUE,
            PYMC_GREEN,
            PYMC_LIGHT_BLUE,
            "#15726C",
            "#C2C240",
            PYMC_DARK_GREEN,
            "#151B72",
            "#40C240",
        ],
        font=dict(color="#333"),
        title=dict(font=dict(color=PYMC_BLUE)),
    )
    pio.templates["pymc_labs"] = pymc_template
    pio.templates.default = "plotly_white+pymc_labs"

    az.style.use("arviz-variat")

    RANDOM_SEED = 42
    return (
        PYMC_BLUE,
        PYMC_GREEN,
        PYMC_LIGHT_BLUE,
        Path,
        RANDOM_SEED,
        az,
        base64,
        go,
        np,
        pm,
        stats,
    )


@app.cell(hide_code=True)
def _(Path, base64, mo):
    def _make_header():
        logo_path = Path(__file__).parent / "images" / "pymc-labs-logo.png"
        if logo_path.exists():
            logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{logo_b64}" width="300" style="margin-bottom: 0.5rem;">'
        return ""

    mo.md(f"""
    {_make_header()}

    # Bayesian Inference and the PyMC API

    This session introduces the core ideas of Bayesian inference and shows you how to
    express those ideas as probabilistic programs in PyMC.

    **What we will cover (~ 90 minutes):**

    | Topic | Time |
    |-------|------|
    | Why Bayesian? Probability as uncertainty quantification | 25 min |
    | Bayesian updating with Beta-Binomial examples | 30 min |
    | PyMC from the start: models, random variables, sampling | 35 min |
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Why Bayes?

    ---

    ## Probability as Uncertainty Quantification

    In the Bayesian framework, **probability measures uncertainty**, not just long-run frequency.

    - A patient's probability of responding to treatment is not the result of repeating the
      same treatment infinitely many times; it is a statement about *our knowledge*.
    - The key advantage: we get **distributions**, not point estimates. Every parameter
      has a full posterior distribution that tells us what values are plausible and how
      plausible they are.

    This has practical consequences:

    | Frequentist answer | Bayesian answer |
    |---|---|
    | "The effect is significant (p < 0.05)" | "There is a 93% probability the effect is positive" |
    | "The 95% CI is [0.02, 0.15]" | "The posterior probability that the effect exceeds 0.10 is 28%" |
    | "Reject / fail to reject" | Full distribution over plausible values |

    Bayesian inference lets us make **direct probability statements** about the quantities
    we care about, and naturally incorporates prior knowledge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayes' Theorem

    At the heart of Bayesian inference is a simple formula for updating beliefs in light of evidence:

    $$
    P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}
    $$

    | Term | Name | Meaning |
    |------|------|---------|
    | $P(\theta)$ | **Prior** | What we believe about $\theta$ before seeing data |
    | $P(D \mid \theta)$ | **Likelihood** | How probable the data is, given a particular $\theta$ |
    | $P(D)$ | **Evidence** (marginal likelihood) | Total probability of the data across all $\theta$ |
    | $P(\theta \mid D)$ | **Posterior** | Our updated belief after seeing the data |

    Since $P(D)$ is an intractable normalising constant, we often write:

    $$
    P(\theta \mid D) \propto P(D \mid \theta) \, P(\theta)
    $$

    **Posterior is proportional to Likelihood times Prior.**

    The posterior is always a *compromise* between the prior and the data. With more data, the likelihood dominates; with less data, the prior has more influence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## The Euro Problem: Do the Data Overwhelm the Prior?

    *From David MacKay, adapted by Allen Downey.*

    A Belgian one-Euro coin was spun 250 times and came up heads 140 times. Is the coin fair?

    Let's compare two priors:
    1. **Uniform:** $\text{Beta}(1, 1)$ -- any bias equally likely
    2. **Tight Normal:** $\mathcal{N}(0.5,\ 0.1^2)$ -- we are confident the coin is close to fair
    """)
    return


@app.cell(hide_code=True)
def _(go, mo, np):
    def plot_euro_problem():
        from scipy.stats import binom, norm

        theta = np.linspace(0, 1, 1001)

        prior_uniform = np.ones_like(theta)
        prior_uniform /= prior_uniform.sum()

        prior_normal = norm.pdf(theta, 0.5, 0.1)
        prior_normal /= prior_normal.sum()

        likelihood = binom.pmf(140, 250, theta)

        post_uniform = prior_uniform * likelihood
        post_uniform /= post_uniform.sum()
        post_normal = prior_normal * likelihood
        post_normal /= post_normal.sum()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=theta,
                y=prior_uniform,
                name="Prior (uniform)",
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(21,74,114,0.2)",
                line=dict(width=0),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=theta,
                y=prior_normal,
                name="Prior (Normal)",
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(129,194,64,0.2)",
                line=dict(width=0),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=theta,
                y=post_uniform,
                name="Posterior (uniform prior)",
                line=dict(color="#154A72", width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=theta,
                y=post_normal,
                name="Posterior (Normal prior)",
                line=dict(color="#81C240", width=2),
            )
        )
        fig.add_vline(
            x=0.5, line_dash="dot", line_color="gray", annotation_text="Fair coin"
        )
        fig.update_layout(
            title="Euro Problem: Is the coin fair? (140 heads in 250 spins)",
            xaxis_title="theta (probability of heads)",
            yaxis_title="Probability",
            width=700,
            height=420,
        )

        mean_uniform = float(np.sum(theta * post_uniform))
        mean_normal = float(np.sum(theta * post_normal))
        return fig, mean_uniform, mean_normal

    euro_fig, euro_mean_u, euro_mean_n = plot_euro_problem()

    mo.vstack(
        [
            euro_fig,
            mo.md(f"""
        **Posterior mean (uniform prior):** {euro_mean_u:.3f} &nbsp;&nbsp;
        **Posterior mean (Normal prior):** {euro_mean_n:.3f}

        The flat prior lets the data speak -- its posterior sits at the observed rate
        (θ ≈ {euro_mean_u:.3f}). The tight Normal(0.5) prior moves only partway toward the data
        (θ ≈ {euro_mean_n:.3f}); it was confident enough that 250 spins
        nudge it rather than overwhelm it (notice the green posterior still sits near its prior).
        **How much the data overwhelm the prior depends on the prior's strength** --
        a weak prior is quickly dominated by data, a strong, confident prior resists it.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## The Beta-Binomial Model

    In digital marketing, A/B testing is used to evaluate different versions of an ad or web page. As an example, suppose we test an email campaign that includes a promotional offer. We send version A to 100 people, and 8 of them accept the offer. So the observed conversion rate in this sample is 8%.

    Some questions we want to answer:

    - What is the probability someone responds to this email?
    - How confident are we about that estimate?
    - If we send the same email to 100 more people, how many will respond?

    ### The Beta Distribution as Prior

    We use a **Beta distribution** as the prior because it lives on [0, 1] -- the right domain for a probability. Its parameters have an intuitive **pseudocounts** interpretation:

    - $\alpha$ = number of prior "successes" (responses)
    - $\beta$ = number of prior "failures" (non-responses)
    - Prior mean = $\alpha / (\alpha + \beta)$
    - Prior "sample size" = $\alpha + \beta$

    After observing $k$ successes in $n$ trials, the posterior is:

    $$\text{Beta}(\alpha + k, \, \beta + n - k)$$

    This is called a **conjugate update** -- the posterior has the same family as the prior.
    """)
    return


@app.cell(hide_code=True)
def _():
    campaign_A = dict(n=100, k=8)
    return (campaign_A,)


@app.cell(hide_code=True)
def _(mo):
    beta_prior_alpha_slider = mo.ui.slider(
        0.5, 20, value=2, step=0.5, label="alpha (prior successes)"
    )
    beta_prior_beta_slider = mo.ui.slider(
        0.5, 20, value=5, step=0.5, label="beta (prior failures)"
    )
    mo.hstack([beta_prior_alpha_slider, beta_prior_beta_slider], gap=2)
    return beta_prior_alpha_slider, beta_prior_beta_slider


@app.cell(hide_code=True)
def _(
    PYMC_BLUE,
    PYMC_GREEN,
    PYMC_LIGHT_BLUE,
    beta_prior_alpha_slider,
    beta_prior_beta_slider,
    campaign_A,
    go,
    mo,
    np,
    stats,
):
    def plot_beta_binomial_update():
        a0 = beta_prior_alpha_slider.value
        b0 = beta_prior_beta_slider.value
        k = campaign_A["k"]
        n = campaign_A["n"]
        a1 = a0 + k
        b1 = b0 + (n - k)

        x = np.linspace(0, 0.5, 500)
        prior_pdf = stats.beta.pdf(x, a0, b0)
        posterior_pdf = stats.beta.pdf(x, a1, b1)
        likelihood = stats.binom.pmf(k, n, x)
        likelihood_scaled = likelihood / likelihood.max() * posterior_pdf.max()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=prior_pdf,
                mode="lines",
                name=f"Prior Beta({a0:.1f}, {b0:.1f})",
                line=dict(color=PYMC_LIGHT_BLUE, width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=likelihood_scaled,
                mode="lines",
                name="Likelihood (scaled)",
                line=dict(color=PYMC_GREEN, width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=posterior_pdf,
                mode="lines",
                name=f"Posterior Beta({a1:.1f}, {b1:.1f})",
                line=dict(color=PYMC_BLUE, width=3),
                fill="tozeroy",
                opacity=0.3,
            )
        )
        fig.update_layout(
            title="Beta-Binomial Updating: Prior x Likelihood = Posterior",
            xaxis_title="Conversion rate (theta)",
            yaxis_title="Density",
            width=750,
            height=420,
        )
        post_mean = a1 / (a1 + b1)
        post_lo, post_hi = stats.beta.ppf([0.055, 0.945], a1, b1)
        return fig, post_mean, post_lo, post_hi

    bb_fig, bb_mean, bb_lo, bb_hi = plot_beta_binomial_update()

    mo.vstack(
        [
            bb_fig,
            mo.md(f"""
        **Posterior mean:** {bb_mean:.3f} &nbsp;&nbsp;
        **89% equal-tailed interval:** [{bb_lo:.3f}, {bb_hi:.3f}]

        Try moving the sliders to see how the prior affects the posterior. With 100 observations,
        even fairly strong priors get overwhelmed by the data.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## A/B Testing: Comparing Two Versions

    Now suppose we introduce a new version B of the email. We think it might be more effective, so we run an A/B test with different sample sizes.
    """)
    return


@app.cell(hide_code=True)
def _():
    ab_data_A = dict(n=180, k=16)
    ab_data_B = dict(n=20, k=5)
    return ab_data_A, ab_data_B


@app.cell
def _(RANDOM_SEED, ab_data_A, ab_data_B, pm):
    with pm.Model():
        cr_A = pm.Beta("conversion_rate_A", alpha=2, beta=5)
        cr_B = pm.Beta("conversion_rate_B", alpha=2, beta=5)
        pm.Deterministic("difference", cr_B - cr_A)
        pm.Binomial("obs_A", p=cr_A, n=ab_data_A["n"], observed=ab_data_A["k"])
        pm.Binomial("obs_B", p=cr_B, n=ab_data_B["n"], observed=ab_data_B["k"])
        ab_idata = pm.sample(random_seed=RANDOM_SEED, nuts_sampler="nutpie")

    ab_difference = ab_idata["posterior"]["difference"]
    ab_p_B_better = (ab_difference > 0).mean(dim=("chain", "draw")).item()
    return ab_idata, ab_p_B_better


@app.cell(hide_code=True)
def _(ab_idata, ab_p_B_better, az, mo):
    def plot_ab_posteriors():
        return az.plot_dist(
            ab_idata,
            var_names=["conversion_rate_A", "conversion_rate_B", "difference"],
            group="posterior",
            ci_prob=0.89,
        )

    mo.vstack(
        [
            plot_ab_posteriors(),
            mo.md(f"""
        **Probability that B is better than A:** {ab_p_B_better:.1%}

        Rather than a binary hypothesis test, we get the full **distribution** of possible
        differences. We know *how different* the conversion rates are, not just *whether* they
        are different.

        Notice that B has a wider posterior because it has fewer observations (n=20 vs n=180).
        This is uncertainty quantification at work.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Prior Sensitivity

    The previous examples use a **weakly informative prior**. But when you have only a small
    dataset, the prior makes a big difference.

    ### The Rookie Problem

    > In Major League Baseball, most players have a batting average between .170 and .310.
    > Suppose a player appearing in their first game gets 3 hits out of 3 attempts.
    > What is the posterior distribution for their probability of getting a hit?
    """)
    return


@app.cell
def _(RANDOM_SEED, pm):
    def compute_beta_params(mean, std):
        alpha = mean * ((mean * (1 - mean)) / std**2 - 1)
        beta = (1 - mean) * ((mean * (1 - mean)) / std**2 - 1)
        return alpha, beta

    with pm.Model():
        p_weak = pm.Beta("p", alpha=2, beta=5)
        pm.Binomial("obs", p=p_weak, n=3, observed=3)
        sensitivity_idata_weak = pm.sample(
            random_seed=RANDOM_SEED, nuts_sampler="nutpie"
        )

    sensitivity_a_info, sensitivity_b_info = compute_beta_params(mean=0.250, std=0.030)
    with pm.Model():
        p_info = pm.Beta("p", alpha=sensitivity_a_info, beta=sensitivity_b_info)
        pm.Binomial("obs", p=p_info, n=3, observed=3)
        sensitivity_idata_info = pm.sample(
            random_seed=RANDOM_SEED, nuts_sampler="nutpie"
        )
    return (
        sensitivity_a_info,
        sensitivity_b_info,
        sensitivity_idata_info,
        sensitivity_idata_weak,
    )


@app.cell(hide_code=True)
def _(
    az,
    mo,
    sensitivity_a_info,
    sensitivity_b_info,
    sensitivity_idata_info,
    sensitivity_idata_weak,
):
    def plot_prior_sensitivity_batting():
        return az.plot_dist(
            {
                "Weak Prior: Beta(2, 5)": sensitivity_idata_weak,
                f"Informative Prior: Beta({sensitivity_a_info:.0f}, {sensitivity_b_info:.0f})": sensitivity_idata_info,
            },
            var_names=["p"],
            group="posterior",
            ci_prob=0.89,
        )

    mo.vstack(
        [
            plot_prior_sensitivity_batting(),
            mo.md("""
        With a **weak** prior (left), 3/3 hits pushes the posterior close to 1.0 -- clearly
        unrealistic. With an **informative** prior calibrated to real MLB data (right), the
        posterior barely moves from the prior. This is **shrinkage**: estimates from small
        samples get pulled toward the prior mean.

        **Lesson:** Prior choice matters most when data are scarce. With large samples, the
        data overwhelm the prior regardless.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # PyMC from the Start

    ---

    ## Model Context and Random Variables

    PyMC models are built inside a **model context** -- a `with pm.Model()` block that acts
    as a tape recorder, keeping track of every random variable you define.

    ```python
    with pm.Model() as my_model:
        theta = pm.Beta("theta", alpha=2, beta=5)
        pm.Binomial("obs", p=theta, n=100, observed=8)
    ```

    The model context does three things:

    1. **Registers** each random variable by name
    2. **Builds** the joint log-probability function (the thing MCMC samples from)
    3. **Handles** transformations so samplers work in unconstrained space
    """)
    return


@app.cell
def _(pm):
    with pm.Model() as demo_model:
        pm.Normal("z", mu=0.0, sigma=5.0)

    demo_model
    return (demo_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyMC v6 also exposes `Model.table()`, a compact tabular companion to the mermaid diagram above. It is handy when a model has many parameters or named dimensions and the diagram gets dense.
    """)
    return


@app.cell(hide_code=True)
def _(demo_model):
    demo_table = demo_model.table()
    demo_table
    return


@app.cell(hide_code=True)
def _(demo_model, mo):
    mo.md(f"""
    The model tracks its variables: `{list(demo_model.named_vars.keys())}`

    We can evaluate the log-probability at any point:
    `model.compile_logp()({{\"z\": 2.5}})` = `{demo_model.compile_logp()({"z": 2.5}):.4f}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Distribution Class

    PyMC includes most probability distributions used in statistical modeling. Key arguments
    when constructing a distribution:

    | Argument | Purpose | Example |
    |----------|---------|---------|
    | `name` | Label for the variable (required) | `"theta"` |
    | `shape` | Array dimensions | `shape=(3, 3)` |
    | `dims` | Named dimensions (preferred over shape) | `dims="city"` |
    | `observed` | Data -- makes it a likelihood term | `observed=np.array([...])` |

    Distributions are either **continuous** (float-valued) or **discrete** (integer-valued).
    """)
    return


@app.cell
def _(pm):
    city_names = ["London", "Paris", "Berlin", "Rome", "Madrid"]
    with pm.Model(coords={"city": city_names}) as city_model:
        pm.Normal("mu", mu=0, sigma=1, dims="city")

    city_model
    return (city_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyMC v6 also ships a dim-first construction API at `pymc.dims` (commonly imported as `pmd`). Distributions in `pmd` take `dims=...` directly and broadcast through their named dimensions automatically. We use it heavily for the hierarchical models in **Session 4**; here is the same `city_model` written against `pmd`:
    """)
    return


@app.cell
def _(city_names, pm):
    import pymc.dims as pmd

    with pm.Model(coords={"city": city_names}) as city_model_pmd:
        pmd.Normal("mu", mu=0, sigma=1, dims="city")

    city_model_pmd
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `pymc.dims` currently covers continuous distributions plus `Categorical`. Discrete count and binary distributions (Poisson, Binomial, Bernoulli, NegativeBinomial) are still authored in the classic `pm.*` namespace, so a model with a count likelihood will use both APIs together.

    The `coords` system is one of PyMC's best features. It gives your dimensions meaningful
    labels that propagate through to ArviZ, making posterior summaries and plots much easier
    to interpret.

    You can also use `pm.draw()` to simulate values from a random variable without running
    full MCMC -- useful for quick checks:
    """)
    return


@app.cell
def _(mo, pm):
    with pm.Model(coords={"city": ["London", "Paris", "Berlin"]}):
        x_draw = pm.Normal("x", mu=0, sigma=1, dims="city")

    draw_samples = pm.draw(x_draw, draws=3, random_seed=42)

    mo.md(f"Three draws from a 3-city Normal: `{draw_samples}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Observed Random Variables

    When you pass data via the `observed` keyword, that variable becomes a **likelihood term**
    rather than an unknown parameter. The model uses it to evaluate how well the parameters
    explain the data.

    ```python
    with pm.Model():
        mu = pm.Normal("mu", 0, 10)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
    ```

    ### Deterministic Variables

    A **deterministic variable** is completely determined by its parents -- no randomness is
    added. There are two kinds:

    1. **Anonymous**: a plain Python expression (not saved during sampling)
    2. **Named** via `pm.Deterministic(name, expr)`: saved in the `posterior` group so you get posterior samples

    ```python
    with pm.Model():
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        # Anonymous -- intermediate calculation, not saved
        p = pm.math.invlogit(alpha + beta * x)
        # Named -- saved in the posterior group for analysis
        pm.Deterministic("ld50", -alpha / beta)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Building the Beta-Binomial in PyMC

    Let's build our email campaign model. The data-generating process is:

    1. Draw a conversion rate from the prior: $\theta \sim \text{Beta}(2, 5)$
    2. Draw observations from the likelihood: $k \sim \text{Binomial}(n, \theta)$
    """)
    return


@app.cell
def _(campaign_A, pm):
    with pm.Model() as beta_binom_model:
        conversion_rate = pm.Beta("conversion_rate", alpha=2, beta=5)
        pm.Binomial(
            "obs", p=conversion_rate, n=campaign_A["n"], observed=campaign_A["k"]
        )

    beta_binom_model
    return (beta_binom_model,)


@app.cell(hide_code=True)
def _(beta_binom_model):
    beta_binom_table = beta_binom_model.table()
    beta_binom_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling with `pm.sample()`

    `pm.sample()` runs MCMC to generate samples from the posterior distribution. We use
    **nutpie** as the sampler throughout this workshop -- it is a modern, fast implementation
    of the No-U-Turn Sampler (NUTS).

    Key arguments:

    | Argument | Purpose | Default |
    |----------|---------|---------|
    | `draws` | Number of posterior samples per chain | 1000 |
    | `tune` | Number of tuning (warm-up) samples | 1000 |
    | `chains` | Number of independent chains | 4 |
    | `nuts_sampler` | Which NUTS implementation | `None`; we pass `"nutpie"` |
    | `random_seed` | Reproducibility | None |
    """)
    return


@app.cell
def _(RANDOM_SEED, beta_binom_model, pm):
    with beta_binom_model:
        idata_bb = pm.sample(random_seed=RANDOM_SEED, nuts_sampler="nutpie")

    idata_bb
    return (idata_bb,)


@app.cell(hide_code=True)
def _(az, idata_bb, mo):
    def plot_bb_posterior():
        return az.plot_dist(
            idata_bb,
            var_names=["conversion_rate"],
            group="posterior",
            ci_prob=0.89,
        )

    bb_posterior_fig = plot_bb_posterior()

    mo.vstack(
        [
            bb_posterior_fig,
            mo.md("""
        The posterior is centered near the observed rate of 8%, but the full distribution
        shows the range of plausible values. The 89% ETI (the ArviZ 1.1 default) tells us
        where the true rate most likely falls.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the Output: DataTree

    `pm.sample()` returns an `xarray.DataTree` object. You will still often see the
    variable name `idata` by convention, but the underlying object is a DataTree.
    Groups are accessed with bracket syntax:

    - **`idata["posterior"]`**: MCMC samples for each parameter, indexed by `(chain, draw)`
    - **`idata["sample_stats"]`**: sampler diagnostics (divergences, tree depth, energy, etc.)
    - **`idata["observed_data"]`**: the data you conditioned on

    For example, posterior draws for `conversion_rate` live at
    `idata["posterior"]["conversion_rate"]`.
    """)
    return


@app.cell
def _(az, idata_bb):
    az.summary(idata_bb, var_names=["conversion_rate"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    **Why Bayesian?**
    - Probability quantifies **uncertainty**, not just frequency
    - **Bayes' Theorem**: Posterior $\propto$ Likelihood $\times$ Prior
    - The **Beta-Binomial** model is a natural framework for A/B testing
    - **Pseudocounts**: Beta parameters encode prior "imaginary observations"
    - **Prior sensitivity**: with small data the prior matters; with large data it is overwhelmed

    **PyMC from the Start**
    - `pm.Model()` context registers random variables and builds the log-probability
    - Distributions have `name`, `shape`/`dims`, and optionally `observed`
    - `pm.Deterministic()` saves derived quantities in the `posterior` group
    - `pm.sample()` runs MCMC and returns an `xarray.DataTree` (we pass `nuts_sampler="nutpie"`)

    ---

    <div style="text-align: center; color: #888; font-size: 0.85rem; padding-top: 1rem;">
    Bayesian Inference with PyMC &mdash; A <a href="https://www.pymc-labs.com" style="color: #154A72;">PyMC Labs</a> Workshop
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
