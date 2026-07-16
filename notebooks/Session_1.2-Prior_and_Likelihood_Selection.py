import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import polars as pl
    import pymc as pm
    from scipy import stats
    import arviz as az
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import base64
    import inspect
    from pathlib import Path
    import preliz as pz

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
    pio.templates["course"] = pymc_template
    pio.templates.default = "plotly_white+course"
    data_path = Path(__file__).parent / "data"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Session 1.2: Prior and Likelihood Selection

    This session guides you through choosing appropriate priors and likelihoods for your models.

    **Topics:**
    Distribution families, choosing likelihoods for different data types, prior predictive simulation, and interactive prior exploration with PreliZ

    ---
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    # Part B: Prior and Likelihood Selection

    ---

    ## Distribution Families

    Getting comfortable with common probability distributions will make it easier to specify prior distributions and models for data. Let's explore some key families interactively.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Continuous distributions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Normal distribution

    Core form: $e^{-x^2}$

    * Symmetrical
    * Highest entropy distribution for a fixed mean and variance
    * "Only noise left"
    * Generalizes to higher dimensions (Multivariate Normal)
    """)
    return


@app.cell(hide_code=True)
def _():
    normal_mu_slider = mo.ui.slider(-5, 5, value=0, step=0.1, label="mu")
    normal_sigma_slider = mo.ui.slider(0.4, 3, value=1, step=0.1, label="sigma")
    mo.hstack([normal_mu_slider, normal_sigma_slider], justify="start")
    return normal_mu_slider, normal_sigma_slider


@app.cell(hide_code=True)
def _(normal_mu_slider, normal_sigma_slider):
    def plot_normal_pdf():
        mu = normal_mu_slider.value
        sigma = normal_sigma_slider.value
        x = np.linspace(-8, 8, 300)
        y = stats.norm.pdf(x, mu, sigma)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="tozeroy",
                line=dict(color=PYMC_BLUE, width=2),
            )
        )
        fig.update_layout(
            title=f"Normal(mu={mu:.1f}, sigma={sigma:.1f})",
            xaxis_title="x",
            yaxis_title="Density",
            yaxis=dict(range=[0, 1.0]),
            width=700,
            height=350,
        )
        return fig

    plot_normal_pdf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### When does the Normal apply?

    The Normal distribution is a good model when the variability is just **symmetric, unstructured noise** — no hidden subgroups, no skew, no heavy tails. It's the maximum entropy distribution for a given mean and variance, so it's the "least informative" choice when all you know is the center and spread.

    But if the data contains **unmodeled structure**, the Normal breaks down. Consider human heights: if we mix males and females, the combined distribution is bimodal. More data makes the bimodality *sharper*, not more Normal. However, **conditioned on gender**, each subgroup is well-described by a Normal — the remaining variability is just noise.
    """)
    return


@app.cell(hide_code=True)
def _():
    height_n_slider = mo.ui.slider(50, 10_000, value=500, step=50, label="n (samples)")
    height_condition_toggle = mo.ui.switch(label="Condition on gender", value=False)
    mo.hstack([height_n_slider, height_condition_toggle], justify="start")
    return height_condition_toggle, height_n_slider


@app.cell(hide_code=True)
def _(height_condition_toggle, height_n_slider):
    def plot_heights():
        n = height_n_slider.value
        conditioned = height_condition_toggle.value
        rng = np.random.default_rng(42)

        n_female = n // 2
        n_male = n - n_female
        female_heights = rng.normal(162, 6, size=n_female)
        male_heights = rng.normal(175, 7, size=n_male)
        all_heights = np.concatenate([female_heights, male_heights])

        x_grid = np.linspace(135, 205, 300)
        fig = go.Figure()

        if not conditioned:
            fig.add_trace(
                go.Histogram(
                    x=all_heights,
                    xbins=dict(size=1),
                    histnorm="probability density",
                    marker_color=PYMC_BLUE,
                    opacity=0.5,
                    name="All heights",
                )
            )
            # Best-fit Normal to combined data
            mu_fit, sigma_fit = all_heights.mean(), all_heights.std()
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=stats.norm.pdf(x_grid, mu_fit, sigma_fit),
                    mode="lines",
                    line=dict(color=PYMC_BLUE, width=2.5, dash="dot"),
                    name=f"Best fit Normal({mu_fit:.1f}, {sigma_fit:.1f})",
                )
            )
            fig.update_layout(
                title=f"Combined heights (n={n:,}) — best Normal fit is poor"
            )
        else:
            fig.add_trace(
                go.Histogram(
                    x=female_heights,
                    xbins=dict(size=1),
                    histnorm="probability density",
                    marker_color=PYMC_GREEN,
                    opacity=0.5,
                    name="Female",
                )
            )
            mu_f, sigma_f = female_heights.mean(), female_heights.std()
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=stats.norm.pdf(x_grid, mu_f, sigma_f),
                    mode="lines",
                    line=dict(color=PYMC_GREEN, width=2.5, dash="dot"),
                    name=f"Normal({mu_f:.1f}, {sigma_f:.1f})",
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=male_heights,
                    xbins=dict(size=1),
                    histnorm="probability density",
                    marker_color=PYMC_LIGHT_BLUE,
                    opacity=0.5,
                    name="Male",
                )
            )
            mu_m, sigma_m = male_heights.mean(), male_heights.std()
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=stats.norm.pdf(x_grid, mu_m, sigma_m),
                    mode="lines",
                    line=dict(color=PYMC_LIGHT_BLUE, width=2.5, dash="dot"),
                    name=f"Normal({mu_m:.1f}, {sigma_m:.1f})",
                )
            )
            fig.update_layout(
                title=f"Conditioned on gender (n={n:,}) — Normal fits each subgroup well",
                barmode="overlay",
            )

        fig.update_layout(
            xaxis_title="Height (cm)",
            yaxis_title="Density",
            width=700,
            height=380,
        )
        return fig

    plot_heights()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Student's t distribution

    Core form: $(1 + x^2)^{-\alpha}$

    * Power law (always falls slower than an exponential)
    * Obtained from Normal by **marginalizing out** unknown variance
      * If variance ~ InverseGamma, the integral has a closed form: Student's t
      * Converges to Normal as uncertainty about variance vanishes ($\nu \to \infty$)
    * Generalives to higher dimensions as multivariate StudentT
    * Robust to outliers
    """)
    return


@app.cell(hide_code=True)
def _():
    t_mu_slider = mo.ui.slider(-5, 5, value=0, step=0.1, label="mu")
    t_sigma_slider = mo.ui.slider(1, 3, value=1, step=0.1, label="sigma")
    t_nu_slider = mo.ui.slider(1, 30, value=3, step=0.5, label="ν")
    mo.hstack([t_mu_slider, t_sigma_slider, t_nu_slider], justify="start")
    return t_mu_slider, t_nu_slider, t_sigma_slider


@app.cell(hide_code=True)
def _(t_mu_slider, t_nu_slider, t_sigma_slider):
    def plot_student_t_pdf():
        mu = t_mu_slider.value
        sigma = t_sigma_slider.value
        nu = t_nu_slider.value
        x = np.linspace(-8, 8, 300)
        y_t = stats.t.pdf(x, df=nu, loc=mu, scale=sigma)
        y_norm = stats.norm.pdf(x, mu, sigma)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_t,
                mode="lines",
                fill="tozeroy",
                line=dict(color=PYMC_BLUE, width=2),
                name=f"StudentT(ν={nu})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_norm,
                mode="lines",
                line=dict(color="red", width=2, dash="dot"),
                name="Normal (same mu, sigma)",
            )
        )
        fig.update_layout(
            title=f"StudentT(mu={mu:.1f}, sigma={sigma:.1f}, ν={nu})",
            xaxis_title="x",
            yaxis_title="Density",
            yaxis=dict(range=[0, 0.5]),
            width=700,
            height=350,
        )
        return fig

    plot_student_t_pdf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Robustness to outliers

    The Student-t's heavy tails make it **robust to outliers**. When an outlier is present, the Normal's best fit distorts to accommodate it (wider σ), while the Student-t barely changes.
    """)
    return


@app.cell
def _():
    outlier_toggle = mo.ui.switch(label="Include outlier", value=False)
    cluster_n_slider = mo.ui.slider(10, 200, value=30, step=5, label="n (cluster size)")
    mo.hstack([cluster_n_slider, outlier_toggle], justify="start")
    return cluster_n_slider, outlier_toggle


@app.cell(hide_code=True)
def _(cluster_n_slider, outlier_toggle):
    def plot_outlier_robustness():
        n = cluster_n_slider.value
        include_outlier = outlier_toggle.value
        rng = np.random.default_rng(42)

        cluster = rng.normal(5, 0.8, size=n)
        outlier_val = 15.0
        data = np.append(cluster, outlier_val) if include_outlier else cluster

        x_grid = np.linspace(0, 18, 400)

        # Best-fit Normal (MLE)
        mu_n, sigma_n = data.mean(), data.std()

        # Student-t with tails fixed at nu=4: only location/scale are fit, so the
        # comparison isolates what heavy tails buy you
        _, t_loc, t_scale = stats.t.fit(data, f0=4)

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=data,
                xbins=dict(size=0.3),
                histnorm="probability density",
                marker_color=PYMC_BLUE,
                opacity=0.4,
                name="Data",
            )
        )

        if include_outlier:
            fig.add_trace(
                go.Scatter(
                    x=[outlier_val],
                    y=[0],
                    mode="markers",
                    marker=dict(color="firebrick", size=14, symbol="triangle-up"),
                    name="Outlier (x = 15)",
                )
            )

        # Student-t first (solid), Normal second (dashed) so the Normal stays
        # visible on top when the two curves coincide
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=stats.t.pdf(x_grid, 4, t_loc, t_scale),
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=4),
                name=f"StudentT (ν=4, μ={t_loc:.2f}, σ={t_scale:.2f})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=stats.norm.pdf(x_grid, mu_n, sigma_n),
                mode="lines",
                line=dict(color=PYMC_GREEN, width=2.5, dash="dash"),
                name=f"Normal (μ={mu_n:.2f}, σ={sigma_n:.2f})",
            )
        )

        if include_outlier:
            title = f"One outlier drags the Normal's σ to {sigma_n:.2f} — the Student-t barely moves (σ={t_scale:.2f})"
        else:
            title = "Without an outlier the two fits coincide (dashed Normal over solid Student-t)"

        fig.update_layout(
            title=title,
            xaxis_title="x",
            yaxis_title="Density",
            width=750,
            height=380,
        )
        return fig

    plot_outlier_robustness()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### HalfNormal distribution

    Core form: $e^{-x^2}$ for $x \geq 0$

    * A Normal folded at zero — only the positive half
    * Single parameter: $\sigma$ (scale)
    * Common prior for standard deviations and other positive scale parameters
    """)
    return


@app.cell
def _():
    hn_sigma_slider = mo.ui.slider(0.5, 5, value=1, step=0.1, label="sigma")
    hn_sigma_slider
    return (hn_sigma_slider,)


@app.cell(hide_code=True)
def _(hn_sigma_slider):
    def plot_halfnormal_pdf():
        sigma = hn_sigma_slider.value
        x = np.linspace(0, 15, 300)
        x_full = np.linspace(-15, 15, 300)
        y = stats.halfnorm.pdf(x, scale=sigma)
        y_full = stats.norm.pdf(x_full, 0, sigma)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_full,
                y=y_full,
                mode="lines",
                line=dict(color=PYMC_GREEN, width=1.5, dash="dash"),
                name=f"Normal(0, {sigma:.1f})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="tozeroy",
                line=dict(color=PYMC_BLUE, width=2),
                name=f"HalfNormal({sigma:.1f})",
            )
        )
        fig.update_layout(
            title=f"HalfNormal(sigma={sigma:.1f})",
            xaxis_title="x",
            yaxis_title="Density",
            yaxis=dict(range=[0, 1.0]),
            width=700,
            height=350,
        )
        return fig

    plot_halfnormal_pdf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### LogNormal distribution

    Core form: $\frac{1}{x}\,e^{-(\ln x)^2}$ for $x > 0$

    * If $\log(X) \sim \text{Normal}(\mu, \sigma)$, then $X \sim \text{LogNormal}(\mu, \sigma)$
    * Parameterized by $\mu$ and $\sigma$ on the **log scale**
    * Useful when you think in terms of multiplicative effects
    * **Caution:** the right tail can grow extremely fast
    """)
    return


@app.cell
def _():
    ln_mu_slider = mo.ui.slider(-1, 3, value=0, step=0.1, label="mu")
    ln_sigma_slider = mo.ui.slider(0.1, 2.5, value=0.5, step=0.1, label="sigma")
    mo.hstack([ln_mu_slider, ln_sigma_slider], justify="start")
    return ln_mu_slider, ln_sigma_slider


@app.cell(hide_code=True)
def _(ln_mu_slider, ln_sigma_slider):
    def plot_lognormal_pdf():
        mu = ln_mu_slider.value
        sigma = ln_sigma_slider.value
        xmax = stats.lognorm.ppf(0.995, s=sigma, scale=np.exp(mu))
        x = np.linspace(0.01, xmax, 300)
        y = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

        mean = np.exp(mu + sigma**2 / 2)
        std = mean * np.sqrt(np.exp(sigma**2) - 1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="tozeroy",
                line=dict(color=PYMC_BLUE, width=2),
            )
        )
        fig.update_layout(
            title=f"LogNormal(mu={mu:.1f}, sigma={sigma:.1f})",
            xaxis_title="x",
            yaxis_title="Density",
            width=700,
            height=350,
        )
        return fig, mean, std

    _fig, _mean, _std = plot_lognormal_pdf()
    mo.vstack([_fig, mo.md(f"**Mean:** {_mean:.2f} | **Std:** {_std:.2f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Gamma distribution

    Core form: $x^{\alpha-1}\,e^{-\beta x}$ for $x > 0$

    * Variance grows linearly with mean (not quadratically like LogNormal)
    * Encompasses Exponential ($\alpha = 1$) as a special case
      * Exponential is max entropy for known mean on [0, ∞)
    * Common prior for rates, precisions, and other positive quantities
    """)
    return


@app.cell
def _():
    gamma_alpha_slider = mo.ui.slider(0.5, 20, value=2, step=0.1, label="alpha (shape)")
    gamma_beta_slider = mo.ui.slider(0.1, 5, value=1, step=0.1, label="beta (rate)")
    mo.hstack([gamma_alpha_slider, gamma_beta_slider], justify="start")
    return gamma_alpha_slider, gamma_beta_slider


@app.cell(hide_code=True)
def _(gamma_alpha_slider, gamma_beta_slider):
    def plot_gamma_pdf():
        a = gamma_alpha_slider.value
        b = gamma_beta_slider.value
        mean = a / b
        x = np.linspace(0, 40, 300)
        y = stats.gamma.pdf(x, a=a, scale=1.0 / b)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="tozeroy",
                line=dict(color=PYMC_BLUE, width=2),
            )
        )
        fig.add_vline(
            x=mean,
            line=dict(color=PYMC_GREEN, dash="dash"),
            annotation_text=f"mean={mean:.2f}",
        )
        fig.update_layout(
            title=f"Gamma(alpha={a:.1f}, beta={b:.1f})",
            xaxis_title="x",
            yaxis_title="Density",
            width=700,
            height=350,
        )
        return fig, mean, np.sqrt(a) / b

    _fig, _mean, _std = plot_gamma_pdf()
    mo.vstack([_fig, mo.md(f"**Mean:** {_mean:.3f} | **Std:** {_std:.3f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Uniform distribution

    Core form: $\frac{1}{b-a}$ for $a \leq x \leq b$

    * Constant density over the interval $[a, b]$
    * Maximum entropy distribution for a bounded variable with known support
    * Often considered "uninformative" but is actually quite strong — it says all values in the range are equally likely and values outside are impossible
    * Special case of Beta(1, 1) on [0, 1]
    """)
    return


@app.cell
def _():
    unif_a_slider = mo.ui.slider(-3, 1, value=0, step=0.5, label="a (lower)")
    unif_b_slider = mo.ui.slider(-1, 3, value=1, step=0.5, label="b (upper)")
    mo.hstack([unif_a_slider, unif_b_slider], justify="start")
    return unif_a_slider, unif_b_slider


@app.cell(hide_code=True)
def _(unif_a_slider, unif_b_slider):
    def plot_uniform_pdf():
        a = unif_a_slider.value
        b = unif_b_slider.value
        if a >= b:
            return mo.callout(mo.md("**a must be less than b**"), kind="warn")

        x = np.linspace(-3.5, 3.5, 300)
        y = stats.uniform.pdf(x, loc=a, scale=b - a)
        mean = (a + b) / 2

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="tozeroy",
                line=dict(color=PYMC_BLUE, width=2),
            )
        )
        fig.add_vline(
            x=mean,
            line=dict(color=PYMC_GREEN, dash="dash"),
            annotation_text=f"mean={mean:.1f}",
        )
        fig.update_layout(
            title=f"Uniform(a={a:.1f}, b={b:.1f})",
            xaxis_title="x",
            yaxis_title="Density",
            yaxis=dict(range=[0, 1.5]),
            width=700,
            height=350,
        )
        return fig

    plot_uniform_pdf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Beta distribution

    Core form: $x^{\alpha-1}(1-x)^{\beta-1}$ for $0 \leq x \leq 1$

    * Generalizes to higher dimensions (Dirichlet)
    * Extraordinarily flexible on $[0, 1]$: uniform, U-shaped, skewed, symmetric, concentrated
    * Conjugate prior for the Bernoulli/Binomial probability parameter
    * Uniform is the special case Beta(1, 1)
    """)
    return


@app.cell
def _():
    beta_alpha_slider = mo.ui.slider(0.1, 20, value=2, step=0.1, label="alpha")
    beta_beta_slider = mo.ui.slider(0.1, 20, value=5, step=0.1, label="beta")
    mo.hstack([beta_alpha_slider, beta_beta_slider], justify="start")
    return beta_alpha_slider, beta_beta_slider


@app.cell(hide_code=True)
def _(beta_alpha_slider, beta_beta_slider):
    def plot_beta_pdf():
        a = beta_alpha_slider.value
        b = beta_beta_slider.value
        x = np.linspace(0, 1, 300)
        y = stats.beta.pdf(x, a, b)
        mean = a / (a + b)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="tozeroy",
                line=dict(color=PYMC_BLUE, width=2),
            )
        )
        fig.add_vline(
            x=mean,
            line=dict(color=PYMC_GREEN, dash="dash"),
            annotation_text=f"mean={mean:.3f}",
        )
        fig.update_layout(
            title=f"Beta(alpha={a:.1f}, beta={b:.1f})",
            xaxis_title="x",
            yaxis_title="Density",
            width=700,
            height=350,
        )

        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return fig, mean, var

    _fig, _mean, _var = plot_beta_pdf()
    mo.vstack([_fig, mo.md(f"**Mean:** {_mean:.4f} | **Variance:** {_var:.4f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Discrete distributions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Bernoulli distribution

    Core form: $p^k(1-p)^{1-k}$ for $k \in \{0, 1\}$

    * The simplest discrete distribution — a single trial with two outcomes
    * Building block for the Binomial (sum of Bernoullis)
    * Generalizes to multivariate as Categorical
    """)
    return


@app.cell
def _():
    bern_p_slider = mo.ui.slider(0.01, 0.99, value=0.3, step=0.01, label="p")
    bern_p_slider
    return (bern_p_slider,)


@app.cell(hide_code=True)
def _(bern_p_slider):
    def plot_bernoulli_pmf():
        p = bern_p_slider.value
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[0, 1],
                y=[1 - p, p],
                marker_color=PYMC_BLUE,
                opacity=0.7,
                width=0.4,
            )
        )
        fig.update_layout(
            title=f"Bernoulli(p={p:.2f})",
            xaxis_title="k",
            yaxis_title="P(X=k)",
            xaxis=dict(tickvals=[0, 1], ticktext=["0 (failure)", "1 (success)"]),
            yaxis=dict(range=[0, 1]),
            width=700,
            height=350,
        )
        return fig

    plot_bernoulli_pmf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Binomial distribution

    Core form: $\binom{n}{k}\,p^k\,(1-p)^{n-k}$ for $k = 0, 1, \ldots, n$

    * Fixed number of trials $n$, each with success probability $p$
    * Order doesn't matter.
    * Bernoulli is special case when n=1
    * Variance is always **less** than the mean
    * The starting point for "counting successes"
    * Generalizes to multivariate as Multinomial
    """)
    return


@app.cell
def _():
    binom_n_slider = mo.ui.slider(1, 50, value=10, step=1, label="n (trials)")
    binom_p_slider = mo.ui.slider(0.01, 0.99, value=0.3, step=0.01, label="p")
    mo.hstack([binom_n_slider, binom_p_slider], justify="start")
    return binom_n_slider, binom_p_slider


@app.cell(hide_code=True)
def _(binom_n_slider, binom_p_slider):
    def plot_binomial_pmf():
        n = binom_n_slider.value
        p = binom_p_slider.value
        x = np.arange(0, 51)
        y = stats.binom.pmf(x, n, p)
        mean = n * p
        var = n * p * (1 - p)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                marker_color=PYMC_BLUE,
                opacity=0.7,
            )
        )
        fig.update_layout(
            title=f"Binomial(n={n}, p={p:.2f})",
            xaxis_title="k",
            yaxis_title="P(X=k)",
            width=700,
            height=350,
        )
        return fig, mean, var

    _fig, _mean, _var = plot_binomial_pmf()
    mo.vstack([_fig, mo.md(f"**Mean:** {_mean:.2f} | **Variance:** {_var:.2f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Poisson distribution

    Core form: $\frac{\lambda^k\,e^{-\lambda}}{k!}$ for $k = 0, 1, 2, \ldots$

    * No upper bound on counts (unlike Binomial)
    * Limit of Binomial as $n \to \infty$, $p \to 0$, $np \to \lambda$
    * **Key constraint:** mean = variance. If your data violates this, Poisson is wrong.
    """)
    return


@app.cell
def _():
    pois_lambda_slider = mo.ui.slider(1, 50, value=10, step=1, label="lambda")
    pois_lambda_slider
    return (pois_lambda_slider,)


@app.cell(hide_code=True)
def _(pois_lambda_slider):
    def plot_poisson_pmf():
        lam = pois_lambda_slider.value
        x = np.arange(0, 100)
        y = stats.poisson.pmf(x, lam)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                marker_color=PYMC_BLUE,
                opacity=0.7,
            )
        )
        fig.update_layout(
            title=f"Poisson(λ={lam})",
            xaxis_title="k",
            yaxis_title="P(X=k)",
            width=700,
            height=350,
        )
        return fig, lam

    _fig, _lam = plot_poisson_pmf()
    mo.vstack([_fig, mo.md(f"**Mean = Variance = {_lam}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Negative Binomial distribution

    Core form:
    1. $\binom{k+r-1}{k} p^k (1-p)^r$
    2. $\int_0^\infty \text{Poisson}(k \mid \lambda) \cdot \text{Gamma}(\lambda \mid \alpha, \beta) \, d\lambda$

    * Two parameters: $\mu$ (mean) and $\alpha$ (concentration)
    * Variance always **greater** than the mean
    * As $\alpha \to \infty$, converges to Poisson — just like StudentT $\to$ Normal
    * The go-to for overdispersed count data
    """)
    return


@app.cell
def _():
    nb_mu_slider = mo.ui.slider(1, 50, value=10, step=1, label="mu")
    nb_alpha_slider = mo.ui.slider(0.5, 100, value=5, step=0.5, label="alpha")
    mo.hstack([nb_mu_slider, nb_alpha_slider], justify="start")
    return nb_alpha_slider, nb_mu_slider


@app.cell(hide_code=True)
def _(nb_alpha_slider, nb_mu_slider):
    def plot_negbin_pmf():
        mu = nb_mu_slider.value
        alpha = nb_alpha_slider.value
        p_nb = alpha / (mu + alpha)
        x = np.arange(0, 100)
        y_nb = stats.nbinom.pmf(x, alpha, p_nb)
        y_pois = stats.poisson.pmf(x, mu)
        var = mu + mu**2 / alpha

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x,
                y=y_nb,
                marker_color=PYMC_BLUE,
                opacity=0.7,
                name=f"NegBin(μ={mu}, α={alpha})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_pois,
                mode="lines+markers",
                line=dict(color=PYMC_GREEN, width=2, dash="dash"),
                marker=dict(size=4),
                name=f"Poisson(λ={mu})",
            )
        )
        fig.update_layout(
            title=f"NegBin(mu={mu}, alpha={alpha}) vs Poisson(lambda={mu})",
            xaxis_title="k",
            yaxis_title="P(X=k)",
            width=700,
            height=350,
        )
        return fig, mu, var

    _fig, _mu, _var = plot_negbin_pmf()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"**Mean:** {_mu} | **Variance:** {_var:.1f} (Poisson would force variance = {_mu})"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Overdispersion: when Poisson fails

    Real count data often has variance much larger than the mean — this is **overdispersion**. The Poisson can't capture it (its mean *is* its variance), so it either overestimates the peak or underestimates the spread. The Negative Binomial handles this naturally via its extra parameter $\alpha$.

    Below: samples are drawn from a NegBin. As you decrease $\alpha$, the data becomes more overdispersed and the best-fit Poisson increasingly fails.
    """)
    return


@app.cell
def _():
    od_alpha_slider = mo.ui.slider(
        0.5, 50, value=3, step=0.5, label="alpha (lower = more overdispersed)"
    )
    od_n_slider = mo.ui.slider(50, 2000, value=500, step=50, label="n (samples)")
    mo.hstack([od_alpha_slider, od_n_slider], justify="start")
    return od_alpha_slider, od_n_slider


@app.cell(hide_code=True)
def _(od_alpha_slider, od_n_slider):
    def plot_overdispersion_demo():
        alpha = od_alpha_slider.value
        n = od_n_slider.value
        mu = 15
        rng = np.random.default_rng(42)

        # Generate overdispersed samples from NegBin
        p_nb = alpha / (mu + alpha)
        samples = rng.negative_binomial(alpha, p_nb, size=n)

        # Best-fit Poisson (MLE: lambda = sample mean)
        lam_fit = samples.mean()

        # Best-fit NegBin (method of moments)
        s_mean = samples.mean()
        s_var = samples.var()
        alpha_fit = s_mean**2 / max(s_var - s_mean, 0.1)
        p_fit = alpha_fit / (s_mean + alpha_fit)

        xmax = int(samples.max()) + 1
        x = np.arange(0, xmax)

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=samples,
                xbins=dict(size=1),
                histnorm="probability",
                marker_color=PYMC_BLUE,
                opacity=0.5,
                name="Data",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stats.poisson.pmf(x, lam_fit),
                mode="lines+markers",
                line=dict(color=PYMC_GREEN, width=2.5, dash="dash"),
                marker=dict(size=4),
                name=f"Best Poisson (λ={lam_fit:.1f})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=stats.nbinom.pmf(x, alpha_fit, p_fit),
                mode="lines+markers",
                line=dict(color=PYMC_LIGHT_BLUE, width=2.5),
                marker=dict(size=4),
                name=f"Best NegBin (α={alpha_fit:.1f})",
            )
        )
        fig.update_layout(
            title=f"Overdispersed data (true α={alpha}) — Poisson can't keep up",
            xaxis_title="Count",
            yaxis_title="Probability",
            width=700,
            height=400,
        )
        return fig

    plot_overdispersion_demo()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Choosing distributions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Priors

    Distributions That Often Model Parameters

    | Parameter type | Common priors |
    |---|---|
    | **Location** ($\mu$) | Normal, StudentT |
    | **Scale** ($\sigma$) | HalfNormal, Exponential |
    | **Proportions** | Beta, LogitNormal |
    | **Rates** | Gamma, Lognormal |
    | **Correlations** | LKJCholeskyCov |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Likelihoods

    The likelihood function should match the data-generating process:

    | Data type | Common likelihoods | Key features |
    |---|---|---|
    | **Continuous measurements** | Normal, StudentT | StudentT is more robust to outliers |
    | **Durations / positive values** | Exponential, Gamma, Weibull, Lognormal | Always positive |
    | **Counts** | Poisson, NegativeBinomial | NegBin handles overdispersion |
    | **Proportions / binary** | Bernoulli, Binomial | Beta prior is conjugate |

    **Key principle:** Choose distributions with appropriate domains and enough flexibility to model your data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Choosing a Likelihood for Count Data (Bike Sharing)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("""
        The UCI Bike Sharing Dataset contains daily counts of bike rentals. We'll compare Poisson and NegativeBinomial likelihoods.

        1. Fit a Poisson model with an appropriate prior for `mu`
        2. Fit a NegativeBinomial model with appropriate priors for `mu` and `alpha`
        3. Compare which fits the data better using posterior predictive checks
        """),
        kind="info",
    )
    return


@app.cell
def _():
    bike_df = pl.read_csv(data_path / "day.csv")
    bike_df
    return (bike_df,)


@app.cell
def _(bike_df):
    bike_df["cnt"].plot.hist()
    return


@app.cell
def _(bike_df):
    bike_df["cnt"].describe()
    return


@app.cell(hide_code=True)
def _(bike_df):
    bike_counts = bike_df["cnt"].to_numpy()
    mo.vstack(
        [
            mo.md(
                f"**{len(bike_counts)} days** of bike rental data. Mean: {bike_counts.mean():.0f}, Std: {bike_counts.std():.0f}"
            ),
            mo.md(
                f"Note: the variance ({bike_counts.var():.0f}) is much larger than the mean ({bike_counts.mean():.0f}) — this is **overdispersion**."
            ),
        ]
    )
    return (bike_counts,)


@app.cell
def _(bike_counts, validate_exercise_2):
    def _fit_poisson_model():
        with pm.Model():
            mu = ...
            pm.Poisson("obs", mu=mu, observed=bike_counts)
            idata = ...
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        return idata


    def _fit_negbin_model():
        with pm.Model():
            mu = ...
            alpha = ...
            pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=bike_counts)
            idata = ...
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        return idata


    validate_exercise_2(_fit_poisson_model, _fit_negbin_model)
    return


@app.cell(hide_code=True)
def _(bike_counts):
    def validate_exercise_2(fit_poisson_fn, fit_negbin_fn):
        if "..." in inspect.getsource(fit_poisson_fn) or "..." in inspect.getsource(fit_negbin_fn):
            return mo.callout(
                mo.md(
                    "Replace the `...` placeholders in the exercise cell above, then re-run."
                ),
                kind="info",
            )
        ex2_idata_p = fit_poisson_fn()
        ex2_idata_nb = fit_negbin_fn()

        summary_p = az.summary(ex2_idata_p, var_names=["mu"])
        summary_nb = az.summary(ex2_idata_nb, var_names=["mu", "alpha"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True)

        axes[0].hist(
            bike_counts, bins=50, density=True, alpha=0.5, color="C1", label="Data"
        )
        ppc_p = ex2_idata_p.posterior_predictive["obs"].values.flatten()
        axes[0].hist(
            ppc_p,
            bins=np.linspace(0, bike_counts.max() * 1.2, 80),
            density=True,
            alpha=0.3,
            color="C0",
            label="Posterior predictive",
        )
        axes[0].set_title("Poisson")
        axes[0].set_xlabel("Daily bike rentals")
        axes[0].legend()

        axes[1].hist(
            bike_counts, bins=50, density=True, alpha=0.5, color="C1", label="Data"
        )
        ppc_nb = ex2_idata_nb.posterior_predictive["obs"].values.flatten()
        axes[1].hist(
            ppc_nb,
            bins=np.linspace(0, bike_counts.max() * 1.2, 80),
            density=True,
            alpha=0.3,
            color="C0",
            label="Posterior predictive",
        )
        axes[1].set_title("Negative Binomial")
        axes[1].set_xlabel("Daily bike rentals")
        axes[1].legend()

        fig.tight_layout()

        return mo.vstack(
            [
                mo.callout(mo.md("**Your solution was successful!**"), kind="success"),
                mo.md("**Poisson model summary:**"),
                summary_p,
                mo.md("**Negative Binomial model summary:**"),
                summary_nb,
                fig,
            ]
        )

    return (validate_exercise_2,)


@app.cell(hide_code=True)
def _(
    solution_fit_negbin_model,
    solution_fit_poisson_model,
    validate_exercise_2,
):
    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_fit_poisson_model)}\n\n{inspect.getsource(solution_fit_negbin_model)}\n```"
                    ),
                    mo.lazy(
                        lambda: validate_exercise_2(
                            solution_fit_poisson_model, solution_fit_negbin_model
                        ),
                        show_loading_indicator=True,
                    ),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Normal vs Lognormal Likelihood (Medical Test Data)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("""
        Medical blood test results are often right-skewed — the Normal distribution may not be appropriate.

        1. Fit a Normal model with appropriate priors for `mu` and `sigma`
        2. Fit a Lognormal model with appropriate priors for `mu` and `sigma`
        3. The CDF comparison will run automatically — which fits better? Why?
        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    lab_df = pl.read_csv(data_path / "Test_Data_JLM.csv")
    male_alat = lab_df.filter(pl.col("sex") == "m")["ALAT"].to_numpy().astype(float)
    male_alat = male_alat[~np.isnan(male_alat)]
    return (male_alat,)


@app.cell
def _(male_alat, validate_exercise_3):
    def _fit_normal_model():
        with pm.Model():
            mu = ...
            sigma = ...
            pm.Normal("obs", mu, sigma, observed=male_alat)
            idata = ...
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        return idata


    def _fit_lognormal_model():
        with pm.Model():
            mu = ...
            sigma = ...
            pm.LogNormal("obs", mu, sigma, observed=male_alat)
            idata = ...
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        return idata


    validate_exercise_3(_fit_normal_model, _fit_lognormal_model)
    return


@app.cell(hide_code=True)
def _(male_alat):
    def validate_exercise_3(fit_normal_fn, fit_lognormal_fn):
        if "..." in inspect.getsource(fit_normal_fn) or "..." in inspect.getsource(fit_lognormal_fn):
            return mo.callout(
                mo.md(
                    "Replace the `...` placeholders in the exercise cell above, then re-run."
                ),
                kind="info",
            )
        ex3_idata_n = fit_normal_fn()
        ex3_idata_ln = fit_lognormal_fn()

        sorted_alat = np.sort(male_alat)
        ecdf = np.arange(1, len(sorted_alat) + 1) / len(sorted_alat)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ppc_draw in ex3_idata_n.posterior_predictive["obs"].values.reshape(
            -1, len(male_alat)
        )[:100]:
            axes[0].step(np.sort(ppc_draw), ecdf, color="C0", alpha=0.05)
        axes[0].step(sorted_alat, ecdf, color="C1", linewidth=2, label="Data")
        axes[0].set_title("Normal Model")
        axes[0].legend()

        for ppc_draw in ex3_idata_ln.posterior_predictive["obs"].values.reshape(
            -1, len(male_alat)
        )[:100]:
            axes[1].step(np.sort(ppc_draw), ecdf, color="C0", alpha=0.05)
        axes[1].step(sorted_alat, ecdf, color="C1", linewidth=2, label="Data")
        axes[1].set_title("Lognormal Model")
        axes[1].legend()

        fig.tight_layout()
        return mo.vstack(
            [
                mo.callout(mo.md("**Your solution was successful!**"), kind="success"),
                fig,
            ]
        )

    return (validate_exercise_3,)


@app.cell(hide_code=True)
def _(
    solution_fit_lognormal_model,
    solution_fit_normal_model,
    validate_exercise_3,
):
    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_fit_normal_model)}\n\n{inspect.getsource(solution_fit_lognormal_model)}\n```"
                    ),
                    mo.lazy(
                        lambda: validate_exercise_3(
                            solution_fit_normal_model, solution_fit_lognormal_model
                        ),
                        show_loading_indicator=True,
                    ),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Prior Predictive Simulation

    Before fitting your model to data, you should check that your priors produce *plausible* predictions. This is called **prior predictive checking**.

    The process:
    1. Draw parameter values from your priors
    2. Simulate data from the likelihood using those parameters
    3. Check: does the simulated data look reasonable for your problem?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Example: Penguin Body Mass

    Let's model the body mass of Adelie penguins. We'll compare a **vague** prior with a **weakly informative** prior.
    """)
    return


@app.cell
def _():
    penguins_df = pl.read_csv(data_path / "penguins.csv", null_values="NA").drop_nulls()
    adelie_mass = penguins_df.filter(pl.col("species") == "Adelie")[
        "body_mass_g"
    ].to_numpy()
    penguins_df
    return adelie_mass, penguins_df


@app.cell
def _(penguins_df):
    penguins_df["body_mass_g"].plot.hist()
    return


@app.cell
def _():
    # Vague prior
    with pm.Model():
        mu = pm.Normal("mu", mu=np.log(4000) + 3, sigma=0.8)
        sigma = pm.HalfNormal("sigma", sigma=1)
        pm.LogNormal("y", mu=mu, sigma=sigma)
        vague_prior = pm.sample_prior_predictive(1000, random_seed=2316)
    return (vague_prior,)


@app.cell
def _():
    # Weakly informative prior
    with pm.Model():
        mu2 = pm.Normal("mu", mu=np.log(4000), sigma=0.1)
        sigma2 = pm.HalfNormal("sigma", sigma=0.15)
        pm.LogNormal("y", mu=mu2, sigma=sigma2)
        informed_prior = pm.sample_prior_predictive(1000, random_seed=2316)
    return (informed_prior,)


@app.cell
def _():
    prior_pred_log_toggle = mo.ui.switch(label="Log x-axis", value=True)
    prior_pred_log_toggle
    return (prior_pred_log_toggle,)


@app.cell(hide_code=True)
def _(informed_prior, prior_pred_log_toggle, vague_prior):
    def plot_prior_predictive_comparison():
        use_log = prior_pred_log_toggle.value

        # Collect both datasets
        vague_y = vague_prior.prior["y"].values.flatten()
        vague_y = vague_y[np.isfinite(vague_y) & (vague_y > 0)]
        informed_y = informed_prior.prior["y"].values.flatten()
        informed_y = informed_y[np.isfinite(informed_y) & (informed_y > 0)]

        # Reference lines for scale
        references = [
            (60, "egg"),
            (4_000, "penguin"),
            (80_000, "person"),
            (500_000, "horse"),
        ]

        # Shared bins: log-spaced if log mode, linear otherwise
        all_y = np.concatenate([vague_y, informed_y])
        if use_log:
            lo, hi = max(all_y.min(), 1e-2), all_y.max()
            bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        else:
            bins = 50

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        def _plot_panel(ax, y, title):
            ax.hist(y, bins=bins, density=True, color="C0", alpha=0.7)
            if use_log:
                ax.set_xscale("log")
                ax.set_xlim(lo, hi)
            ax.set_title(title)
            ax.set_xlabel("Predicted mass (g)")
            if use_log:
                ymax = ax.get_ylim()[1]
                for val, label in references:
                    if lo < val < hi:
                        ax.axvline(x=val, color="gray", linestyle=":", alpha=0.7)
                        ax.text(
                            val,
                            ymax * 0.95,
                            f" {label}",
                            fontsize=8,
                            color="gray",
                            va="top",
                            ha="left",
                        )

        _plot_panel(axes[0], vague_y, "Vague priors")
        _plot_panel(axes[1], informed_y, "Weakly informative priors")

        plt.tight_layout()
        return fig

    mo.vstack(
        [
            plot_prior_predictive_comparison(),
            mo.md("""
            The vague priors produce wildly unrealistic predictions — penguins heavier than a person, or lighter than an egg. The weakly informative priors generate data in a plausible range. **Always check your prior predictive distribution before fitting.**
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Exercise: Prior Predictive Check for Penguin Mass
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("""
        Using the Adelie penguin body mass data:

        1. Complete the PyMC model (which uses a `Normal` likelihood)
        2. Define mu and sigma priors
        3. Use `pm.sample_prior_predictive()` to generate predictions
        4. Use `pm.sample()` to get the posterior
        5. Compare the prior and posterior predictive distributions
        """),
        kind="info",
    )
    return


@app.cell
def _(adelie_mass, validate_exercise_1):
    def _exercise_penguin_prior_vs_posterior():
        with pm.Model():
            mu = ...
            sigma = ...
            obs = pm.Normal("obs", mu, sigma, observed=adelie_mass)
            prior_idata = ...
            posterior_idata = ...

        return prior_idata, posterior_idata

    validate_exercise_1(_exercise_penguin_prior_vs_posterior)
    return


@app.cell(hide_code=True)
def _(adelie_mass):
    def validate_exercise_1(exercise_fn):
        if "..." in inspect.getsource(exercise_fn):
            return mo.callout(
                mo.md(
                    "Replace the `...` placeholders in the exercise cell above, then re-run."
                ),
                kind="info",
            )
        ex1_prior, ex1_posterior = exercise_fn()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # mu: prior vs posterior
        _g, _p, _ = az.kde(ex1_prior.prior["mu"].values.flatten())
        axes[0].plot(_g, _p, label="Prior")
        _g, _p, _ = az.kde(ex1_posterior.posterior["mu"].values.flatten())
        axes[0].plot(_g, _p, color="C1", label="Posterior")
        axes[0].set_title("mu")
        axes[0].legend()

        # sigma: prior vs posterior
        _g, _p, _ = az.kde(ex1_prior.prior["sigma"].values.flatten())
        axes[1].plot(_g, _p, label="Prior")
        _g, _p, _ = az.kde(ex1_posterior.posterior["sigma"].values.flatten())
        axes[1].plot(_g, _p, color="C1", label="Posterior")
        axes[1].set_title("sigma")
        axes[1].legend()

        # Prior predictive vs observed data
        _g, _p, _ = az.kde(ex1_prior.prior_predictive["obs"].values.flatten())
        axes[2].plot(_g, _p, label="Prior predictive")
        _g, _p, _ = az.kde(adelie_mass)
        axes[2].plot(_g, _p, color="C1", label="Observed data")
        axes[2].set_title("Prior Predictive vs Data")
        axes[2].legend()

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        fig.tight_layout()

        return mo.vstack(
            [
                mo.callout(
                    mo.md(
                        "**Your solution was successful!** Here is what the prior and posterior look like:"
                    ),
                    kind="success",
                ),
                fig,
            ]
        )

    return (validate_exercise_1,)


@app.cell(hide_code=True)
def _(solution_exercise_1, validate_exercise_1):
    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_exercise_1)}\n```"),
                    mo.lazy(
                        lambda: validate_exercise_1(solution_exercise_1),
                        show_loading_indicator=True,
                    ),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Interactive Prior Exploration with PreliZ

    [PreliZ](https://preliz.readthedocs.io/) is a library designed specifically for prior elicitation. It provides tools to find distributions that match your domain knowledge.

    Two key functions:
    - **`pz.maxent()`** — find the maximum entropy distribution matching specified quantile constraints
    - **`pz.mle()`** — fit a distribution to data using maximum likelihood
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `pz.maxent()`: Maximum Entropy Priors

    If you know that 95% of values should fall between two bounds, `pz.maxent` finds the "least informative" (maximum entropy) distribution consistent with that constraint.
    """)
    return


@app.cell
def _():
    # Example: we think penguin mass is between 2500g and 5500g with 94% probability
    maxent_normal_dist = pz.Normal()
    pz.maxent(maxent_normal_dist, lower=2500, upper=5500, mass=0.94)
    return


@app.cell
def _():
    # Example: response time in ms — we think 94% are between 50ms and 2000ms
    maxent_lognormal_dist = pz.LogNormal()
    pz.maxent(maxent_lognormal_dist, lower=50, upper=2000, mass=0.94)
    return


@app.cell
def _():
    # Example: a rate parameter — we think 94% of values are between 0.1 and 5.0
    maxent_gamma_dist = pz.Gamma()
    pz.maxent(maxent_gamma_dist, lower=0.1, upper=5.0, mass=0.94)
    return


@app.cell
def _():
    # Example: conversion rates — we think 94% of rates are between 0.01 and 0.30
    maxent_beta_dist = pz.Beta()
    pz.maxent(maxent_beta_dist, lower=0.01, upper=0.30, mass=0.94)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `pz.mle()`: Fitting Distributions to Data

    If you have historical data from a similar process, you can fit a distribution to it and use that as your prior.
    """)
    return


@app.cell
def _(adelie_mass):
    # Fit a Normal to the Adelie penguin mass data
    mle_dist = pz.Normal()
    pz.mle([mle_dist], adelie_mass)
    return (mle_dist,)


@app.cell(hide_code=True)
def _(adelie_mass, mle_dist):
    _ax = mle_dist.plot_pdf()
    _ax.set_xlabel("Body mass (g)")
    _ax.set_title("MLE-fit Normal to Adelie penguin mass")
    _ax.hist(adelie_mass, bins=30, density=True, alpha=0.3, color="gray")
    _fig = plt.gcf()
    _fig.set_size_inches(7, 2)
    mo.vstack(
        [
            mo.md(
                f"**MLE fit:** Normal(mu={mle_dist.mu:.0f}, sigma={mle_dist.sigma:.0f})"
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Use PreliZ to Find Priors for the Penguins Model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("""
        Use `pz.maxent()` to find appropriate priors for a model of penguin body mass, then fit the model and examine the posterior.

        **Domain knowledge:**
        - Penguin body mass is between 2500g and 6500g (with 94% probability)
        - The standard deviation of mass is probably between 100g and 800g (with 94% probability)

        1. Use `pz.maxent(pz.Normal(), ?)` to get the prior for mu
        2. Use `pz.maxent(pz.Gamma(), ?)` to get the prior for sigma
        3. Build the model with these priors and sample from the prior predictive
        """),
        kind="info",
    )
    return


@app.cell
def _(adelie_mass, validate_exercise_4):
    def _exercise_preliz_priors():
        # Find priors with maxent
        mu_prior = ...
        sigma_prior = ...

        # Build model and sample prior predictive
        with pm.Model():
            mu = ...
            sigma = ...
            pm.Normal("obs", mu, sigma, observed=adelie_mass)
            prior_idata = ...

        return mu_prior, sigma_prior, prior_idata

    validate_exercise_4(_exercise_preliz_priors)
    return


@app.cell(hide_code=True)
def _(adelie_mass):
    def validate_exercise_4(exercise_fn):
        if "..." in inspect.getsource(exercise_fn):
            return mo.callout(
                mo.md(
                    "Replace the `...` placeholders in the exercise cell above, then re-run."
                ),
                kind="info",
            )
        ex4_mu_prior, ex4_sigma_prior, ex4_prior = exercise_fn()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # Plot mu prior from PreliZ
        ex4_mu_prior.plot_pdf(ax=axes[0])
        axes[0].set_title(
            f"mu prior: Normal({ex4_mu_prior.mu:.0f}, {ex4_mu_prior.sigma:.0f})"
        )

        # Plot sigma prior from PreliZ
        ex4_sigma_prior.plot_pdf(ax=axes[1])
        axes[1].set_title(
            f"sigma prior: Gamma({ex4_sigma_prior.alpha:.2f}, {ex4_sigma_prior.beta:.4f})"
        )

        # Prior predictive vs observed data
        _g, _p, _ = az.kde(ex4_prior.prior_predictive["obs"].values.flatten())
        axes[2].plot(_g, _p, label="Prior predictive")
        _g, _p, _ = az.kde(adelie_mass)
        axes[2].plot(_g, _p, color="C1", label="Observed data")
        axes[2].set_title("Prior Predictive vs Data")
        axes[2].legend()

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        fig.tight_layout()

        return mo.vstack(
            [
                mo.callout(mo.md("**Your solution was successful!**"), kind="success"),
                fig,
            ]
        )

    return (validate_exercise_4,)


@app.cell(hide_code=True)
def _(solution_exercise_4, validate_exercise_4):
    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_exercise_4)}\n```"),
                    mo.lazy(
                        lambda: validate_exercise_4(solution_exercise_4),
                        show_loading_indicator=True,
                    ),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Priors for Correlation Matrices: The LKJ Distribution

    When modelling multivariate data, we often need priors on **correlation matrices**. Correlation matrices have special constraints (symmetric, positive definite, diagonal = 1), so we can't use standard priors.

    The **LKJ distribution** (Lewandowski, Kurowicka, Joe) is designed for this purpose. It has a single parameter $\eta$:

    - $\eta = 1$: uniform over valid correlation matrices
    - $\eta > 1$: favours matrices closer to the identity (weak correlations)
    - $\eta < 1$: favours matrices with strong correlations

    In PyMC, we use `pm.LKJCholeskyCov` which jointly samples a correlation matrix and standard deviations:

    ```python
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol", n=p, eta=2.0,
        sd_dist=pm.Exponential.dist(1.0),
    )
    ```

    This is essential for hierarchical models with **correlated random effects** (Session 5.2) and multivariate models.

    ---

    ## Summary

    **Part B — Prior and Likelihood Selection:**
    - Match your likelihood to the data-generating process (Normal, Lognormal, Poisson, NegBin, etc.)
    - Always do prior predictive checks
    - Use PreliZ (`pz.maxent`, `pz.mle`) for principled prior elicitation
    - The choice of prior matters most with small data
    - **LKJ distribution** for correlation matrix priors

    ---

    <div style="text-align: center; color: #888; font-size: 0.85rem; padding-top: 1rem;">
    Introduction to PyMC and Bayesian Modeling
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(adelie_mass):
    def solution_exercise_1():
        with pm.Model():
            mu = pm.Normal("mu", mu=4000, sigma=500)
            sigma = pm.HalfNormal("sigma", sigma=500)
            pm.Normal("obs", mu, sigma, observed=adelie_mass)
            prior_idata = pm.sample_prior_predictive()
            posterior_idata = pm.sample()

        return prior_idata, posterior_idata

    return (solution_exercise_1,)


@app.cell(hide_code=True)
def _(bike_counts):
    def solution_fit_poisson_model():
        with pm.Model():
            mu = pm.HalfNormal("mu", sigma=5000)
            pm.Poisson("obs", mu=mu, observed=bike_counts)
            idata = pm.sample(500)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        return idata

    def solution_fit_negbin_model():
        with pm.Model():
            mu = pm.HalfNormal("mu", sigma=5000)
            alpha = pm.HalfNormal("alpha", sigma=10)
            pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=bike_counts)
            idata = pm.sample(500)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        return idata

    return solution_fit_negbin_model, solution_fit_poisson_model


@app.cell(hide_code=True)
def _(male_alat):
    def solution_fit_normal_model():
        with pm.Model():
            mu = pm.Normal("mu", 30, 20)
            sigma = pm.HalfNormal("sigma", 20)
            pm.Normal("obs", mu, sigma, observed=male_alat)
            idata = pm.sample(500)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        return idata

    def solution_fit_lognormal_model():
        with pm.Model():
            mu = pm.Normal("mu", 3, 1)
            sigma = pm.HalfNormal("sigma", 1)
            pm.LogNormal("obs", mu, sigma, observed=male_alat)
            idata = pm.sample(500)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        return idata

    return solution_fit_lognormal_model, solution_fit_normal_model


@app.cell(hide_code=True)
def _(adelie_mass):
    def solution_exercise_4():
        # Find priors with maxent
        mu_prior = pz.Normal()
        pz.maxent(mu_prior, lower=2500, upper=6500, mass=0.94)

        sigma_prior = pz.Gamma()
        pz.maxent(sigma_prior, lower=100, upper=800, mass=0.94)

        # Build model and sample prior predictive
        with pm.Model():
            mu = pm.Normal("mu", mu=mu_prior.mu, sigma=mu_prior.sigma)
            sigma = pm.Gamma("sigma", alpha=sigma_prior.alpha, beta=sigma_prior.beta)
            pm.Normal("obs", mu, sigma, observed=adelie_mass)
            prior_idata = pm.sample_prior_predictive()

        return mu_prior, sigma_prior, prior_idata

    return (solution_exercise_4,)


if __name__ == "__main__":
    app.run()
