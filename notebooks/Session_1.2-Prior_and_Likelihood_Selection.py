import marimo

__generated_with = "0.23.14"
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
    Distribution families (univariate and multivariate), choosing likelihoods for different data types, censored and truncated data, prior predictive simulation, and interactive prior exploration with PreliZ

    ---
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
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

    The Normal distribution is a good model when the variability is just **symmetric, unstructured noise**: no hidden subgroups, no skew, no heavy tails. It's the maximum entropy distribution for a given mean and variance, so it's the "least informative" choice when all you know is the center and spread.

    But if the data contains **unmodeled structure**, the Normal breaks down. Consider human heights: if we mix males and females, the combined distribution is bimodal. More data makes the bimodality *sharper*, not more Normal. However, **conditioned on gender**, each subgroup is well-described by a Normal; the remaining variability is just noise.
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


@app.cell(hide_code=True)
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

    * A Normal folded at zero: only the positive half
    * Single parameter: $\sigma$ (scale)
    * Common prior for standard deviations and other positive scale parameters
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    * Often considered "uninformative" but is actually quite strong: it says all values in the range are equally likely and values outside are impossible
    * Special case of Beta(1, 1) on [0, 1]
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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

    * The simplest discrete distribution: a single trial with two outcomes
    * Building block for the Binomial (sum of Bernoullis)
    * Generalizes to multivariate as Categorical
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    * As $\alpha \to \infty$, converges to Poisson, just like StudentT $\to$ Normal
    * The go-to for overdispersed count data
    """)
    return


@app.cell(hide_code=True)
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

    Real count data often has variance much larger than the mean; this is **overdispersion**. The Poisson can't capture it (its mean *is* its variance), so it either overestimates the peak or underestimates the spread. The Negative Binomial handles this naturally via its extra parameter $\alpha$.

    Below: samples are drawn from a NegBin. As you decrease $\alpha$, the data becomes more overdispersed and the best-fit Poisson increasingly fails.
    """)
    return


@app.cell(hide_code=True)
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
    ### Zero-inflated distributions

    Overdispersion isn't the only way count data breaks the Poisson. Just as often, the problem is **too many zeros**: cigarettes smoked per day, doctor visits per year, parasite eggs per fecal sample. These datasets have a spike at zero that no Poisson (or even NegBin) can reproduce, because the zeros come from *two different processes*:

    * **Structural zeros** — never at risk: non-smokers smoke zero cigarettes no matter what
    * **Sampling zeros** — at risk, but happened to be zero: a smoker on a day they didn't smoke

    The **Zero-Inflated Poisson (ZIP)** models this as a mixture: with probability $1-\psi$ the observation is a structural zero, and with probability $\psi$ it comes from a Poisson:

    $$P(X=0) = (1-\psi) + \psi e^{-\theta}, \qquad P(X=k) = \psi \frac{\theta^k e^{-\theta}}{k!} \quad (k \geq 1)$$

    PyMC provides these ready-made: `pm.ZeroInflatedPoisson`, `pm.ZeroInflatedNegativeBinomial`, and `pm.ZeroInflatedBinomial`.
    """)
    return


@app.cell(hide_code=True)
def _():
    zip_psi_slider = mo.ui.slider(
        0.1, 1.0, value=0.6, step=0.05, label="psi (P(Poisson component))"
    )
    zip_theta_slider = mo.ui.slider(1, 20, value=6, step=1, label="theta")
    mo.hstack([zip_psi_slider, zip_theta_slider], justify="start")
    return zip_psi_slider, zip_theta_slider


@app.cell(hide_code=True)
def _(zip_psi_slider, zip_theta_slider):
    def plot_zip_pmf():
        psi = zip_psi_slider.value
        theta = zip_theta_slider.value
        x = np.arange(0, 30)

        y_zip = psi * stats.poisson.pmf(x, theta)
        y_zip[0] += 1 - psi

        # Poisson forced to match the ZIP's mean — what a naive fit would use
        lam = psi * theta
        y_pois = stats.poisson.pmf(x, lam)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x,
                y=y_zip,
                marker_color=PYMC_BLUE,
                opacity=0.7,
                name=f"ZIP(ψ={psi}, θ={theta})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_pois,
                mode="lines+markers",
                line=dict(color=PYMC_GREEN, width=2, dash="dash"),
                marker=dict(size=4),
                name=f"Poisson(λ={lam:.1f}), same mean",
            )
        )
        fig.update_layout(
            title=f"ZIP(psi={psi}, theta={theta}) vs Poisson with the same mean",
            xaxis_title="k",
            yaxis_title="P(X=k)",
            width=700,
            height=350,
        )
        return fig, y_zip[0], y_pois[0]

    _fig, _zip0, _pois0 = plot_zip_pmf()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"**P(X=0):** ZIP gives {_zip0:.3f} — a same-mean Poisson can only manage {_pois0:.3f}"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Mixtures: the general tool

    Zero-inflation is a special case of a **mixture distribution** — data drawn from latent subpopulations, each with its own distribution. The ZIP above is exactly:

    ```python
    pm.Mixture(
        "y",
        w=[1 - psi, psi],
        comp_dists=[pm.DiracDelta.dist(0), pm.Poisson.dist(theta)],
    )
    ```

    a point mass at zero mixed with a Poisson. `pm.Mixture` accepts any weights and component distributions, so the same construction handles bimodal measurements, outlier-contaminated data, or any situation where observations come from groups you can't directly label.
    """)
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
    | **Counts with excess zeros** | ZeroInflatedPoisson, ZeroInflatedNegativeBinomial | Mixture of structural zeros and counts |
    | **Proportions / binary** | Bernoulli, Binomial | Beta prior is conjugate |

    **Key principle:** Choose distributions with appropriate domains and enough flexibility to model your data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Censored and Truncated Data

    Sometimes the mismatch between data and likelihood isn't the distribution family; it's the **observation process**:

    - **Truncated**: values outside the bounds are *never observed at all*, and you don't know how many you missed. Only insurance claims above the deductible are filed; only admitted students appear in the dataset. The density is renormalized over the observable range.
    - **Censored**: an observation happened, but you only learn that it hit a bound. A sensor reads "≥ 100" at its detection limit; a patient is still alive when the study ends. Probability mass piles up *at* the bounds.

    PyMC models both with `pm.Censored` and `pm.Truncated`, which take any base distribution and apply the observation process to it:

    ```python
    pm.Censored("y", pm.Normal.dist(mu, sigma), lower=None, upper=limit, observed=y)
    pm.Truncated("y", pm.Normal.dist(mu, sigma), lower=0, observed=y)
    ```

    (`pm.TruncatedNormal` is a named shortcut for the common case, you'll see it in Session 5.1.) Applying the wrong type (or ignoring the observation process entirely and fitting the plain distribution) biases every parameter estimate, so recognizing censoring versus truncation is as important as picking the right family.
    """)
    return


@app.cell
def _():
    censored_draws = pm.draw(
        pm.Censored.dist(pm.Normal.dist(0, 1), lower=None, upper=1.0),
        draws=5000,
        random_seed=42,
    )
    truncated_draws = pm.draw(
        pm.Truncated.dist(pm.Normal.dist(0, 1), lower=None, upper=1.0),
        draws=5000,
        random_seed=42,
    )
    return censored_draws, truncated_draws


@app.cell(hide_code=True)
def _(censored_draws, truncated_draws):
    def plot_censored_truncated():
        from plotly.subplots import make_subplots

        upper = 1.0
        x = np.linspace(-4, 4, 300)
        base_pdf = stats.norm.pdf(x)

        fig = make_subplots(
            rows=1, cols=2, subplot_titles=["Censored at 1", "Truncated at 1"]
        )
        for col, draws in [(1, censored_draws), (2, truncated_draws)]:
            fig.add_trace(
                go.Histogram(
                    x=draws,
                    xbins=dict(start=-4, end=4, size=0.1),
                    histnorm="probability density",
                    marker_color=PYMC_BLUE,
                    opacity=0.6,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=base_pdf,
                    mode="lines",
                    line=dict(color=PYMC_GREEN, width=2, dash="dash"),
                    name="Normal(0, 1)",
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
            fig.add_vline(
                x=upper, line_dash="dot", line_color="firebrick", row=1, col=col
            )

        fig.update_layout(
            title=(
                "5000 draws from Normal(0, 1) bounded above at 1 — censoring piles mass"
                " on the bound, truncation renormalizes what remains"
            ),
            width=900,
            height=350,
            margin=dict(l=40, r=20, t=70, b=40),
        )
        return fig

    plot_censored_truncated()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Choosing a Likelihood by Simulation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("""
        The Mayo Clinic **primary biliary cholangitis (PBC)** study recorded serum
        bilirubin (mg/dL) for 418 patients, a strictly positive, right-skewed
        biomarker.

        Your job: find a distribution whose **draws look like this data**. There is
        no fitting and no MCMC here: you pick a family and its parameters, simulate
        from it with `pm.draw`, and compare the simulated values to the real ones by
        eye. Adjust and repeat until the shape and tail match.

        Two things to weigh as you choose:

        1. **Support**: which values can the distribution even produce? Bilirubin is
           a concentration, so it can never be negative.
        2. **Shape**: the data are heavily right-skewed with a long upper tail.
           Which families can reproduce that, and which cannot?
        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    pbc_df = pl.read_csv(data_path / "pbc_bilirubin.csv")
    pbc_df
    return (pbc_df,)


@app.cell(hide_code=True)
def _(pbc_df):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.hist(pbc_df["bili"].to_numpy(), bins=50, density=True, color="C1", alpha=0.7)
    _ax.set_xlabel("Serum bilirubin (mg/dL)")
    _ax.set_ylabel("Density")
    _ax.set_title("Serum bilirubin — 418 PBC patients")
    _fig
    return


@app.cell(hide_code=True)
def _(pbc_df):
    pbc_df["bili"].describe()
    return


@app.cell(hide_code=True)
def _(pbc_df):
    bilirubin = pbc_df["bili"].to_numpy()
    mo.vstack(
        [
            mo.md(
                f"**{len(bilirubin)} patients.** Serum bilirubin ranges from "
                f"{bilirubin.min():.1f} to {bilirubin.max():.1f} mg/dL "
                f"(mean {bilirubin.mean():.2f}, median {np.median(bilirubin):.2f})."
            ),
            mo.md(
                "Note: every value is **strictly positive** and the mean sits well "
                "above the median; the data are heavily **right-skewed**, with a "
                "few patients carrying very high bilirubin. Keep both facts in mind "
                "when you choose a likelihood."
            ),
        ]
    )
    return (bilirubin,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `pm.draw` samples directly from a distribution: no fitting, no MCMC. Build an
    unfit distribution with `pm.SomeDistribution.dist(...)`, then
    `pm.draw(dist, draws=N)` returns an array of simulated values:

    ```python
    draws = pm.draw(pm.LogNormal.dist(mu=0.5, sigma=1.0), draws=2000)
    ```

    Change the family or its parameters and the draws change instantly. That
    immediate feedback is how you build intuition for what a prior or likelihood
    actually implies about your data.
    """)
    return


@app.cell
def _(bilirubin, show_draws_vs_data):
    # Pick a distribution whose draws resemble the bilirubin data.
    # The starter is a Normal with the data's mean and sd — run it, see what goes
    # wrong, then change `candidate` to a better family and parameters.
    candidate = pm.Normal.dist(mu=bilirubin.mean(), sigma=bilirubin.std())

    draws = pm.draw(candidate, draws=4000, random_seed=42)
    show_draws_vs_data(draws)
    return


@app.cell(hide_code=True)
def _(bilirubin):
    def show_draws_vs_data(draws):
        draws = np.asarray(draws)
        lo = min(0.0, float(draws.min()))
        hi = min(max(float(bilirubin.max()), float(np.percentile(draws, 99))), 40.0)
        bins = np.linspace(lo, hi, 60)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(
            bilirubin,
            bins=bins,
            density=True,
            alpha=0.5,
            color="C1",
            label="Bilirubin data",
        )
        ax.hist(
            draws,
            bins=bins,
            density=True,
            alpha=0.4,
            color="C0",
            label="Your draws",
        )
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.set_xlabel("Serum bilirubin (mg/dL)")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()

        frac_neg = float(np.mean(draws < 0))
        stats_line = (
            f"Draws: median {np.median(draws):.2f}, mean {draws.mean():.2f} "
            f"(data: median {np.median(bilirubin):.2f}, mean {bilirubin.mean():.2f})."
        )
        if frac_neg > 0.001:
            verdict = (
                f" **{frac_neg:.0%} of your draws are negative**: impossible for a "
                "concentration, so this family is wrong."
            )
        else:
            verdict = (
                " All draws are positive; now check how well the peak and the right "
                "tail line up with the data."
            )
        return mo.vstack([mo.md(stats_line + verdict), fig])

    return (show_draws_vs_data,)


@app.cell(hide_code=True)
def _(bilirubin, show_draws_vs_data):
    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        "Bilirubin is a **strictly positive, right-skewed** "
                        "concentration. The Normal fails on support alone: centered "
                        "near the mean it puts about a quarter of its draws below "
                        "zero. A **LogNormal** lives on the positive axis and is "
                        "right-skewed by construction, so its draws match both the "
                        "peak and the long tail. A clean way to set its parameters is "
                        "straight from the data on the log scale: "
                        "`mu = np.log(bilirubin).mean()`, "
                        "`sigma = np.log(bilirubin).std()`. (A Gamma works well too, "
                        "try it and compare.)"
                    ),
                    mo.md(
                        "```python\n"
                        "candidate = pm.LogNormal.dist(\n"
                        "    mu=np.log(bilirubin).mean(),\n"
                        "    sigma=np.log(bilirubin).std(),\n"
                        ")\n"
                        "draws = pm.draw(candidate, draws=4000, random_seed=1)\n"
                        "show_draws_vs_data(draws)\n"
                        "```"
                    ),
                    show_draws_vs_data(
                        pm.draw(
                            pm.LogNormal.dist(
                                mu=np.log(bilirubin).mean(),
                                sigma=np.log(bilirubin).std(),
                            ),
                            draws=4000,
                            random_seed=1,
                        )
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

    These are the Palmer Penguins data, body measurements (mass in grams) for
    three penguin species collected at Palmer Station, Antarctica.
    """)
    return


@app.cell(hide_code=True)
def _():
    penguins_df = pl.read_csv(data_path / "penguins.csv", null_values="NA").drop_nulls()
    adelie_mass = penguins_df.filter(pl.col("species") == "Adelie")[
        "body_mass_g"
    ].to_numpy()
    penguins_df
    return adelie_mass, penguins_df


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
            The vague priors produce wildly unrealistic predictions: penguins heavier than a person, or lighter than an egg. The weakly informative priors generate data in a plausible range. **Always check your prior predictive distribution before fitting.**
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Exercise: Vague prior specification for modeling binary outcomes
    """)
    return


@app.cell(hide_code=True)
def _():
    n_emails = 100
    observed_conversions = 8

    _fig, _ax = plt.subplots(figsize=(7, 2.4))
    _counts = [observed_conversions, n_emails - observed_conversions]
    _ax.barh(["Converted", "Did not convert"], _counts, color=["C1", "C0"])
    for _i, _v in enumerate(_counts):
        _ax.text(_v + 1, _i, str(_v), va="center")
    _ax.set_xlabel("Number of recipients")
    _ax.set_xlim(0, n_emails + 8)
    _ax.set_title(f"Email campaign: {observed_conversions} of {n_emails} recipients converted")
    _fig.tight_layout()

    mo.vstack(
        [
            _fig,
            mo.md(
                f"**The data (from Session 1.1):** a promotional email sent to **{n_emails}** "
                f"recipients, of whom **{observed_conversions}** converted "
                f"({observed_conversions / n_emails:.0%}). You will set a prior for the "
                "conversion rate before fitting."
            ),
        ]
    )
    return n_emails, observed_conversions


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("""
        You are about to estimate the conversion rate `p` of the campaign above.
        Before fitting, you set a **prior** for `p`. Since `p` lives between 0 and 1,
        you model it on the **log-odds** scale, `logit(p)`, with a Normal, and map
        back with `invlogit`. How would we specify a vague or uninformative prior on `p`?

        In the scaffold below:

        1. Fill in a **wide** `Normal` prior on the log-odds (start with `sigma=10`)
           and sample the prior predictive.
        2. Click ▶ Run exercise. Look at the conversion counts your prior predicts for
           100 emails, and the implied prior on `p`. Are they plausible?
        3. Lower the width and re-run until the predicted conversions cover a sensible
           range instead of all-or-nothing.
        """),
        kind="info",
    )
    return


@app.cell
def _(n_emails, observed_conversions):
    def exercise_uninformative_prior():
        with pm.Model():
            # YOUR CODE HERE — a "vague" Normal prior on the log-odds
            logit_p = ...
            p = pm.Deterministic("p", pm.math.invlogit(logit_p))
            pm.Binomial("y", n=n_emails, p=p, observed=observed_conversions)
            # YOUR CODE HERE — sample the prior predictive
            prior_trace = ...

        return show_prior_predictive_check(prior_trace, n_emails)

    return (exercise_uninformative_prior,)


@app.cell(hide_code=True)
def _():
    run_uninformative_prior = mo.ui.run_button(label="▶ Run exercise")
    run_uninformative_prior
    return (run_uninformative_prior,)


@app.cell(hide_code=True)
def _(exercise_uninformative_prior, run_uninformative_prior):
    mo.stop(
        not run_uninformative_prior.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_uninformative_prior()
    return


@app.function(hide_code=True)
def show_prior_predictive_check(prior_trace, n_emails):
    observed_k = int(prior_trace.observed_data["y"])
    p = prior_trace.prior["p"].values.flatten()
    y = prior_trace.prior_predictive["y"].values.flatten()
    extreme = float(np.mean((p < 0.02) | (p > 0.98)))
    all_or_nothing = float(np.mean((y == 0) | (y == n_emails)))

    az.plot_dist(prior_trace, group="prior_predictive", var_names=["y"], kind="hist")
    fig_counts = plt.gcf()
    fig_counts.axes[0].axvline(observed_k, color="C1", lw=2, label=f"observed = {observed_k}")
    fig_counts.axes[0].set_title(f"Prior predictive conversions (out of {n_emails})")
    fig_counts.axes[0].legend()

    az.plot_dist(prior_trace, group="prior", var_names=["p"], kind="hist")
    fig_p = plt.gcf()
    fig_p.axes[0].set_title("Implied prior on the conversion rate p")

    if all_or_nothing > 0.15:
        verdict = (
            f"This prior predicts **{all_or_nothing:.0%}** of campaigns as all-or-nothing "
            f"(0 or {n_emails} conversions), and puts **{extreme:.0%}** of the prior mass on "
            "p beyond 0.02 or 0.98. That is not uninformative: a wide prior on the log-odds is "
            "betting the rate is near-certain, one way or the other."
        )
    else:
        verdict = (
            f"Now only **{all_or_nothing:.0%}** of predicted campaigns are all-or-nothing and "
            f"**{extreme:.0%}** of the mass on p is at the extremes: the prior spreads "
            "conversions over a plausible range that comfortably includes the observed count."
        )
    return mo.vstack([mo.md(verdict), fig_counts, fig_p])


@app.cell(hide_code=True)
def _(n_emails, solution_uninformative_prior):
    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        "A wide Normal on the log-odds is **not** uninformative. `invlogit` "
                        "squashes the whole real line back into (0, 1), so the long tails of a "
                        "`Normal(0, 10)` pile up against 0 and 1, and a Binomial with p near 0 or "
                        "1 produces all-or-nothing conversion counts. A `sigma` near **1.6** keeps "
                        "the implied prior on p roughly flat and the predicted counts spread over a "
                        "plausible range around the observed 8. Same trap as the Uniform "
                        "'uninformative' prior from earlier: vague on one scale can be quietly "
                        "strong on the scale you care about."
                    ),
                    mo.md(
                        f"```python\n{inspect.getsource(solution_uninformative_prior)}\n```"
                    ),
                    mo.lazy(
                        lambda: show_prior_predictive_check(
                            solution_uninformative_prior(), n_emails
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

    ## Interactive Prior Exploration with PreliZ

    [PreliZ](https://preliz.readthedocs.io/) is a library designed specifically for prior elicitation. It provides tools to find distributions that match your domain knowledge.

    Two key functions:
    - **`pz.maxent()`**: find the maximum entropy distribution matching specified quantile constraints
    - **`pz.mle()`**: fit a distribution to data using maximum likelihood
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
    A fitted PreliZ distribution can be dropped straight into a PyMC model with
    `.to_pymc("name")`; you will use this in Session 4.1.
    """)
    return


@app.cell
def _():
    # A small maxent-fitted prior, purely to demonstrate .to_pymc()
    elicited_dist = pz.Normal()
    pz.maxent(elicited_dist, lower=2500, upper=5500, mass=0.94, plot=False)
    with pm.Model():
        elicited_prior = elicited_dist.to_pymc("elicited_prior")
    elicited_prior
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
    ---

    ## Multivariate distributions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Multivariate Normal distribution

    Core form: $e^{-\frac{1}{2}\, \mathbf{x}^\top \Sigma^{-1} \mathbf{x}}$

    * Joint distribution of correlated continuous variables
    * Parameterized by a mean vector $\mu$ and covariance matrix $\Sigma$
    * All marginals and conditionals are Normal
    * The workhorse behind correlated random effects and Gaussian processes (Session 5.2)
    """)
    return


@app.cell(hide_code=True)
def _():
    mvn_rho_slider = mo.ui.slider(-0.95, 0.95, value=0.6, step=0.05, label="rho")
    mvn_sigma2_slider = mo.ui.slider(0.4, 3, value=1.0, step=0.1, label="sigma_y")
    mo.hstack([mvn_rho_slider, mvn_sigma2_slider], justify="start")
    return mvn_rho_slider, mvn_sigma2_slider


@app.cell(hide_code=True)
def _(mvn_rho_slider, mvn_sigma2_slider):
    def plot_mvn_scatter():
        rho = mvn_rho_slider.value
        sy = mvn_sigma2_slider.value
        cov = np.array([[1.0, rho * sy], [rho * sy, sy**2]])
        rng = np.random.default_rng(42)
        xy = rng.multivariate_normal([0, 0], cov, size=1000)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers",
                marker=dict(color=PYMC_BLUE, size=4, opacity=0.45),
            )
        )
        fig.update_layout(
            title=f"1000 draws from MVN(0, Σ) — rho={rho:.2f}, sigma_y={sy:.1f}",
            xaxis_title="x",
            yaxis_title="y",
            xaxis=dict(range=[-4, 4]),
            yaxis=dict(range=[-4 * max(sy, 1), 4 * max(sy, 1)], scaleanchor="x"),
            width=600,
            height=500,
        )
        return fig

    plot_mvn_scatter()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Dirichlet distribution

    Core form: $\prod_{i} x_i^{\alpha_i - 1}$ on the simplex ($x_i \geq 0$, $\sum_i x_i = 1$)

    * Multivariate generalization of the Beta, a distribution over **proportions**
    * Conjugate prior for Categorical/Multinomial probabilities
    * $\alpha_i < 1$: mass piles up at the corners; $\alpha = (1,1,1)$: uniform over the simplex; large $\alpha$: concentrated near the mean
    """)
    return


@app.cell(hide_code=True)
def _():
    dir_a1_slider = mo.ui.slider(0.1, 10, value=1.0, step=0.1, label="alpha_1")
    dir_a2_slider = mo.ui.slider(0.1, 10, value=1.0, step=0.1, label="alpha_2")
    dir_a3_slider = mo.ui.slider(0.1, 10, value=1.0, step=0.1, label="alpha_3")
    mo.hstack([dir_a1_slider, dir_a2_slider, dir_a3_slider], justify="start")
    return dir_a1_slider, dir_a2_slider, dir_a3_slider


@app.cell(hide_code=True)
def _(dir_a1_slider, dir_a2_slider, dir_a3_slider):
    def plot_dirichlet_scatter():
        alphas = (dir_a1_slider.value, dir_a2_slider.value, dir_a3_slider.value)
        rng = np.random.default_rng(42)
        samples = rng.dirichlet(alphas, size=1500)

        fig = go.Figure()
        fig.add_trace(
            go.Scatterternary(
                a=samples[:, 0],
                b=samples[:, 1],
                c=samples[:, 2],
                mode="markers",
                marker=dict(color=PYMC_BLUE, size=3.5, opacity=0.4),
            )
        )
        fig.update_layout(
            title=(
                f"1500 draws from Dirichlet({alphas[0]:g}, {alphas[1]:g}, {alphas[2]:g})"
            ),
            ternary=dict(
                aaxis=dict(title="x1"),
                baxis=dict(title="x2"),
                caxis=dict(title="x3"),
            ),
            width=600,
            height=500,
        )
        return fig

    plot_dirichlet_scatter()
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
    """)
    return


@app.cell(hide_code=True)
def _():
    lkj_eta_slider = mo.ui.slider(
        steps=[0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0], value=1.0, label="eta"
    )
    lkj_eta_slider
    return (lkj_eta_slider,)


@app.cell(hide_code=True)
def _(lkj_eta_slider):
    def plot_lkj_samples():
        eta = lkj_eta_slider.value
        p = 5
        n_show, n_hist = 6, 500

        chol = np.asarray(
            pm.draw(pm.LKJCorr.dist(n=p, eta=eta), draws=n_hist, random_seed=42)
        )
        corrs = chol @ np.swapaxes(chol, 1, 2)

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=n_show,
            horizontal_spacing=0.02,
            subplot_titles=[f"draw {i + 1}" for i in range(n_show)],
        )
        for i in range(n_show):
            fig.add_trace(
                go.Heatmap(
                    z=corrs[i][::-1],
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                    showscale=(i == n_show - 1),
                    colorbar=dict(title="r", thickness=12),
                ),
                row=1,
                col=i + 1,
            )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            title=f"Six correlation matrices drawn from LKJ(eta={eta:g}), p={p}",
            width=950,
            height=220,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        off_diag = corrs[:, *np.triu_indices(p, k=1)].flatten()
        hist = go.Figure()
        hist.add_trace(
            go.Histogram(
                x=off_diag,
                xbins=dict(start=-1, end=1, size=0.05),
                histnorm="probability density",
                marker_color=PYMC_BLUE,
                opacity=0.7,
            )
        )
        hist.update_layout(
            title=f"Marginal distribution of off-diagonal correlations ({n_hist} draws)",
            xaxis_title="pairwise correlation r",
            yaxis_title="Density",
            xaxis=dict(range=[-1, 1]),
            width=950,
            height=280,
            margin=dict(l=20, r=20, t=60, b=40),
        )
        return mo.vstack([fig, hist])

    plot_lkj_samples()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Summary

    - Match your likelihood to the data-generating process (Normal, Lognormal, Poisson, NegBin, etc.)
    - Multivariate families: **MVN** for correlated continuous variables, **Dirichlet** for proportions
    - Model the *observation process* too: **`pm.Censored`** / **`pm.Truncated`** take any base distribution and model how it is observed
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
def _(n_emails, observed_conversions):
    def solution_uninformative_prior():
        with pm.Model():
            logit_p = pm.Normal("logit_p", mu=0, sigma=1.6)
            p = pm.Deterministic("p", pm.math.invlogit(logit_p))
            pm.Binomial("y", n=n_emails, p=p, observed=observed_conversions)
            prior_trace = pm.sample_prior_predictive(2000, random_seed=42)
        return prior_trace

    return (solution_uninformative_prior,)


if __name__ == "__main__":
    app.run()
