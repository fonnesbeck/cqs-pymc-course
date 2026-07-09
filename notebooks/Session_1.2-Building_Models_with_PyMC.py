import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
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
    pio.templates["pymc_labs"] = pymc_template
    pio.templates.default = "plotly_white+pymc_labs"

    # Data path
    data_path = Path(__file__).parent / "data"
    return (
        PYMC_BLUE,
        PYMC_GREEN,
        PYMC_LIGHT_BLUE,
        Path,
        base64,
        data_path,
        go,
        inspect,
        np,
        pl,
        plt,
        pm,
        pz,
        stats,
    )


@app.cell(hide_code=True)
def _(Path, base64, mo):
    def make_header():
        logo_path = Path(__file__).parent / "images" / "pymc-labs-logo.png"
        if logo_path.exists():
            logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="300" style="margin-bottom: 0.5rem;">'
        else:
            logo_html = ""
        return logo_html

    mo.md(f"""
    {make_header()}

    # Session 1B: Prior and Likelihood Selection

    This session guides you through choosing appropriate priors and likelihoods for your models.

    **Topics:**
    Distribution families, choosing likelihoods for different data types, prior predictive simulation, and interactive prior exploration with PreliZ

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part B: Prior and Likelihood Selection

    ---

    ## Distribution Families

    Getting comfortable with common probability distributions will make it easier to specify prior distributions and models for data. Let's explore some key families interactively.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continuous distributions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
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
def _(mo):
    normal_mu_slider = mo.ui.slider(-5, 5, value=0, step=0.1, label="mu")
    normal_sigma_slider = mo.ui.slider(0.4, 3, value=1, step=0.1, label="sigma")
    mo.hstack([normal_mu_slider, normal_sigma_slider], justify="start")
    return normal_mu_slider, normal_sigma_slider


@app.cell(hide_code=True)
def _(PYMC_BLUE, go, normal_mu_slider, normal_sigma_slider, np, stats):
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
def _(mo):
    mo.md(r"""
    #### When does the Normal apply?

    The Normal distribution is a good model when the variability is just **symmetric, unstructured noise** — no hidden subgroups, no skew, no heavy tails. It's the maximum entropy distribution for a given mean and variance, so it's the "least informative" choice when all you know is the center and spread.

    But if the data contains **unmodeled structure**, the Normal breaks down. Consider human heights: if we mix males and females, the combined distribution is bimodal. More data makes the bimodality *sharper*, not more Normal. However, **conditioned on gender**, each subgroup is well-described by a Normal — the remaining variability is just noise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    height_n_slider = mo.ui.slider(50, 10_000, value=500, step=50, label="n (samples)")
    height_condition_toggle = mo.ui.switch(label="Condition on gender", value=False)
    mo.hstack([height_n_slider, height_condition_toggle], justify="start")
    return height_condition_toggle, height_n_slider


@app.cell(hide_code=True)
def _(
    PYMC_BLUE,
    PYMC_GREEN,
    PYMC_LIGHT_BLUE,
    go,
    height_condition_toggle,
    height_n_slider,
    np,
    stats,
):
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
            fig.update_layout(title=f"Combined heights (n={n:,}) — best Normal fit is poor")
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
def _(mo):
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
def _(mo):
    t_mu_slider = mo.ui.slider(-5, 5, value=0, step=0.1, label="mu")
    t_sigma_slider = mo.ui.slider(1, 3, value=1, step=0.1, label="sigma")
    t_nu_slider = mo.ui.slider(1, 30, value=3, step=.5, label="ν")
    mo.hstack([t_mu_slider, t_sigma_slider, t_nu_slider], justify="start")
    return t_mu_slider, t_nu_slider, t_sigma_slider


@app.cell(hide_code=True)
def _(PYMC_BLUE, go, np, stats, t_mu_slider, t_nu_slider, t_sigma_slider):
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
                x=x, y=y_t, mode="lines", fill="tozeroy",
                line=dict(color=PYMC_BLUE, width=2),
                name=f"StudentT(ν={nu})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=y_norm, mode="lines",
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
def _(mo):
    mo.md(r"""
    #### Robustness to outliers

    The Student-t's heavy tails make it **robust to outliers**. When an outlier is present, the Normal's best fit distorts to accommodate it (wider σ), while the Student-t barely changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    outlier_toggle = mo.ui.switch(label="Include outlier", value=False)
    cluster_n_slider = mo.ui.slider(10, 200, value=30, step=5, label="n (cluster size)")
    mo.hstack([cluster_n_slider, outlier_toggle], justify="start")
    return cluster_n_slider, outlier_toggle


@app.cell(hide_code=True)
def _(
    PYMC_BLUE,
    PYMC_GREEN,
    PYMC_LIGHT_BLUE,
    cluster_n_slider,
    go,
    np,
    outlier_toggle,
    stats,
):
    def plot_outlier_robustness():
        n = cluster_n_slider.value
        include_outlier = outlier_toggle.value
        rng = np.random.default_rng(42)

        cluster = rng.normal(5, 0.8, size=n)
        outlier_val = 15.0

        if include_outlier:
            data = np.append(cluster, outlier_val)
        else:
            data = cluster

        x_grid = np.linspace(0, 18, 300)

        # Best-fit Normal
        mu_n, sigma_n = data.mean(), data.std()

        # Best-fit Student-t (MLE)
        t_df, t_loc, t_scale = stats.t.fit(data)

        fig = go.Figure()

        # Histogram of data
        fig.add_trace(
            go.Histogram(
                x=data,
                xbins=dict(size=0.3),
                histnorm="probability density",
                marker_color=PYMC_BLUE,
                opacity=0.5 if include_outlier else 0.7,
                name="Data",
            )
        )

        # Best-fit Normal
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=stats.norm.pdf(x_grid, mu_n, sigma_n),
                mode="lines",
                line=dict(color=PYMC_GREEN, width=2.5, dash="dash"),
                name=f"Normal (μ={mu_n:.2f}, σ={sigma_n:.2f})",
            )
        )

        # Best-fit Student-t
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=stats.t.pdf(x_grid, t_df, t_loc, t_scale),
                mode="lines",
                line=dict(color=PYMC_LIGHT_BLUE, width=2.5),
                name=f"StudentT (ν={t_df:.1f}, μ={t_loc:.2f}, σ={t_scale:.2f})",
            )
        )

        fig.update_layout(
            title="Normal vs Student-t fit — toggle the outlier",
            xaxis_title="x",
            yaxis_title="Density",
            width=700,
            height=380,
        )
        return fig

    plot_outlier_robustness()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### HalfNormal distribution

    Core form: $e^{-x^2}$ for $x \geq 0$

    * A Normal folded at zero — only the positive half
    * Single parameter: $\sigma$ (scale)
    * Common prior for standard deviations and other positive scale parameters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    hn_sigma_slider = mo.ui.slider(0.5, 5, value=1, step=0.1, label="sigma")
    hn_sigma_slider
    return (hn_sigma_slider,)


@app.cell(hide_code=True)
def _(PYMC_BLUE, PYMC_GREEN, go, hn_sigma_slider, np, stats):
    def plot_halfnormal_pdf():
        sigma = hn_sigma_slider.value
        x = np.linspace(0, 15, 300)
        x_full = np.linspace(-15, 15, 300)
        y = stats.halfnorm.pdf(x, scale=sigma)
        y_full = stats.norm.pdf(x_full, 0, sigma)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_full, y=y_full, mode="lines",
                line=dict(color=PYMC_GREEN, width=1.5, dash="dash"),
                name=f"Normal(0, {sigma:.1f})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", fill="tozeroy",
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
def _(mo):
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
def _(mo):
    ln_mu_slider = mo.ui.slider(-1, 3, value=0, step=0.1, label="mu")
    ln_sigma_slider = mo.ui.slider(0.1, 2.5, value=0.5, step=0.1, label="sigma")
    mo.hstack([ln_mu_slider, ln_sigma_slider], justify="start")
    return ln_mu_slider, ln_sigma_slider


@app.cell(hide_code=True)
def _(PYMC_BLUE, go, ln_mu_slider, ln_sigma_slider, mo, np, stats):
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
                x=x, y=y, mode="lines", fill="tozeroy",
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
def _(mo):
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
def _(mo):
    gamma_alpha_slider = mo.ui.slider(0.5, 20, value=2, step=0.1, label="alpha (shape)")
    gamma_beta_slider = mo.ui.slider(0.1, 5, value=1, step=0.1, label="beta (rate)")
    mo.hstack([gamma_alpha_slider, gamma_beta_slider], justify="start")
    return gamma_alpha_slider, gamma_beta_slider


@app.cell(hide_code=True)
def _(
    PYMC_BLUE,
    PYMC_GREEN,
    gamma_alpha_slider,
    gamma_beta_slider,
    go,
    mo,
    np,
    stats,
):
    def plot_gamma_pdf():
        a = gamma_alpha_slider.value
        b = gamma_beta_slider.value
        mean = a / b
        x = np.linspace(0, 40, 300)
        y = stats.gamma.pdf(x, a=a, scale=1.0 / b)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", fill="tozeroy",
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
def _(mo):
    mo.md(r"""
    ### Uniform distribution

    Core form: $\frac{1}{b-a}$ for $a \leq x \leq b$

    * Constant density over the interval $[a, b]$
    * Maximum entropy distribution for a bounded variable with known support
    * Often considered "uninformative" but is actually quite strong — it says all values in the range are equally likely and values outside are impossible
    * Special case of Beta(1, 1) on [0, 1]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    unif_a_slider = mo.ui.slider(-3, 1, value=0, step=0.5, label="a (lower)")
    unif_b_slider = mo.ui.slider(-1, 3, value=1, step=0.5, label="b (upper)")
    mo.hstack([unif_a_slider, unif_b_slider], justify="start")
    return unif_a_slider, unif_b_slider


@app.cell(hide_code=True)
def _(PYMC_BLUE, PYMC_GREEN, go, mo, np, stats, unif_a_slider, unif_b_slider):
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
                x=x, y=y, mode="lines", fill="tozeroy",
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
def _(mo):
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
def _(mo):
    beta_alpha_slider = mo.ui.slider(0.1, 20, value=2, step=0.1, label="alpha")
    beta_beta_slider = mo.ui.slider(0.1, 20, value=5, step=0.1, label="beta")
    mo.hstack([beta_alpha_slider, beta_beta_slider], justify="start")
    return beta_alpha_slider, beta_beta_slider


@app.cell(hide_code=True)
def _(
    PYMC_BLUE,
    PYMC_GREEN,
    beta_alpha_slider,
    beta_beta_slider,
    go,
    mo,
    np,
    stats,
):
    def plot_beta_pdf():
        a = beta_alpha_slider.value
        b = beta_beta_slider.value
        x = np.linspace(0, 1, 300)
        y = stats.beta.pdf(x, a, b)
        mean = a / (a + b)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines", fill="tozeroy",
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
def _(mo):
    mo.md(r"""
    ## Discrete distributions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bernoulli distribution

    Core form: $p^k(1-p)^{1-k}$ for $k \in \{0, 1\}$

    * The simplest discrete distribution — a single trial with two outcomes
    * Building block for the Binomial (sum of Bernoullis)
    * Generalizes to multivariate as Categorical
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bern_p_slider = mo.ui.slider(0.01, 0.99, value=0.3, step=0.01, label="p")
    bern_p_slider
    return (bern_p_slider,)


@app.cell(hide_code=True)
def _(PYMC_BLUE, bern_p_slider, go):
    def plot_bernoulli_pmf():
        p = bern_p_slider.value
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[0, 1], y=[1 - p, p],
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
def _(mo):
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
def _(mo):
    binom_n_slider = mo.ui.slider(1, 50, value=10, step=1, label="n (trials)")
    binom_p_slider = mo.ui.slider(0.01, 0.99, value=0.3, step=0.01, label="p")
    mo.hstack([binom_n_slider, binom_p_slider], justify="start")
    return binom_n_slider, binom_p_slider


@app.cell(hide_code=True)
def _(PYMC_BLUE, binom_n_slider, binom_p_slider, go, mo, np, stats):
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
                x=x, y=y,
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
def _(mo):
    mo.md(r"""
    ### Poisson distribution

    Core form: $\frac{\lambda^k\,e^{-\lambda}}{k!}$ for $k = 0, 1, 2, \ldots$

    * No upper bound on counts (unlike Binomial)
    * Limit of Binomial as $n \to \infty$, $p \to 0$, $np \to \lambda$
    * **Key constraint:** mean = variance. If your data violates this, Poisson is wrong.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pois_lambda_slider = mo.ui.slider(1, 50, value=10, step=1, label="lambda")
    pois_lambda_slider
    return (pois_lambda_slider,)


@app.cell(hide_code=True)
def _(PYMC_BLUE, go, mo, np, pois_lambda_slider, stats):
    def plot_poisson_pmf():
        lam = pois_lambda_slider.value
        x = np.arange(0, 100)
        y = stats.poisson.pmf(x, lam)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x, y=y,
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
def _(mo):
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


@app.cell(hide_code=True)
def _(mo):
    nb_mu_slider = mo.ui.slider(1, 50, value=10, step=1, label="mu")
    nb_alpha_slider = mo.ui.slider(0.5, 100, value=5, step=0.5, label="alpha")
    mo.hstack([nb_mu_slider, nb_alpha_slider], justify="start")
    return nb_alpha_slider, nb_mu_slider


@app.cell(hide_code=True)
def _(PYMC_BLUE, PYMC_GREEN, go, mo, nb_alpha_slider, nb_mu_slider, np, stats):
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
                x=x, y=y_nb,
                marker_color=PYMC_BLUE,
                opacity=0.7,
                name=f"NegBin(μ={mu}, α={alpha})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=y_pois,
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
    mo.vstack([_fig, mo.md(f"**Mean:** {_mu} | **Variance:** {_var:.1f} (Poisson would force variance = {_mu})")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Overdispersion: when Poisson fails

    Real count data often has variance much larger than the mean — this is **overdispersion**. The Poisson can't capture it (its mean *is* its variance), so it either overestimates the peak or underestimates the spread. The Negative Binomial handles this naturally via its extra parameter $\alpha$.

    Below: samples are drawn from a NegBin. As you decrease $\alpha$, the data becomes more overdispersed and the best-fit Poisson increasingly fails.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    od_alpha_slider = mo.ui.slider(0.5, 50, value=3, step=0.5, label="alpha (lower = more overdispersed)")
    od_n_slider = mo.ui.slider(50, 2000, value=500, step=50, label="n (samples)")
    mo.hstack([od_alpha_slider, od_n_slider], justify="start")
    return od_alpha_slider, od_n_slider


@app.cell(hide_code=True)
def _(
    PYMC_BLUE,
    PYMC_GREEN,
    PYMC_LIGHT_BLUE,
    go,
    np,
    od_alpha_slider,
    od_n_slider,
    stats,
):
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
                x=x, y=stats.poisson.pmf(x, lam_fit),
                mode="lines+markers",
                line=dict(color=PYMC_GREEN, width=2.5, dash="dash"),
                marker=dict(size=4),
                name=f"Best Poisson (λ={lam_fit:.1f})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=stats.nbinom.pmf(x, alpha_fit, p_fit),
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
def _(mo):
    mo.md(r"""
    ---

    ## Multivariate distributions

    Multivariate distributions model vectors rather than single numbers. They are useful when
    the components are constrained to work together, or when the dependence between variables
    is the thing you need to model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dirichlet distribution

    Core form:

    $$
    p(\mathbf{x}) =
    \frac{\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)}
    \prod_i x_i^{\alpha_i - 1}
    $$

    for $x_i \ge 0$ and $\sum_i x_i = 1$.

    * A distribution over **compositions**: vectors of positive parts that sum to one
    * Common prior for Categorical or Multinomial probabilities
    * $\alpha_i < 1$ pushes mass toward corners; $\alpha_i = 1$ is uniform; larger values concentrate around the mean
    * Mean component: $E[x_i] = \alpha_i / \sum_j \alpha_j$

    The plot below shows the 3-part Dirichlet on a simplex. Each point is a possible
    probability vector `(category 1, category 2, category 3)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dirichlet_alpha_1_slider = mo.ui.slider(
        0.5, 20, value=2, step=0.5, label="alpha 1"
    )
    dirichlet_alpha_2_slider = mo.ui.slider(
        0.5, 20, value=2, step=0.5, label="alpha 2"
    )
    dirichlet_alpha_3_slider = mo.ui.slider(
        0.5, 20, value=2, step=0.5, label="alpha 3"
    )
    mo.hstack(
        [
            dirichlet_alpha_1_slider,
            dirichlet_alpha_2_slider,
            dirichlet_alpha_3_slider,
        ],
        justify="start",
    )
    return (
        dirichlet_alpha_1_slider,
        dirichlet_alpha_2_slider,
        dirichlet_alpha_3_slider,
    )


@app.cell(hide_code=True)
def _(
    PYMC_GREEN,
    dirichlet_alpha_1_slider,
    dirichlet_alpha_2_slider,
    dirichlet_alpha_3_slider,
    go,
    mo,
    np,
    stats,
):
    def plot_dirichlet_simplex():
        alpha = np.array(
            [
                dirichlet_alpha_1_slider.value,
                dirichlet_alpha_2_slider.value,
                dirichlet_alpha_3_slider.value,
            ],
            dtype=float,
        )
        eps = 0.01
        step = 0.025
        points = []
        for x1 in np.arange(eps, 1 - eps, step):
            for x2 in np.arange(eps, 1 - x1 - eps, step):
                x3 = 1 - x1 - x2
                if x3 >= eps:
                    points.append((x1, x2, x3))
        points = np.asarray(points)
        density = np.array([stats.dirichlet.pdf(point, alpha) for point in points])
        log_density = np.log(density)
        mean = alpha / alpha.sum()

        fig = go.Figure()
        fig.add_trace(
            go.Scatterternary(
                a=points[:, 0],
                b=points[:, 1],
                c=points[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=log_density,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="log density"),
                ),
                name="Density grid",
                hovertemplate=(
                    "p1=%{a:.2f}<br>p2=%{b:.2f}<br>p3=%{c:.2f}"
                    "<br>log density=%{marker.color:.2f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatterternary(
                a=[mean[0]],
                b=[mean[1]],
                c=[mean[2]],
                mode="markers",
                marker=dict(size=14, color=PYMC_GREEN, symbol="star"),
                name="Mean",
                hovertemplate="mean=(%{a:.2f}, %{b:.2f}, %{c:.2f})<extra></extra>",
            )
        )
        fig.update_layout(
            title=(
                "Dirichlet("
                f"{alpha[0]:.1f}, {alpha[1]:.1f}, {alpha[2]:.1f})"
            ),
            ternary=dict(
                sum=1,
                aaxis=dict(title="category 1"),
                baxis=dict(title="category 2"),
                caxis=dict(title="category 3"),
            ),
            width=700,
            height=500,
        )
        return fig, mean, alpha.sum()

    _fig, _mean, _concentration = plot_dirichlet_simplex()
    mo.vstack([
        _fig,
        mo.md(
            f"**Mean:** ({_mean[0]:.2f}, {_mean[1]:.2f}, {_mean[2]:.2f}) | "
            f"**Total concentration:** {_concentration:.1f}"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multivariate Normal distribution

    Core form:

    $$
    p(\mathbf{x}) =
    \frac{1}{(2\pi)^{k/2} |\Sigma|^{1/2}}
    \exp\left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top
    \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]
    $$

    * A distribution over continuous vectors
    * The mean vector $\boldsymbol{\mu}$ controls location
    * The covariance matrix $\Sigma$ controls marginal scales and correlations
    * Off-diagonal covariance creates dependence: knowing one component changes what you expect about the other

    The plot below shows a two-dimensional Normal with mean `(0, 0)`. The contours are the
    density; the dots are random draws.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mvn_sigma_x_slider = mo.ui.slider(0.5, 3.0, value=1.0, step=0.1, label="sigma x")
    mvn_sigma_y_slider = mo.ui.slider(0.5, 3.0, value=1.5, step=0.1, label="sigma y")
    mvn_rho_slider = mo.ui.slider(-0.95, 0.95, value=0.6, step=0.05, label="rho")
    mo.hstack(
        [mvn_sigma_x_slider, mvn_sigma_y_slider, mvn_rho_slider],
        justify="start",
    )
    return mvn_rho_slider, mvn_sigma_x_slider, mvn_sigma_y_slider


@app.cell(hide_code=True)
def _(
    PYMC_GREEN,
    go,
    mo,
    mvn_rho_slider,
    mvn_sigma_x_slider,
    mvn_sigma_y_slider,
    np,
    stats,
):
    def plot_multivariate_normal():
        sigma_x = mvn_sigma_x_slider.value
        sigma_y = mvn_sigma_y_slider.value
        rho = mvn_rho_slider.value
        cov = np.array(
            [
                [sigma_x**2, rho * sigma_x * sigma_y],
                [rho * sigma_x * sigma_y, sigma_y**2],
            ]
        )
        dist = stats.multivariate_normal(mean=[0, 0], cov=cov)
        x = np.linspace(-4 * sigma_x, 4 * sigma_x, 120)
        y = np.linspace(-4 * sigma_y, 4 * sigma_y, 120)
        xx, yy = np.meshgrid(x, y)
        grid = np.dstack((xx, yy))
        density = dist.pdf(grid)

        rng = np.random.default_rng(42)
        draws = rng.multivariate_normal(mean=[0, 0], cov=cov, size=400)

        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=density,
                colorscale="Blues",
                contours=dict(showlabels=False),
                showscale=False,
                name="Density",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=draws[:, 0],
                y=draws[:, 1],
                mode="markers",
                marker=dict(size=4, color=PYMC_GREEN, opacity=0.45),
                name="Draws",
            )
        )
        fig.add_vline(x=0, line=dict(color="gray", dash="dot"))
        fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
        fig.update_layout(
            title=(
                f"Multivariate Normal: σx={sigma_x:.1f}, "
                f"σy={sigma_y:.1f}, ρ={rho:.2f}"
            ),
            xaxis=dict(title="x", zeroline=False),
            yaxis=dict(title="y", zeroline=False, scaleanchor="x", scaleratio=1),
            width=700,
            height=500,
        )
        covariance = rho * sigma_x * sigma_y
        return fig, cov, covariance

    _fig, _cov, _covariance = plot_multivariate_normal()
    mo.vstack([
        _fig,
        mo.md(
            f"**Covariance matrix:** "
            f"`[[{_cov[0, 0]:.2f}, {_cov[0, 1]:.2f}], "
            f"[{_cov[1, 0]:.2f}, {_cov[1, 1]:.2f}]]` | "
            f"**Cov(x, y):** {_covariance:.2f}"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Priors for Correlation Matrices: The LKJ Distribution

    Multivariate Normal models often need a prior on the covariance matrix. It is usually
    easier to separate that into:

    1. standard deviations for each variable, and
    2. a correlation matrix.

    Correlation matrices have special constraints (symmetric, positive definite, diagonal = 1),
    so we can't use standard univariate priors on each entry independently.

    The **LKJ distribution** (Lewandowski, Kurowicka, Joe) is designed for correlation
    matrices. It has a single shape parameter $\eta$:

    - $\eta = 1$: uniform over valid correlation matrices
    - $\eta > 1$: favours matrices closer to the identity (weak correlations)
    - $\eta < 1$: favours matrices with strong correlations

    In PyMC, we use `pm.LKJCholeskyCov` which jointly samples a correlation matrix and
    standard deviations:

    ```python
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol", n=p, eta=2.0,
        sd_dist=pm.Exponential.dist(1.0),
    )
    ```

    `chol` can then be passed as the Cholesky factor of the covariance matrix in a
    multivariate Normal model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Choosing distributions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
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
def _(mo):
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
def _(mo):
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
def _(mo):
    mo.md(r"""
    ### Example: Penguin Body Mass

    Let's model the body mass of Adelie penguins. We'll compare a **vague** prior with a **weakly informative** prior.
    """)
    return


@app.cell
def _(data_path, pl):
    penguins_df = pl.read_csv(data_path / "penguins.csv", null_values="NA").drop_nulls()
    penguin_mass = penguins_df["body_mass_g"].to_numpy()
    adelie_mass = penguins_df.filter(pl.col("species") == "Adelie")[
        "body_mass_g"
    ].to_numpy()
    penguins_df
    return adelie_mass, penguin_mass, penguins_df


@app.cell
def _(penguins_df):
    penguins_df["body_mass_g"].plot.hist()
    return


@app.cell
def _(np, pm):
    # Vague prior: centered on a plausible penguin mass, but too diffuse on the log scale
    with pm.Model():
        mu = pm.Normal("mu", mu=np.log(4000), sigma=0.8)
        sigma = pm.HalfNormal("sigma", sigma=1)
        pm.LogNormal("y", mu=mu, sigma=sigma)
        vague_prior = pm.sample_prior_predictive(1000, random_seed=2316)
    return (vague_prior,)


@app.cell
def _(np, pm):
    # Weakly informative prior
    with pm.Model():
        mu2 = pm.Normal("mu", mu=np.log(4000), sigma=0.1)
        sigma2 = pm.HalfNormal("sigma", sigma=0.15)
        pm.LogNormal("y", mu=mu2, sigma=sigma2)
        informed_prior = pm.sample_prior_predictive(1000, random_seed=2316)
    return (informed_prior,)


@app.cell
def _(mo):
    prior_pred_log_toggle = mo.ui.switch(label="Log x-axis", value=True)
    prior_pred_log_toggle
    return (prior_pred_log_toggle,)


@app.cell(hide_code=True)
def _(informed_prior, mo, np, plt, prior_pred_log_toggle, vague_prior):
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
                        ax.text(val, ymax * 0.95, f" {label}", fontsize=8, color="gray", va="top", ha="left")

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
def _(mo):
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
def _(mo):
    mo.md(r"""
    ### `pz.maxent()`: Maximum Entropy Priors

    If you know that 95% of values should fall between two bounds, `pz.maxent` finds the "least informative" (maximum entropy) distribution consistent with that constraint.
    """)
    return


@app.cell
def _(pz):
    # Example: we think penguin mass is between 2500g and 5500g with 94% probability
    maxent_normal_dist = pz.Normal()
    pz.maxent(maxent_normal_dist, lower=2500, upper=5500, mass=0.94)
    return


@app.cell
def _(pz):
    # Example: response time in ms — we think 94% are between 50ms and 2000ms
    maxent_lognormal_dist = pz.LogNormal()
    pz.maxent(maxent_lognormal_dist, lower=50, upper=2000, mass=0.94)
    return


@app.cell
def _(pz):
    # Example: a rate parameter — we think 94% of values are between 0.1 and 5.0
    maxent_gamma_dist = pz.Gamma()
    pz.maxent(maxent_gamma_dist, lower=0.1, upper=5.0, mass=0.94)
    return


@app.cell
def _(pz):
    # Example: conversion rates — we think 94% of rates are between 0.01 and 0.30
    maxent_beta_dist = pz.Beta()
    pz.maxent(maxent_beta_dist, lower=0.01, upper=0.30, mass=0.94)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `pz.mle()`: Fitting Distributions to Data

    If you have historical data from a similar process, you can fit a distribution to it and use that as your prior.
    """)
    return


@app.cell
def _(adelie_mass, pz):
    # Fit a Normal to the Adelie penguin mass data
    mle_dist = pz.Normal()
    pz.mle([mle_dist], adelie_mass);
    return (mle_dist,)


@app.cell(hide_code=True)
def _(adelie_mass, mle_dist, mo, plt):
    _ax = mle_dist.plot_pdf()
    _ax.set_xlabel("Body mass (g)")
    _ax.set_title("MLE-fit Normal to Adelie penguin mass")
    _ax.hist(adelie_mass, bins=30, density=True, alpha=0.3, color="gray")
    _fig = plt.gcf()
    _fig.set_size_inches(7, 2)
    mo.vstack([
        mo.md(f"**MLE fit:** Normal(mu={mle_dist.mu:.0f}, sigma={mle_dist.sigma:.0f})"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Exercise: Penguin Body Mass Workflow
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""
        Use the cleaned penguin body-mass data for every part of this exercise.

        **Part A — Normal model and prior predictive checks**
        1. Build a PyMC model with a `Normal` likelihood for `body_mass_g`.
        2. Choose initial priors for `mu` and `sigma`.
        3. Draw from `pm.sample_prior_predictive()` before fitting.
        4. Fit the model with `pm.sample()` and draw from the posterior predictive.

        **Part B — Elicit priors with PreliZ**
        1. Use `pz.maxent()` to encode domain knowledge rather than hand-picking parameters.
        2. Convert the PreliZ distributions to PyMC priors with `.to_pymc(...)`.
        3. Refit the Normal model and compare the prior predictive distribution to Part A.

        **Part C — Try a mixture model**
        1. Replace the single Normal likelihood with `pm.NormalMixture`.
        2. Use two mixture components: one for lighter penguins and one for heavier penguins.
        3. Sample from the posterior predictive and compare it with the observed mass distribution.

        The question to answer at the end: does the mixture model capture structure that the
        single-Normal model misses?
        """),
        kind="info",
    )
    return


@app.cell
def _(penguin_mass, pm):
    def fit_penguin_normal_model():
        with pm.Model():
            mu = ...
            sigma = ...
            pm.Normal("obs", mu=mu, sigma=sigma, observed=penguin_mass)

            prior_idata = ...
            idata = ...
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        return prior_idata, idata

    def fit_penguin_preliz_model():
        mu_prior = ...
        sigma_prior = ...

        with pm.Model():
            mu = ...
            sigma = ...
            pm.Normal("obs", mu=mu, sigma=sigma, observed=penguin_mass)

            prior_idata = ...
            idata = ...
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        return mu_prior, sigma_prior, prior_idata, idata

    def fit_penguin_mixture_model():
        with pm.Model():
            weights = ...
            component_mu = ...
            component_sigma = ...
            pm.NormalMixture(
                "obs",
                w=weights,
                mu=component_mu,
                sigma=component_sigma,
                observed=penguin_mass,
            )

            idata = ...
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        return idata

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Exercise: British Coal Mining Disasters
    """)
    return


@app.cell(hide_code=True)
def _(go, np):
    coal_disasters = np.array(
        [
            4, 5, 4, 0, 1, 4, 3, 4, 0, 6,
            3, 3, 4, 0, 2, 6, 3, 3, 5, 4,
            5, 3, 1, 4, 4, 1, 5, 5, 3, 4,
            2, 5, 2, 2, 3, 4, 2, 1, 3, 2,
            2, 1, 1, 1, 1, 3, 0, 0, 1, 0,
            1, 1, 0, 0, 3, 1, 0, 3, 2, 2,
            0, 1, 1, 1, 0, 1, 0, 1, 0, 0,
            0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
            3, 3, 1, 1, 2, 1, 1, 1, 1, 2,
            4, 2, 0, 0, 1, 4, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
            1,
        ],
        dtype=int,
    )
    coal_years = np.arange(1851, 1851 + coal_disasters.size)
    coal_summary = {
        "first_year": int(coal_years.min()),
        "last_year": int(coal_years.max()),
        "n_years": int(coal_disasters.size),
        "total_disasters": int(coal_disasters.sum()),
        "mean_disasters_per_year": float(coal_disasters.mean()),
        "max_disasters_in_a_year": int(coal_disasters.max()),
    }

    _fig = go.Figure()
    _fig.add_trace(
        go.Bar(
            x=coal_years,
            y=coal_disasters,
            marker_color="#154A72",
            hovertemplate="Year %{x}<br>Disasters %{y}<extra></extra>",
        )
    )
    _fig.update_layout(
        title="British coal mining disasters per year",
        xaxis_title="Year",
        yaxis_title="Number of disasters",
        width=750,
        height=350,
    )
    _fig.update_yaxes(dtick=1)
    _fig
    return coal_disasters, coal_summary, coal_years


@app.cell(hide_code=True)
def _(coal_summary, mo):
    mo.callout(
        mo.md(f"""
        The British coal mining disasters dataset records the number of major coal mining
        disasters in the United Kingdom for each year from **{coal_summary["first_year"]}**
        through **{coal_summary["last_year"]}**.

        Each row is one calendar year. The observed value is a non-negative integer count:
        how many disasters occurred that year. Across {coal_summary["n_years"]} years, the
        dataset contains {coal_summary["total_disasters"]} disasters, with a maximum of
        {coal_summary["max_disasters_in_a_year"]} disasters in one year and an average of
        {coal_summary["mean_disasters_per_year"]:.2f} disasters per year.

        Historically, the rate appears higher in the early part of the series and lower
        later on. A plausible scientific story is that safety practices, regulation, and
        industry changes reduced the rate, but we do not know exactly when the change
        happened or how large it was.

        **Start from the data-generating process, not from PyMC syntax:**

        1. What would you choose for the likelihood(s)?
           - What is the support of the observations?
           - Is there a fixed maximum count per year?
           - Should every year share one rate, or might different periods have different rates?
        2. What are the unknown variables in the model?
           - One disaster rate for all years?
           - Separate early and late disaster rates?
           - An unknown change point year?
        3. Choose priors for the unknown variables.
           - Use the observed scale only as a reality check: rates near 0--6 disasters/year are plausible.
           - Avoid priors that put most mass near values like 0.01 disasters/year unless that is truly your belief.
        4. Draw from the prior predictive distribution. Do the simulated annual counts look plausible?
        5. Fit the model and draw from the posterior predictive distribution. Does the model reproduce the observed pattern?

        **Optional extension:** compare a single-rate model with a change-point model. Which one better
        captures the decline in disasters over time?
        """),
        kind="info",
    )
    return


@app.cell
def _(coal_disasters, pm):
    def fit_coal_disasters_model():
        with pm.Model():
            # Choose priors for the unknown rate(s) and, if needed, a change point.
            early_rate = ...
            late_rate = ...
            change_point = ...

            # Build a rate for every year, then connect it to the observed counts.
            disaster_rate = ...
            pm.Poisson("disasters", mu=disaster_rate, observed=coal_disasters)

            prior_idata = ...
            idata = ...
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)

        return prior_idata, idata

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Solution
    """)
    return


@app.cell(hide_code=True)
def _(
    inspect,
    mo,
    solution_coal_mining_disasters_workflow,
    solution_penguin_body_mass_workflow,
):
    mo.accordion(
        {
            "Penguin body mass workflow": mo.md(
                f"```python\n{inspect.getsource(solution_penguin_body_mass_workflow)}\n```"
            ),
            "Coal mining disasters workflow": mo.md(
                f"```python\n{inspect.getsource(solution_coal_mining_disasters_workflow)}\n```"
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(coal_disasters, coal_years, np, penguin_mass, pm, pz):
    def solution_penguin_body_mass_workflow():
        # Part A: hand-specified Normal model
        with pm.Model():
            mu = pm.Normal("mu", mu=4_200, sigma=700)
            sigma = pm.HalfNormal("sigma", sigma=700)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=penguin_mass)
            normal_prior_idata = pm.sample_prior_predictive(random_seed=42)
            normal_idata = pm.sample(random_seed=42, nuts_sampler="nutpie")
            pm.sample_posterior_predictive(
                normal_idata,
                random_seed=42,
                extend_inferencedata=True,
            )

        # Part B: priors elicited from domain knowledge with PreliZ
        mu_prior = pz.Normal()
        pz.maxent(mu_prior, lower=2_700, upper=6_300, mass=0.94, plot=False)

        sigma_prior = pz.Gamma()
        pz.maxent(sigma_prior, lower=100, upper=1_200, mass=0.94, plot=False)

        with pm.Model():
            mu = mu_prior.to_pymc("mu")
            sigma = sigma_prior.to_pymc("sigma")
            pm.Normal("obs", mu=mu, sigma=sigma, observed=penguin_mass)
            preliz_prior_idata = pm.sample_prior_predictive(random_seed=42)
            preliz_idata = pm.sample(random_seed=42, nuts_sampler="nutpie")
            pm.sample_posterior_predictive(
                preliz_idata,
                random_seed=42,
                extend_inferencedata=True,
            )

        # Part C: two-component Normal mixture for the all-species mass distribution
        with pm.Model():
            weights = pm.Dirichlet("weights", a=np.ones(2))
            component_mu = pm.Normal(
                "component_mu",
                mu=mu_prior.mu,
                sigma=mu_prior.sigma,
                shape=2,
                transform=pm.distributions.transforms.ordered,
                initval=np.array([3_500, 5_000]),
            )
            component_sigma = pm.Gamma(
                "component_sigma",
                alpha=sigma_prior.alpha,
                beta=sigma_prior.beta,
                shape=2,
            )
            pm.NormalMixture(
                "obs",
                w=weights,
                mu=component_mu,
                sigma=component_sigma,
                observed=penguin_mass,
            )
            mixture_idata = pm.sample(random_seed=42, nuts_sampler="nutpie")
            pm.sample_posterior_predictive(
                mixture_idata,
                random_seed=42,
                extend_inferencedata=True,
            )

        return {
            "normal": (normal_prior_idata, normal_idata),
            "preliz": (mu_prior, sigma_prior, preliz_prior_idata, preliz_idata),
            "mixture": mixture_idata,
        }


    def solution_coal_mining_disasters_workflow():
        # Counts per year are non-negative integers with no fixed upper bound,
        # so a Poisson likelihood is a natural starting point.
        #
        # The observed series suggests one unknown rate before a change point
        # and another unknown rate after it.
        with pm.Model():
            early_rate = pm.Exponential("early_rate", lam=1 / 3)
            late_rate = pm.Exponential("late_rate", lam=1)
            change_point = pm.DiscreteUniform(
                "change_point",
                lower=int(coal_years.min()),
                upper=int(coal_years.max()),
            )

            disaster_rate = pm.math.switch(
                coal_years <= change_point,
                early_rate,
                late_rate,
            )
            pm.Poisson("disasters", mu=disaster_rate, observed=coal_disasters)

            prior_idata = pm.sample_prior_predictive(random_seed=42)

            # The discrete change point cannot be sampled by NUTS, so use PyMC's
            # default compound sampler rather than nutpie.
            idata = pm.sample(random_seed=42)
            pm.sample_posterior_predictive(
                idata,
                random_seed=42,
                extend_inferencedata=True,
            )

        return prior_idata, idata

    return (
        solution_coal_mining_disasters_workflow,
        solution_penguin_body_mass_workflow,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    **Part B — Prior and Likelihood Selection:**
    - Match your likelihood to the data-generating process (Normal, Lognormal, Poisson, NegBin, etc.)
    - Use multivariate distributions when observations or parameters are vectors
    - Use Dirichlet priors for probability vectors that must sum to one
    - Use Multivariate Normal models for continuous vectors with covariance structure
    - Use LKJ priors for correlation matrices in multivariate and hierarchical models
    - Always do prior predictive checks
    - Use PreliZ (`pz.maxent`, `pz.mle`) for principled prior elicitation
    - The choice of prior matters most with small data

    ---

    <div style="text-align: center; color: #888; font-size: 0.85rem; padding-top: 1rem;">
    Bayesian Inference with PyMC &mdash; A <a href="https://www.pymc-labs.com" style="color: #154A72;">PyMC Labs</a> Workshop
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Complete model from scratch: dose-response (LD50)

    > ⚠ TO BE AUTHORED.

    Reuse pointer: `Session_1.1` includes an LD50 markdown stub with `pm.math.invlogit(alpha + beta*x)` and `pm.Deterministic("ld50", -alpha/beta)`.

    A dose-response dataset must be added. The current runnable from-scratch example is the Beta-Binomial in `Session_1.1`.
    """)
    return


if __name__ == "__main__":
    app.run()
