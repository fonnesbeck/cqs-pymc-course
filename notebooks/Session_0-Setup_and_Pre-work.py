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
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
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
    return PYMC_BLUE, PYMC_GREEN, go, np, px, stats


@app.cell(hide_code=True)
def header(mo):
    import base64
    from pathlib import Path

    logo_path = Path(__file__).parent / "images" / "pymc-labs-logo.png"
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="300" style="margin-bottom: 0.5rem;">'
    else:
        logo_html = ""

    mo.md(f"""
    {logo_html}

    # Workshop Setup & Pre-work

    Welcome to the **London Bayesian Inference Workshop**. This self-paced notebook has two goals:

    1. **Verify your environment**: confirm that all required packages are installed and working.
    2. **Refresh prerequisite skills**: NumPy, Polars, plotting, probability, and simulation.

    Complete this notebook before Day 1. If you run into issues, reach out to the instructor.

    ---

    ## How This Notebook Works

    This is a [**marimo**](https://marimo.io) notebook. Unlike Jupyter, marimo notebooks are **reactive**: when you change a cell, all cells that depend on it automatically re-run. Cells execute in dependency order, not top-to-bottom order.

    A few things to know:

    - **Run a cell** by pressing `Ctrl+Enter` (or `Cmd+Enter` on Mac).
    - **Interactive elements** use `mo.ui`: sliders, dropdowns, buttons, and so on.
    - **Variables flow between cells**: if cell A defines `x`, cell B can use `x` as a parameter.
    - **Each variable has one home**: a variable can only be defined in one cell.

    Try moving the slider or changing the dropdown below and watch the histogram redraw.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The widgets below are wired to a Plotly histogram. Move the slider to change the sample size, or pick a different distribution from the dropdown. The chart updates immediately.
    """)
    return


@app.cell
def _(mo):
    intro_n = mo.ui.slider(50, 5000, value=500, step=50, label="Sample size")
    intro_dist = mo.ui.dropdown(
        ["Normal", "Exponential", "Uniform"],
        value="Normal",
        label="Distribution",
    )
    mo.hstack([intro_n, intro_dist], justify="start")
    return intro_dist, intro_n


@app.cell
def _(intro_dist, intro_n, np):
    intro_rng = np.random.default_rng(seed=0)
    if intro_dist.value == "Normal":
        intro_samples = intro_rng.normal(0, 1, size=intro_n.value)
    elif intro_dist.value == "Exponential":
        intro_samples = intro_rng.exponential(1.0, size=intro_n.value)
    else:
        intro_samples = intro_rng.uniform(-2, 2, size=intro_n.value)
    return (intro_samples,)


@app.cell(hide_code=True)
def _(PYMC_BLUE, intro_dist, intro_n, intro_samples, px):
    intro_fig = px.histogram(
        x=intro_samples,
        nbins=40,
        title=f"{intro_dist.value} samples (n = {intro_n.value:,})",
        labels={"x": "value"},
        opacity=0.8,
        color_discrete_sequence=[PYMC_BLUE],
    )
    intro_fig.update_layout(width=700, height=350, showlegend=False)
    intro_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three cells just talked to each other reactively: a UI cell created the widgets, a compute cell read `intro_n.value` and `intro_dist.value` to draw samples, and a render cell drew the chart. You'll see this same split throughout the notebook: cells that do real work stay visible, and the marimo plumbing around them is hidden.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Environment Check

    The cell below imports all the packages we'll use throughout the workshop. If everything is installed correctly, you'll see a green checkmark. If something fails, follow the error message to fix your environment.
    """)
    return


@app.cell(hide_code=True)
def environment_check(mo):
    import importlib

    required = {
        "numpy": "NumPy (numerical computing)",
        "scipy": "SciPy (scientific computing)",
        "polars": "Polars (DataFrames)",
        "matplotlib": "Matplotlib (plotting)",
        "plotly": "Plotly (interactive plotting)",
        "pymc": "PyMC (Bayesian modeling)",
        "pytensor": "PyTensor (computational backend)",
        "arviz": "ArviZ (diagnostics and plots)",
        "nutpie": "nutpie (fast MCMC sampler)",
        "preliz": "PreliZ (prior elicitation)",
        "pymc_extras": "PyMC Extras (extensions)",
    }
    rows = []
    all_ok = True
    for pkg, description in required.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            rows.append(f"| {description} | `{pkg}` | {version} | ::lucide:check:: |")
        except ImportError:
            rows.append(f"| {description} | `{pkg}` | . | ::lucide:x:: **MISSING** |")
            all_ok = False
    env_table = "\n".join(rows)

    if all_ok:
        env_status = mo.callout(
            mo.md("All packages installed successfully."), kind="success"
        )
    else:
        env_status = mo.callout(
            mo.md(
                "Some packages are missing. Run `pixi install` in your terminal and restart."
            ),
            kind="danger",
        )

    mo.vstack(
        [
            mo.md(f"""
    | Package | Import | Version | Status |
    |---------|--------|---------|--------|
    {env_table}
    """),
            env_status,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's also verify that PyMC can sample a trivial model. First, define and fit the model:
    """)
    return


@app.cell
def pymc_smoke_test(np):
    import pymc as pm

    with pm.Model():
        mu = pm.Normal("mu", mu=0, sigma=1)
        pm.Normal("obs", mu=mu, sigma=1, observed=np.array([0.5, 1.0, -0.3]))
        smoke_trace = pm.sample(
            200, tune=200, chains=2, progressbar=False, random_seed=42
        )
    return (smoke_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then summarize the posterior with ArviZ:
    """)
    return


@app.cell
def arviz_smoke_test(smoke_trace):
    import arviz as az

    az.summary(smoke_trace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. NumPy Refresher

    NumPy is the foundation for numerical computing in Python. We'll use it constantly for arrays, random number generation, and mathematical operations.
    """)
    return


@app.cell
def numpy_basics(np):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.linspace(0, 1, 5)
    c = a * 2 + 1
    d = np.exp(b)
    return a, b, c, d


@app.cell(hide_code=True)
def numpy_basics_render(a, b, c, d, mo, np):
    mo.md(f"""
    - `a` → `{a}`
    - `b` → `{np.array2string(b, precision=2)}`
    - `a * 2 + 1` → `{c}`  (vectorized, no loop needed)
    - `np.exp(b)` → `{np.array2string(d, precision=3)}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Broadcasting** lets you operate on arrays of different shapes. Below, a `(3, 1)` column of row means is subtracted from a `(3, 4)` matrix without writing a loop:
    """)
    return


@app.cell
def broadcasting_demo(np):
    matrix = np.arange(12).reshape(3, 4)
    row_means = matrix.mean(axis=1, keepdims=True)
    centered = matrix - row_means
    return centered, matrix, row_means


@app.cell(hide_code=True)
def broadcasting_render(centered, matrix, mo, row_means):
    mo.vstack(
        [
            mo.md("`matrix` (3×4):"),
            mo.plain_text(str(matrix)),
            mo.md("`row_means` (3×1):"),
            mo.plain_text(str(row_means)),
            mo.md("`matrix - row_means` (each row centered):"),
            mo.plain_text(str(centered)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For random number generation, always use `default_rng` with a seed for reproducibility:
    """)
    return


@app.cell
def random_generation(np):
    rng = np.random.default_rng(seed=42)
    uniform_samples = rng.uniform(0, 1, size=5)
    normal_samples = rng.normal(loc=0, scale=1, size=5)
    return normal_samples, rng, uniform_samples


@app.cell(hide_code=True)
def random_generation_render(mo, normal_samples, np, uniform_samples):
    mo.md(f"""
    - `rng.uniform(0, 1, size=5)` → `{np.array2string(uniform_samples, precision=3)}`
    - `rng.normal(0, 1, size=5)` → `{np.array2string(normal_samples, precision=3)}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Polars Refresher

    [Polars](https://pola.rs) is a fast DataFrame library. We'll use it for data manipulation throughout the workshop.
    """)
    return


@app.cell
def polars_basics():
    import polars as pl

    demo_df = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "group": ["A", "B", "A", "B", "A"],
            "score": [85, 92, 78, 88, 95],
            "hours": [10, 15, 8, 12, 20],
        }
    )
    demo_df
    return demo_df, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Filtering** with `df.filter(...)`:
    """)
    return


@app.cell
def polars_filter(demo_df, pl):
    high_scorers = demo_df.filter(pl.col("score") > 85).select("name", "score")
    high_scorers
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Group-by aggregation** with `df.group_by(...).agg(...)`:
    """)
    return


@app.cell
def polars_groupby(demo_df, pl):
    group_stats = demo_df.group_by("group").agg(
        pl.col("score").mean().alias("mean_score"),
        pl.col("hours").sum().alias("total_hours"),
        pl.len().alias("count"),
    )
    group_stats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Adding computed columns** with `df.with_columns(...)`:
    """)
    return


@app.cell
def polars_with_columns(demo_df, pl):
    df_extended = demo_df.with_columns(
        (pl.col("score") / pl.col("hours")).round(1).alias("efficiency"),
        pl.col("score").rank().alias("rank"),
    )
    df_extended
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Plotting

    We'll primarily use **Plotly** for interactive plots and **Matplotlib** for static plots (especially with ArviZ).
    """)
    return


@app.cell
def plotting_plotly(PYMC_BLUE, go, np, px, rng):
    scatter_x = np.linspace(0, 4 * np.pi, 200)
    scatter_y = np.sin(scatter_x) + rng.normal(0, 0.2, size=len(scatter_x))

    scatter_fig = px.scatter(
        x=scatter_x,
        y=scatter_y,
        labels={"x": "x", "y": "sin(x) + noise"},
        title="Plotly: Interactive Scatter Plot",
        opacity=0.6,
    )
    scatter_fig.add_trace(
        go.Scatter(
            x=scatter_x,
            y=np.sin(scatter_x),
            mode="lines",
            name="sin(x)",
            line=dict(color=PYMC_BLUE, width=2),
        )
    )
    scatter_fig.update_layout(width=700, height=400)
    scatter_fig
    return


@app.cell
def plotting_matplotlib(PYMC_BLUE, PYMC_GREEN, np, rng, stats):
    import matplotlib.pyplot as plt

    hist_samples = rng.normal(loc=5, scale=2, size=1000)

    hist_fig, hist_ax = plt.subplots(figsize=(7, 3.5))
    hist_ax.hist(
        hist_samples,
        bins=40,
        density=True,
        alpha=0.7,
        color=PYMC_BLUE,
        edgecolor="white",
    )
    hist_x = np.linspace(-2, 12, 200)
    hist_ax.plot(
        hist_x,
        stats.norm.pdf(hist_x, 5, 2),
        color=PYMC_GREEN,
        lw=2,
        label="N(5, 2)",
    )
    hist_ax.set_xlabel("Value")
    hist_ax.set_ylabel("Density")
    hist_ax.set_title("Matplotlib: Histogram with Theoretical PDF")
    hist_ax.legend()
    plt.tight_layout()
    hist_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Probability Foundations

    Bayesian inference is built on probability. Let's review the key concepts.

    ### Distributions: Discrete and Continuous

    A **distribution** describes the probability of different values a random variable can take. **Discrete** distributions (e.g. Binomial, Poisson) are described by a probability mass function (PMF); **continuous** distributions (e.g. Normal, Exponential) by a probability density function (PDF).

    Let's visualize a few common ones using `scipy.stats`.
    """)
    return


@app.cell
def discrete_distributions(go, np, stats):
    discrete_x = np.arange(0, 21)

    discrete_fig = go.Figure()
    discrete_fig.add_trace(
        go.Bar(
            x=discrete_x,
            y=stats.binom.pmf(discrete_x, n=20, p=0.3),
            name="Binomial(n=20, p=0.3)",
            opacity=0.7,
            width=0.4,
            offset=-0.2,
        )
    )
    discrete_fig.add_trace(
        go.Bar(
            x=discrete_x,
            y=stats.poisson.pmf(discrete_x, mu=6),
            name="Poisson(lambda=6)",
            opacity=0.7,
            width=0.4,
            offset=0.2,
        )
    )
    discrete_fig.update_layout(
        title="Discrete Distributions: PMFs",
        xaxis_title="x",
        yaxis_title="P(X = x)",
        width=700,
        height=400,
        barmode="overlay",
    )
    discrete_fig
    return


@app.cell
def continuous_distributions(go, np, stats):
    continuous_x = np.linspace(-4, 8, 500)

    continuous_fig = go.Figure()
    continuous_fig.add_trace(
        go.Scatter(
            x=continuous_x,
            y=stats.norm.pdf(continuous_x, 2, 1),
            mode="lines",
            name="Normal(mu=2, sigma=1)",
            line=dict(width=2),
        )
    )
    continuous_fig.add_trace(
        go.Scatter(
            x=continuous_x,
            y=stats.norm.pdf(continuous_x, 2, 2),
            mode="lines",
            name="Normal(mu=2, sigma=2)",
            line=dict(width=2, dash="dash"),
        )
    )
    continuous_fig.add_trace(
        go.Scatter(
            x=continuous_x[continuous_x >= 0],
            y=stats.expon.pdf(continuous_x[continuous_x >= 0], scale=2),
            mode="lines",
            name="Exponential(lambda=0.5)",
            line=dict(width=2),
        )
    )
    continuous_fig.update_layout(
        title="Continuous Distributions: PDFs",
        xaxis_title="x",
        yaxis_title="f(x)",
        width=700,
        height=400,
    )
    continuous_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With enough samples, empirical statistics converge to their theoretical values. Below we draw 10,000 samples from three distributions and compare sample moments to their theoretical counterparts:
    """)
    return


@app.cell
def sample_vs_theoretical(np, pl, stats):
    sample_rng = np.random.default_rng(seed=7)
    n = 10_000
    distributions = {
        "Normal(2, 1)": (sample_rng.normal(2, 1, size=n), stats.norm(2, 1)),
        "Exponential(lambda=0.5)": (
            sample_rng.exponential(2, size=n),
            stats.expon(scale=2),
        ),
        "Poisson(mu=6)": (
            sample_rng.poisson(6, size=n).astype(float),
            stats.poisson(6),
        ),
    }
    stats_comparison = pl.DataFrame(
        [
            {
                "Distribution": name,
                "Sample Mean": round(samples.mean(), 3),
                "Theoretical Mean": round(dist.mean(), 3),
                "Sample Std": round(samples.std(), 3),
                "Theoretical Std": round(dist.std(), 3),
            }
            for name, (samples, dist) in distributions.items()
        ]
    )
    stats_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Simulation and Monte Carlo

    Simulation is a powerful tool for building intuition about probability. The idea: instead of deriving answers analytically, **generate random samples** and compute statistics from them.

    ### The Law of Large Numbers

    As sample size grows, the sample mean converges to the true expected value. Below: a die roll (expected value 3.5):
    """)
    return


@app.cell
def law_of_large_numbers(PYMC_BLUE, go, np, rng):
    lln_n_rolls = 5000
    lln_rolls = rng.integers(1, 7, size=lln_n_rolls)
    lln_cumulative_mean = np.cumsum(lln_rolls) / np.arange(1, lln_n_rolls + 1)

    lln_fig = go.Figure()
    lln_fig.add_trace(
        go.Scatter(
            x=np.arange(1, lln_n_rolls + 1),
            y=lln_cumulative_mean,
            mode="lines",
            name="Running mean",
            line=dict(color=PYMC_BLUE),
        )
    )
    lln_fig.add_hline(
        y=3.5,
        line=dict(color="#81C240", dash="dash"),
        annotation_text="E[X] = 3.5",
    )
    lln_fig.update_layout(
        title="Law of Large Numbers: Die Rolls",
        xaxis_title="Number of rolls",
        yaxis_title="Running average",
        width=700,
        height=400,
    )
    lln_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Central Limit Theorem

    The sum (or mean) of many independent random variables tends toward a normal distribution, regardless of the original distribution. This is why the normal distribution is so ubiquitous.
    """)
    return


@app.cell
def central_limit_theorem(np, px, rng, stats):
    clt_n_per_mean = 30
    clt_n_means = 2000

    clt_exp_samples = rng.exponential(scale=2.0, size=(clt_n_means, clt_n_per_mean))
    clt_sample_means = clt_exp_samples.mean(axis=1)

    clt_fig = px.histogram(
        x=clt_sample_means,
        nbins=50,
        histnorm="probability density",
        title=f"CLT: {clt_n_means} sample means (n={clt_n_per_mean} each, from Exponential)",
        labels={"x": "Sample Mean", "y": "Density"},
        opacity=0.7,
    )
    clt_x_range = np.linspace(clt_sample_means.min(), clt_sample_means.max(), 200)
    clt_theoretical_mean = 2.0
    clt_theoretical_se = 2.0 / np.sqrt(clt_n_per_mean)
    clt_fig.add_scatter(
        x=clt_x_range,
        y=stats.norm.pdf(clt_x_range, clt_theoretical_mean, clt_theoretical_se),
        mode="lines",
        name=f"N({clt_theoretical_mean}, {clt_theoretical_se:.2f})",
        line=dict(color="red", width=2),
    )
    clt_fig.update_layout(width=700, height=400)
    clt_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Monte Carlo Estimation

    **Monte Carlo methods** use random sampling to estimate quantities that might be hard to compute analytically.

    **Example:** Estimate $\pi$ by randomly throwing darts at a unit square and counting how many land inside the inscribed quarter-circle.

    $$\frac{\text{area of quarter-circle}}{\text{area of square}} = \frac{\pi/4}{1} = \frac{\pi}{4}$$

    So $\pi \approx 4 \times \frac{\text{darts inside circle}}{\text{total darts}}$.
    """)
    return


@app.cell
def _(mo):
    n_darts_slider = mo.ui.slider(
        100,
        50000,
        value=5000,
        step=100,
        label="Number of darts",
    )
    n_darts_slider
    return (n_darts_slider,)


@app.cell
def monte_carlo_pi(PYMC_BLUE, go, n_darts_slider, np):
    mc_rng = np.random.default_rng(seed=123)
    mc_n = n_darts_slider.value
    mc_x = mc_rng.uniform(0, 1, size=mc_n)
    mc_y = mc_rng.uniform(0, 1, size=mc_n)
    mc_inside = mc_x**2 + mc_y**2 <= 1.0
    pi_estimate = 4.0 * mc_inside.sum() / mc_n

    mc_max_plot = min(mc_n, 5000)
    mc_fig = go.Figure()
    mc_fig.add_trace(
        go.Scatter(
            x=mc_x[:mc_max_plot][mc_inside[:mc_max_plot]],
            y=mc_y[:mc_max_plot][mc_inside[:mc_max_plot]],
            mode="markers",
            marker=dict(size=2, color=PYMC_BLUE),
            name="Inside",
        )
    )
    mc_fig.add_trace(
        go.Scatter(
            x=mc_x[:mc_max_plot][~mc_inside[:mc_max_plot]],
            y=mc_y[:mc_max_plot][~mc_inside[:mc_max_plot]],
            mode="markers",
            marker=dict(size=2, color="salmon"),
            name="Outside",
        )
    )
    mc_arc_theta = np.linspace(0, np.pi / 2, 100)
    mc_fig.add_trace(
        go.Scatter(
            x=np.cos(mc_arc_theta),
            y=np.sin(mc_arc_theta),
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )
    mc_fig.update_layout(
        title=f"Monte Carlo estimate of pi: {pi_estimate:.4f} (true: {np.pi:.4f})",
        xaxis=dict(scaleanchor="y", range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=500,
        height=500,
    )
    mc_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. Self-Check Exercises

    Complete these exercises to confirm your environment works and you're comfortable with the prerequisites. Each exercise has a hidden solution you can reveal after trying.

    ### Exercise 1: Simulate Coin Flips

    Simulate 10,000 fair coin flips using NumPy. Compute the proportion of heads. It should be close to 0.5.

    *Hint:* Use `rng.binomial(n=1, p=0.5, size=...)` or `rng.choice([0, 1], size=...)`.
    """)
    return


@app.cell
def exercise_1(np):
    ex1_rng = np.random.default_rng(seed=0)
    flips = ...
    proportion_heads = ...
    return


@app.cell
def _(mo):
    show_solution_1 = mo.ui.run_button(label="Show Solution")
    show_solution_1
    return (show_solution_1,)


@app.cell
def solution_1(mo, np, show_solution_1):
    mo.stop(not show_solution_1.value)

    sol1_rng = np.random.default_rng(seed=0)
    sol1_flips = sol1_rng.binomial(n=1, p=0.5, size=10_000)
    sol1_proportion = sol1_flips.mean()
    return (sol1_proportion,)


@app.cell(hide_code=True)
def solution_1_render(mo, show_solution_1, sol1_proportion):
    mo.stop(not show_solution_1.value)
    mo.md(
        f"Proportion of heads: **{sol1_proportion:.4f}**. "
        f"With 10,000 flips, we expect this to be close to 0.5 by the Law of Large Numbers."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Estimate a Probability via Monte Carlo

    A factory produces bolts with diameters that follow a Normal distribution with mean 10mm and standard deviation 0.3mm. Bolts are rejected if their diameter is outside the range [9.5, 10.5]mm.

    **Use simulation to estimate the rejection rate.**

    1. Generate 100,000 bolt diameters from $N(10, 0.3^2)$.
    2. Count how many fall outside [9.5, 10.5].
    3. Compare your Monte Carlo estimate to the exact answer from `scipy.stats`.
    """)
    return


@app.cell
def exercise_2(np):
    ex2_rng = np.random.default_rng(seed=1)
    diameters = ...
    rejected = ...
    mc_rejection_rate = ...
    return


@app.cell
def _(mo):
    show_solution_2 = mo.ui.run_button(label="Show Solution")
    show_solution_2
    return (show_solution_2,)


@app.cell
def solution_2(mo, np, show_solution_2, stats):
    mo.stop(not show_solution_2.value)

    sol2_rng = np.random.default_rng(seed=1)
    sol2_diameters = sol2_rng.normal(10, 0.3, size=100_000)
    sol2_rejected = (sol2_diameters < 9.5) | (sol2_diameters > 10.5)
    sol2_mc_rate = sol2_rejected.mean()
    sol2_exact_rate = 1 - (stats.norm.cdf(10.5, 10, 0.3) - stats.norm.cdf(9.5, 10, 0.3))
    return sol2_exact_rate, sol2_mc_rate


@app.cell(hide_code=True)
def solution_2_render(mo, show_solution_2, sol2_exact_rate, sol2_mc_rate):
    mo.stop(not show_solution_2.value)
    mo.md(f"""
    - Monte Carlo rejection rate: **{sol2_mc_rate:.5f}**
    - Exact rejection rate: **{sol2_exact_rate:.5f}**

    The estimates are very close: this is Monte Carlo in action.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Explore a Distribution Interactively

    Use the sliders below to explore the Beta distribution: an important distribution in Bayesian inference for modeling probabilities.

    The Beta distribution has two parameters, $\alpha$ and $\beta$:

    - $\alpha = \beta = 1$: uniform on [0, 1]
    - $\alpha > \beta$: skewed toward 1
    - $\alpha < \beta$: skewed toward 0
    - Large $\alpha$ and $\beta$: concentrated around $\frac{\alpha}{\alpha + \beta}$
    """)
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(0.1, 20, value=2, step=0.1, label="alpha")
    beta_slider = mo.ui.slider(0.1, 20, value=5, step=0.1, label="beta")
    mo.hstack([alpha_slider, beta_slider], gap=2)
    return alpha_slider, beta_slider


@app.cell
def beta_explorer(
    PYMC_BLUE,
    PYMC_GREEN,
    alpha_slider,
    beta_slider,
    go,
    np,
    stats,
):
    beta_alpha = alpha_slider.value
    beta_beta = beta_slider.value

    beta_x = np.linspace(0, 1, 300)
    beta_pdf = stats.beta.pdf(beta_x, beta_alpha, beta_beta)
    beta_mean = beta_alpha / (beta_alpha + beta_beta)

    beta_fig = go.Figure()
    beta_fig.add_trace(
        go.Scatter(
            x=beta_x,
            y=beta_pdf,
            mode="lines",
            fill="tozeroy",
            line=dict(color=PYMC_BLUE, width=2),
        )
    )
    beta_fig.add_vline(
        x=beta_mean,
        line=dict(color=PYMC_GREEN, dash="dash"),
        annotation_text=f"mean={beta_mean:.3f}",
    )
    beta_fig.update_layout(
        title=f"Beta(alpha={beta_alpha:.1f}, beta={beta_beta:.1f})",
        xaxis_title="x",
        yaxis_title="Density",
        width=700,
        height=400,
    )
    return beta_alpha, beta_beta, beta_fig, beta_mean


@app.cell(hide_code=True)
def beta_explorer_render(beta_alpha, beta_beta, beta_fig, beta_mean, mo):
    beta_variance = (beta_alpha * beta_beta) / (
        (beta_alpha + beta_beta) ** 2 * (beta_alpha + beta_beta + 1)
    )
    beta_mode = (
        max(0.0, (beta_alpha - 1) / (beta_alpha + beta_beta - 2))
        if beta_alpha > 1 and beta_beta > 1
        else float("nan")
    )
    mo.vstack(
        [
            beta_fig,
            mo.md(
                f"**Mean:** {beta_mean:.4f} &nbsp; | &nbsp; "
                f"**Variance:** {beta_variance:.4f} &nbsp; | &nbsp; "
                f"**Mode:** {beta_mode:.4f} (when alpha, beta > 1)"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4: Bayesian Updating by Simulation

    You're testing whether a coin is fair. Your prior belief is that $\theta$ (the probability of heads) follows a $\text{Beta}(2, 2)$ distribution: slightly concentrated around 0.5 but open to other values.

    You flip the coin 20 times and observe 14 heads.

    **Tasks:**

    1. Draw 100,000 samples from your prior: $\text{Beta}(2, 2)$
    2. For each sampled $\theta$, simulate 20 coin flips
    3. Keep only the $\theta$ values where the simulation produced exactly 14 heads (**rejection sampling**)
    4. Plot a histogram of the accepted $\theta$ values: this approximates the posterior.
    5. Compare to the exact posterior: $\text{Beta}(2 + 14, 2 + 6) = \text{Beta}(16, 8)$
    """)
    return


@app.cell
def exercise_4(np):
    ex4_rng = np.random.default_rng(seed=42)
    prior_thetas = ...
    return


@app.cell
def _(mo):
    show_solution_4 = mo.ui.run_button(label="Show Solution")
    show_solution_4
    return (show_solution_4,)


@app.cell
def solution_4(mo, np, show_solution_4):
    mo.stop(not show_solution_4.value)

    sol4_rng = np.random.default_rng(seed=42)
    sol4_prior = sol4_rng.beta(2, 2, size=100_000)
    sol4_simulated_heads = sol4_rng.binomial(n=20, p=sol4_prior)
    sol4_accepted = sol4_prior[sol4_simulated_heads == 14]
    return (sol4_accepted,)


@app.cell(hide_code=True)
def solution_4_render(go, mo, np, show_solution_4, sol4_accepted, stats):
    mo.stop(not show_solution_4.value)

    sol4_x = np.linspace(0, 1, 300)
    sol4_exact_posterior = stats.beta.pdf(sol4_x, 16, 8)

    sol4_fig = go.Figure()
    sol4_fig.add_trace(
        go.Histogram(
            x=sol4_accepted,
            nbinsx=50,
            histnorm="probability density",
            name=f"Rejection sampling (n={len(sol4_accepted)})",
            opacity=0.7,
        )
    )
    sol4_fig.add_trace(
        go.Scatter(
            x=sol4_x,
            y=sol4_exact_posterior,
            mode="lines",
            name="Exact: Beta(16, 8)",
            line=dict(color="red", width=2),
        )
    )
    sol4_fig.update_layout(
        title="Approximate vs Exact Posterior",
        xaxis_title="theta (probability of heads)",
        yaxis_title="Density",
        width=700,
        height=400,
    )

    mo.vstack(
        [
            sol4_fig,
            mo.md(
                f"Accepted {len(sol4_accepted)} samples out of 100,000. "
                f"Posterior mean: **{sol4_accepted.mean():.3f}** "
                f"(exact: {16 / 24:.3f})"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## You're Ready!

    If you've made it through this notebook and everything ran without errors, you're all set for Day 1. Here's what we covered:

    - **Environment**: All packages installed and PyMC can sample
    - **NumPy**: Arrays, broadcasting, random generation
    - **Polars**: DataFrames, filtering, grouping, computed columns
    - **Plotting**: Plotly for interactive plots, Matplotlib for static plots
    - **Probability**: Distributions, PMFs/PDFs, sample statistics vs. theoretical values
    - **Simulation**: Law of large numbers, central limit theorem, Monte Carlo estimation, rejection sampling

    See you at the workshop.

    ---

    <div style="text-align: center; color: #888; font-size: 0.85rem; padding-top: 1rem;">
    Bayesian Inference with PyMC. A <a href="https://www.pymc-labs.com" style="color: #154A72;">PyMC Labs</a> Workshop.
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
