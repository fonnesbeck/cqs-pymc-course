import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import inspect
    import numpy as np
    import pymc as pm
    import preliz as pz
    import arviz as az
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import matplotlib.pyplot as plt
    import warnings
    from pathlib import Path

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
    az.style.use("arviz-variat")
    data_path = Path(__file__).parent / "data"
    RANDOM_SEED = 42
    warnings.filterwarnings("ignore", module="mkl_fft")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Session 4.1: Bayesian Linear Regression

    **Prediction, uncertainty, model checking, and decision-making with PyMC**

    ---

    Now that we have covered the basics of Bayesian inference and Markov Chain Monte Carlo (MCMC) methods, we can apply these concepts to a specific class of statistical model: **linear regression**. Regression models are everywhere in data science, and Bayesian linear regression offers a powerful framework for incorporating prior knowledge and uncertainty into the modeling process.

    Bayesian linear regression extends the traditional linear regression framework by incorporating prior beliefs about the parameters and updating these beliefs with data to return a posterior distribution of the model's latent parameters. These posterior distributions can be used to make predictions, estimate uncertainty, and evaluate hypotheses.

    The model assumes that the response variable $y$ is generated from a normal distribution with a mean that is a linear function of the predictors and a constant variance:

    $$
    y = X\beta + \epsilon, \quad \epsilon \sim N(0, \sigma^2I_n)
    $$

    where $y$ is the vector of response variables, $X$ is the design matrix of predictors, $\beta$ is the vector of regression coefficients, and $\epsilon$ is the error term with a normal distribution.

    The posterior distribution, which combines the prior information with the likelihood of the observed data, is derived using Bayes' theorem:

    $$
    p(\beta, \sigma | y, X) \propto p(y | X, \beta, \sigma^2) \; p(\beta) \; p(\sigma)
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fish Weight Prediction — Problem Setup

    Imagine we work for an e-commerce company that sells fresh fish to restaurants. When we ship our products, we need to know the weight of each fish for two reasons:

    1. We **bill** our clients by weight.
    2. Our delivery partner charges **different price tiers** based on weight, and those tiers can get expensive. So we want to know the **probability** of a fish exceeding a given weight threshold.

    We purchase fish in bulk, so we only know the total weight of an order, not individual fish weights. Our supplier has camera-based measurements (length, height, width) for each fish, plus a historical training dataset where fish were actually weighed.

    **Goal:** Build a Bayesian regression model that predicts individual fish weight from physical measurements, with calibrated uncertainty.
    """)
    return


@app.cell(hide_code=True)
def _():
    fish_image_path = data_path.parent.parent / "images" / "weighingfish.jpg"
    fish_image = __import__("base64").b64encode(fish_image_path.read_bytes()).decode()
    mo.md(f'<img src="data:image/jpeg;base64,{fish_image}" alt="Fresh fish being weighed" style="max-width: 100%; height: auto;">')

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exploratory Data Analysis
    """)
    return


@app.cell
def _():
    fish_market = pl.read_csv(data_path / "fish-market.csv")
    _output = fish_market.head()
    _output
    return (fish_market,)


@app.cell(hide_code=True)
def _(fish_market):
    _output = fish_market.describe()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Things to note:

    - There are some **zero-weight** fish: either below the scale minimum or data-entry errors.
    - The standard deviations are large, especially for weight, suggesting substantial variation across species and sizes.

    Let's visualize the relationships between variables:
    """)
    return


@app.cell
def _(fish_market):
    _output = px.scatter_matrix(
        fish_market,
        dimensions=["Length", "Height", "Width", "Weight"],
        color="Species",
        opacity=0.6,
        height=800,
        width=800,
        title="Fish Market — Pairwise Relationships",
    )
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Fish weight and dimensions often follow a **power-law relationship**: weight changes approximately as a dimension raised to a power. Taking logs makes that relationship linear, which is appropriate for linear regression.

    There are also clear species-level differences; any model we build must account for these.
    """)
    return


@app.cell
def _(fish_market):
    fish_log = fish_market.filter(pl.col("Weight") > 0).with_columns(
        [
            pl.col("Width").log().alias("log_width"),
            pl.col("Height").log().alias("log_height"),
            pl.col("Length").log().alias("log_length"),
            pl.col("Weight").log().alias("log_weight"),
        ]
    )
    _output = fish_log.head()
    _output
    return (fish_log,)


@app.cell(hide_code=True)
def _(fish_log):
    _output = px.scatter_matrix(
        fish_log,
        dimensions=["log_length", "log_height", "log_width", "log_weight"],
        color="Species",
        opacity=0.6,
        height=800,
        width=800,
        title="Fish Market — Log-transformed Pairwise Relationships",
    )
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After the log transform, the relationships are much more linear. Each species cluster is roughly elliptical. We can now confidently proceed with linear regression on the log scale.

    ## Train/Test Split

    Before building models, let's split the data so we can evaluate out-of-sample prediction later.
    """)
    return


@app.cell
def _(fish_log):
    fish_test = fish_log.sample(fraction=0.1, seed=1).with_row_index()
    test_idx = fish_test.get_column("index").to_list()
    fish_train = (
        fish_log.with_row_index().filter(~pl.col("index").is_in(test_idx)).drop("index")
    )
    fish_test = fish_test.drop("index")
    mo_fish_msg = (
        f"Training set: {fish_train.height} fish, Test set: {fish_test.height} fish"
    )
    return fish_test, fish_train, mo_fish_msg


@app.cell(hide_code=True)
def _(mo_fish_msg):
    _output = mo.callout(mo.md(mo_fish_msg), kind="info")
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model 1 — Baseline (Intercept Only)

    We start with the simplest possible model: just a global mean with **no predictors**.

    $$
    \log(\text{weight}) \sim \text{Normal}(\mu, \sigma), \quad \mu \sim \text{Normal}(5.5, 2), \quad \sigma \sim \text{HalfNormal}(1)
    $$

    This corresponds to `log(weight) ~ 1` in Wilkinson notation. It will not fit well, but it gives us a baseline for comparison.
    """)
    return


@app.cell
def _(fish_train):
    def build_baseline():
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=5.5, sigma=2.0)
            sigma = pm.HalfNormal("sigma", 1.0)
            pm.Normal(
                "log_obs",
                mu=mu,
                sigma=sigma,
                observed=fish_train["log_weight"].to_numpy(),
            )

            prior_trace = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
            trace = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(trace)
            pm.sample_posterior_predictive(
                trace,
                extend_inferencedata=True,
                random_seed=RANDOM_SEED,
            )
        return model, prior_trace, trace

    baseline_model, baseline_prior, baseline_trace = build_baseline()
    baseline_model
    return baseline_prior, baseline_trace


@app.cell(hide_code=True)
def _(baseline_prior):
    _output = az.plot_dist(
        baseline_prior,
        group="prior_predictive",
        sample_dims=baseline_prior["prior_predictive"]["log_obs"].dims,
    )
    _output
    return


@app.cell(hide_code=True)
def _(baseline_trace):
    _output = az.plot_ppc_dist(baseline_trace)
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The data are clearly **multimodal** (different species have different weight ranges), but our baseline model produces a single unimodal distribution. It compensates by widening `sigma`: poor fit.

    We need predictors!

    ## Model 2 — Unpooled Regression by Species

    Each species gets its own intercept and slopes for width, height, and length:

    $$
    \log(\text{weight}_i) = \mu_{s_i} + \beta_{s_i,0} \cdot \log(\text{width}_i) + \beta_{s_i,1} \cdot \log(\text{height}_i) + \beta_{s_i,2} \cdot \log(\text{length}_i) + \epsilon_i
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Centering predictors

    Before fitting, we **center** the log-transformed predictors by subtracting the training-set mean. This makes the intercept interpretable as the expected log-weight for a fish with average dimensions, and generally improves sampling efficiency.
    """)
    return


@app.cell
def _(fish_train):
    train_mean_log_width = fish_train["log_width"].mean()
    train_mean_log_height = fish_train["log_height"].mean()
    train_mean_log_length = fish_train["log_length"].mean()
    return train_mean_log_height, train_mean_log_length, train_mean_log_width


@app.cell
def _(
    fish_train,
    train_mean_log_height,
    train_mean_log_length,
    train_mean_log_width,
):
    species_names = fish_train["Species"].unique(maintain_order=True).sort().to_list()
    species_to_idx = {species: idx for idx, species in enumerate(species_names)}
    species_idx_train = np.array(
        [species_to_idx[species] for species in fish_train["Species"].to_list()],
        dtype="int64",
    )
    unpooled_coords = {
        "slopes": ["width_effect", "height_effect", "length_effect"],
        "species": species_names,
        "obs_idx": range(fish_train.height),
    }

    def build_unpooled():
        with pm.Model(coords=unpooled_coords) as model:
            lw = pm.Data(
                "log_width",
                (fish_train["log_width"] - train_mean_log_width).to_numpy(),
                dims="obs_idx",
            )
            lh = pm.Data(
                "log_height",
                (fish_train["log_height"] - train_mean_log_height).to_numpy(),
                dims="obs_idx",
            )
            ll = pm.Data(
                "log_length",
                (fish_train["log_length"] - train_mean_log_length).to_numpy(),
                dims="obs_idx",
            )
            lweight = pm.Data(
                "log_weight", fish_train["log_weight"].to_numpy(), dims="obs_idx"
            )
            s = pm.Data("species_idx", species_idx_train, dims="obs_idx")

            mu = pm.Normal("mu", mu=5.5, sigma=2.0, dims="species")
            beta = pm.Normal("beta", sigma=0.5, dims=("slopes", "species"))
            expected = mu[s] + beta[0, s] * lw + beta[1, s] * lh + beta[2, s] * ll
            sigma = pm.HalfNormal("sigma", 1.0)
            pm.Normal(
                "log_obs", mu=expected, sigma=sigma, observed=lweight, dims="obs_idx"
            )

            trace = pm.sample(
                random_seed=RANDOM_SEED,
            )
            pm.compute_log_likelihood(trace)
            pm.sample_posterior_predictive(
                trace,
                extend_inferencedata=True,
                random_seed=RANDOM_SEED,
            )
        return model, trace

    unpooled_model, unpooled_trace = build_unpooled()
    unpooled_model
    return species_to_idx, unpooled_model, unpooled_trace


@app.cell(hide_code=True)
def _(unpooled_trace):
    _output = az.plot_trace_dist(unpooled_trace, var_names=["mu", "beta", "sigma"])
    _output
    return


@app.cell(hide_code=True)
def _(unpooled_trace):
    _output = az.plot_ppc_dist(unpooled_trace)
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Much better! The posterior predictive distribution now captures the multimodal structure of the data. The residual `sigma` is also much smaller.

    ## Model Comparison with LOO

    Let's formally compare the baseline and unpooled models using **Leave-One-Out cross-validation** (LOO-CV), computed via Pareto-Smoothed Importance Sampling (PSIS).
    """)
    return


@app.cell
def _(baseline_trace, unpooled_trace):
    model_comparison = az.compare(
        {"baseline": baseline_trace, "unpooled": unpooled_trace},
    )
    model_comparison
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Refitting the model to new data

    Given the success of the model, you go back and try to fit it to data collected by another vendor, only to find that the predictions aren't nearly as good!

    Frustrated, you go back to the drawing board... they deal with the same type of fish, but what's wrong with their data?

    One of their colleagues mentions something about not having use the same equipment to weight the fish, because the "old manager always tried to cut costs".
    They used a much cheaper scale ...

    Here is the data:
    """)
    return


@app.cell
def _():
    new_fish = pl.read_csv(data_path / "new_fish.csv")
    new_fish.describe()
    return (new_fish,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here is the model, this time fit to the new data.
    """)
    return


@app.cell(hide_code=True)
def _(new_fish):
    new_species_names = new_fish["Species"].unique(maintain_order=True).sort().to_list()
    new_species_to_idx = {species: idx for idx, species in enumerate(new_species_names)}
    new_species_idx = np.array([new_species_to_idx[species] for species in new_fish["Species"].to_list()], dtype="int64")
    new_fish_coords = {"slopes": ["width_effect", "height_effect", "length_effect"], "species": new_species_names, "obs_idx": range(new_fish.height)}
    with pm.Model(coords=new_fish_coords) as fish_unpooled_new:
        log_width = pm.Data("log_width", new_fish["log_width"].to_numpy(), dims="obs_idx")
        log_height = pm.Data("log_height", new_fish["log_height"].to_numpy(), dims="obs_idx")
        log_length = pm.Data("log_length", new_fish["log_length"].to_numpy(), dims="obs_idx")
        log_weight = pm.Data("log_weight", new_fish["log_weight"].to_numpy(), dims="obs_idx")
        species_idx = pm.Data("species_idx", new_species_idx, dims="obs_idx")
        mu = pm.Normal("mu", mu=5.5, sigma=2.0, dims="species")
        beta = pm.Normal("beta", sigma=0.5, dims=("slopes", "species"))
        expected_weight = mu[species_idx] + beta[0, species_idx] * log_width + beta[1, species_idx] * log_height + beta[2, species_idx] * log_length
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("log_obs", mu=expected_weight, sigma=sigma, observed=log_weight, dims="obs_idx")
        new_fish_normal_trace = pm.sample(random_seed=RANDOM_SEED)
        pm.sample_posterior_predictive(new_fish_normal_trace, extend_inferencedata=True, random_seed=RANDOM_SEED)
    fish_unpooled_new
    return new_fish_coords, new_fish_normal_trace, new_species_idx


@app.cell(hide_code=True)
def _(new_fish_normal_trace):
    az.plot_ppc_dist(new_fish_normal_trace, var_names=["log_obs"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Try to diagnose the issue and make the appropriate modifications to the model to accomodate the new data. Since we are trying to make this model more robust, call this new model `fish_unpooled_robust`.
    """)
    return


@app.cell
def _(new_fish, new_fish_coords, new_species_idx):
    def exercise_refit_new_fish():
        with pm.Model(coords=new_fish_coords) as fish_unpooled_robust:
            log_width = pm.Data("log_width", new_fish["log_width"].to_numpy(), dims="obs_idx")
            log_height = pm.Data("log_height", new_fish["log_height"].to_numpy(), dims="obs_idx")
            log_length = pm.Data("log_length", new_fish["log_length"].to_numpy(), dims="obs_idx")
            log_weight = pm.Data("log_weight", new_fish["log_weight"].to_numpy(), dims="obs_idx")
            species_idx = pm.Data("species_idx", new_species_idx, dims="obs_idx")
            mu = pm.Normal("mu", mu=5.5, sigma=2.0, dims="species")
            beta = pm.Normal("beta", sigma=0.5, dims=("slopes", "species"))
            expected_weight = mu[species_idx] + beta[0, species_idx] * log_width + beta[1, species_idx] * log_height + beta[2, species_idx] * log_length
            sigma = pm.HalfNormal("sigma", 1.0)
            # YOUR CODE HERE — define the robust likelihood for log_obs.
            ...
            trace = pm.sample(random_seed=RANDOM_SEED)
            pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=RANDOM_SEED)
        return az.plot_ppc_dist(trace, var_names=["log_obs"])

    return (exercise_refit_new_fish,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Run the new model and plot the posterior predictive checks.
    """)
    return


@app.cell(hide_code=True)
def _():
    run_refit_new_fish = mo.ui.run_button(label="▶ Run exercise")
    run_refit_new_fish
    return (run_refit_new_fish,)


@app.cell(hide_code=True)
def _(exercise_refit_new_fish, run_refit_new_fish):
    mo.stop(not run_refit_new_fish.value, mo.md("*Click ▶ Run exercise once your code is ready.*"))
    exercise_refit_new_fish()
    return


@app.cell(hide_code=True)
def _(new_fish, new_fish_coords, new_species_idx):
    def solution_refit_new_fish():
        with pm.Model(coords=new_fish_coords) as fish_unpooled_robust:
            log_width = pm.Data("log_width", new_fish["log_width"].to_numpy(), dims="obs_idx")
            log_height = pm.Data("log_height", new_fish["log_height"].to_numpy(), dims="obs_idx")
            log_length = pm.Data("log_length", new_fish["log_length"].to_numpy(), dims="obs_idx")
            log_weight = pm.Data("log_weight", new_fish["log_weight"].to_numpy(), dims="obs_idx")
            species_idx = pm.Data("species_idx", new_species_idx, dims="obs_idx")
            mu = pm.Normal("mu", mu=5.5, sigma=2.0, dims="species")
            beta = pm.Normal("beta", sigma=0.5, dims=("slopes", "species"))
            expected_weight = mu[species_idx] + beta[0, species_idx] * log_width + beta[1, species_idx] * log_height + beta[2, species_idx] * log_length
            sigma = pm.HalfNormal("sigma", 1.0)
            pm.StudentT("log_obs", nu=2, mu=expected_weight, sigma=sigma, observed=log_weight, dims="obs_idx")
            trace = pm.sample(random_seed=RANDOM_SEED)
            pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=RANDOM_SEED)
        return az.plot_ppc_dist(trace, var_names=["log_obs"])
    mo.accordion({"Solution": mo.vstack([mo.md(f"```python\n{inspect.getsource(solution_refit_new_fish)}\n```"), mo.lazy(solution_refit_new_fish, show_loading_indicator=True)])})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Out-of-Sample Prediction

    Now let's use the fitted model to predict weights for the **held-out test set**. We update the `Data` containers with test-set values and sample posterior predictions.
    """)
    return


@app.cell
def _(
    fish_test,
    species_to_idx,
    train_mean_log_height,
    train_mean_log_length,
    train_mean_log_width,
    unpooled_model,
    unpooled_trace,
):
    species_idx_test = np.array(
        [species_to_idx[species] for species in fish_test["Species"].to_list()],
        dtype="int64",
    )
    with unpooled_model:
        pm.set_data(
            coords={"obs_idx": range(fish_test.height)},
            new_data={
                "log_width": fish_test["log_width"].to_numpy() - train_mean_log_width,
                "log_height": fish_test["log_height"].to_numpy()
                - train_mean_log_height,
                "log_length": fish_test["log_length"].to_numpy()
                - train_mean_log_length,
                "log_weight": np.zeros(fish_test.height),
                "species_idx": species_idx_test,
            },
        )
        oos_predictions = pm.sample_posterior_predictive(
            unpooled_trace,
            predictions=True,
            extend_inferencedata=True,
            random_seed=RANDOM_SEED,
        )
    oos_predictions
    return


@app.cell(hide_code=True)
def _(fish_test, unpooled_trace):
    plots = az.plot_dist(
        unpooled_trace["predictions"].dataset.map(np.exp),
    )
    for obs_idx, observed_weight in enumerate(fish_test["Weight"].to_numpy()):
        ax = plots.get_target("log_obs", {"obs_idx": obs_idx})
        ax.axvline(observed_weight, color="tab:red", linestyle="--", linewidth=1.5)
    plots
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each dashed line is the matching held-out fish's observed weight. Use the plot to inspect how plausible those values are under the posterior predictions for this split; it is not an estimate of predictive calibration.

    ### Business Insight: Weight-Tier Probabilities

    With full posterior distributions we can compute the probability that any new fish exceeds a weight threshold, critical for estimating shipping costs.
    """)
    return


@app.cell
def _(unpooled_trace):
    oos_pred_weights = np.exp(
        az.extract(unpooled_trace, group="predictions").to_numpy().squeeze()
    )
    return (oos_pred_weights,)


@app.cell(hide_code=True)
def _(oos_pred_weights):
    def make_threshold_figure():
        thresholds = [250, 500, 750, 1000]
        threshold_colors = [PYMC_BLUE, PYMC_LIGHT_BLUE, PYMC_GREEN, PYMC_DARK_GREEN]

        n_fish = oos_pred_weights.shape[0]
        n_cols = 4
        n_rows = (n_fish + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.4))
        axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

        for i in range(n_fish):
            ax = axes_flat[i]
            draws = oos_pred_weights[i]
            x, pdf, _ = az.kde(draws)
            ax.plot(x, pdf, color="k", lw=1.2)
            ax.fill_between(x, pdf, color="k", alpha=0.1)

            pdf_max = float(pdf.max())
            for k, (thr, color) in enumerate(zip(thresholds, threshold_colors)):
                prob = float((draws >= thr).mean())
                ax.axvline(thr, color=color, lw=1.5)
                ax.text(
                    thr,
                    pdf_max * (0.92 - 0.18 * k),
                    f"  >={thr}: {prob * 100:.0f}%",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )

            ax.set_title(f"Test fish {i}", fontsize=10)
            ax.set_yticks([])
            ax.set_xlabel("Weight (g)", fontsize=9)

        for j in range(n_fish, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(
            "Probability of weighing more than thresholds",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig

    make_threshold_figure()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Improve the Fish Model with `preliz`

    The priors in our unpooled model were generic (`Normal(0, 0.5)` for slopes, `Normal(5.5, 2)` for intercepts). Use `preliz` to **elicit** more principled priors from domain knowledge instead of hand-picking hyperparameters.

    **Your task:**

    1. Express what you know as ranges, not distribution parameters: a plausible interval and how much probability mass it should hold.
    2. Use `pz.maxent(dist, lower=..., upper=..., mass=...)` to find the maximum-entropy prior matching each range. Recall the log scale: a 100 g fish is about 4.6, a 2 kg fish about 7.6.
    3. Convert each elicited PreliZ distribution into a PyMC prior with `prior.to_pymc(...)`: no hyperparameters typed by hand.
    4. Re-fit the model with your elicited priors and compare to the original unpooled model using LOO.
    """)
    return


@app.cell
def _(
    fish_train,
    train_mean_log_height,
    train_mean_log_length,
    train_mean_log_width,
    unpooled_trace,
):
    def exercise_preliz_priors():
        # YOUR CODE HERE — elicit intercept, slope, and sigma priors with
        # pz.maxent from the ranges in the Hint (pass plot=False)
        intercept_prior = ...
        slope_prior = ...
        sigma_prior = ...

        species_list = (
            fish_train["Species"].unique(maintain_order=True).sort().to_list()
        )
        species_to_idx = {species: idx for idx, species in enumerate(species_list)}
        species_idx = np.array(
            [species_to_idx[species] for species in fish_train["Species"].to_list()],
            dtype="int64",
        )
        coords = {
            "slopes": ["width_effect", "height_effect", "length_effect"],
            "species": species_list,
            "obs_idx": range(fish_train.height),
        }

        with pm.Model(coords=coords):
            lw = pm.Data(
                "log_width",
                (fish_train["log_width"] - train_mean_log_width).to_numpy(),
                dims="obs_idx",
            )
            lh = pm.Data(
                "log_height",
                (fish_train["log_height"] - train_mean_log_height).to_numpy(),
                dims="obs_idx",
            )
            ll = pm.Data(
                "log_length",
                (fish_train["log_length"] - train_mean_log_length).to_numpy(),
                dims="obs_idx",
            )
            lweight = pm.Data(
                "log_weight", fish_train["log_weight"].to_numpy(), dims="obs_idx"
            )
            s = pm.Data("species_idx", species_idx, dims="obs_idx")

            # YOUR CODE HERE — convert the elicited priors with .to_pymc(...),
            # build the expected log-weight, and add the Normal likelihood
            ...

            improved_trace = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(improved_trace)

        return az.compare(
            {"original_unpooled": unpooled_trace, "improved_priors": improved_trace}
        )

    return (exercise_preliz_priors,)


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Hint": mo.md(r"""
        See Session 1.2's PreliZ section for `pz.maxent` and the `.to_pymc("name")` pattern for dropping an elicited prior into a model. Elicit each prior from a range rather than setting `mu`/`sigma` directly:

        - **Intercept** (average-dimension log-weight): ~5 g to ~1600 g is log-weight ~1.6 to ~7.4, so `pz.maxent(pz.Normal(), lower=1.6, upper=7.4, mass=0.9)` returns a Normal centred near 4.5. Convert with `.to_pymc("mu", dims="species")`.
        - **Slopes** (log-log allometric coefficients, typically ~1-3): `pz.maxent(pz.Normal(), lower=0.0, upper=3.0, mass=0.9)`.
        - **Residual sd** (log scale): `pz.maxent(pz.HalfNormal(), lower=0.0, upper=1.0, mass=0.9)`.

        Pass `plot=False` to keep the elicitation quiet inside the model-building cell.
        """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    run_preliz_priors = mo.ui.run_button(label="▶ Run exercise")
    run_preliz_priors
    return (run_preliz_priors,)


@app.cell(hide_code=True)
def _(exercise_preliz_priors, run_preliz_priors):
    mo.stop(
        not run_preliz_priors.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_preliz_priors()
    return


@app.cell(hide_code=True)
def _(
    fish_train,
    train_mean_log_height,
    train_mean_log_length,
    train_mean_log_width,
    unpooled_trace,
):
    def solution_preliz_priors():
        # Elicit priors from domain knowledge with PreliZ maxent
        # (only ranges are stated; PreliZ computes the hyperparameters)
        intercept_prior = pz.maxent(
            pz.Normal(), lower=1.6, upper=7.4, mass=0.9, plot=False
        )
        slope_prior = pz.maxent(pz.Normal(), lower=0.0, upper=3.0, mass=0.9, plot=False)
        sigma_prior = pz.maxent(
            pz.HalfNormal(), lower=0.0, upper=1.0, mass=0.9, plot=False
        )

        species_list = (
            fish_train["Species"].unique(maintain_order=True).sort().to_list()
        )
        species_to_idx = {species: idx for idx, species in enumerate(species_list)}
        species_idx = np.array(
            [species_to_idx[species] for species in fish_train["Species"].to_list()],
            dtype="int64",
        )
        coords = {
            "slopes": ["width_effect", "height_effect", "length_effect"],
            "species": species_list,
            "obs_idx": range(fish_train.height),
        }

        with pm.Model(coords=coords) as improved_model:
            lw = pm.Data(
                "log_width",
                (fish_train["log_width"] - train_mean_log_width).to_numpy(),
                dims="obs_idx",
            )
            lh = pm.Data(
                "log_height",
                (fish_train["log_height"] - train_mean_log_height).to_numpy(),
                dims="obs_idx",
            )
            ll = pm.Data(
                "log_length",
                (fish_train["log_length"] - train_mean_log_length).to_numpy(),
                dims="obs_idx",
            )
            lweight = pm.Data(
                "log_weight", fish_train["log_weight"].to_numpy(), dims="obs_idx"
            )
            s = pm.Data("species_idx", species_idx, dims="obs_idx")

            mu = intercept_prior.to_pymc("mu", dims="species")
            beta = slope_prior.to_pymc("beta", dims=("slopes", "species"))
            expected = mu[s] + beta[0, s] * lw + beta[1, s] * lh + beta[2, s] * ll
            sigma = sigma_prior.to_pymc("sigma")
            pm.Normal(
                "log_obs", mu=expected, sigma=sigma, observed=lweight, dims="obs_idx"
            )

            improved_trace = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(improved_trace)

        return az.compare(
            {"original_unpooled": unpooled_trace, "improved_priors": improved_trace}
        )

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_preliz_priors)}\n```"
                    ),
                    mo.lazy(solution_preliz_priors, show_loading_indicator=True),
                    mo.md(
                        "The priors are now **elicited** with `pz.maxent` from stated ranges (no distribution hyperparameters are typed by hand) and converted into PyMC variables with `to_pymc`. The intercept range pins the prior on the correct log-weight scale (centred near 4.5 rather than 0). LOO is usually comparable because the data are informative, but the elicited priors are better located and easier to justify."
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

    ## From Probabilities to Decisions

    Posterior predictive distributions tell us what we *believe* about each fish's weight. But every business question requires a single *action*:

    - We must invoice each restaurant with **one number**.
    - We must declare each shipment to the carrier in **one tier**.

    Bayesian **decision analysis** turns belief into action. We pair the posterior predictive with a **loss function** that scores actions against outcomes, then choose the action that minimises *expected* loss:

    $$
    a^* = \arg\min_a \; \mathbb{E}_{y \sim p(y \mid \text{data})}\bigl[\, L(a, y) \,\bigr].
    $$

    Working with the posterior predictive (not the posterior over parameters) is the key: $y$ is what costs us money.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Loss Functions and Bayes Estimators

    The *form* of $L$ determines which posterior summary is optimal:

    | Loss $L(a, y)$ | Optimal action $a^*$ |
    |---|---|
    | $(a - y)^2$ (squared) | Posterior **mean** |
    | $\lvert a - y \rvert$ (absolute) | Posterior **median** |
    | $\mathbf{1}[a \neq y]$ (0/1) | Posterior **mode** |

    #### Asymmetric costs

    For billing, the loss has separate rates for undercharging and overcharging:

    $$
    L(a, y) = c_u \max(y-a, 0) + c_o \max(a-y, 0).
    $$

    Its optimal action is the posterior quantile

    $$
    a^* = F^{-1}\!\left(\frac{c_u}{c_u + c_o}\right),
    $$

    where $F^{-1}$ is the posterior predictive quantile function. When undercharging costs more ($c_u > c_o$), this is a quantile above the posterior median. That is the asymmetric linear (or "pinball") loss we need for billing: undercharging by 50g costs margin, while overcharging by 50g risks the customer relationship.
    """)
    return


@app.cell
def _(oos_pred_weights):
    example_fish = 0
    example_draws = oos_pred_weights[example_fish]

    c_under_demo, c_over_demo = 2.0, 1.0
    asym_q = c_under_demo / (c_under_demo + c_over_demo)

    bayes_estimators = {
        "posterior mean (squared loss)": float(example_draws.mean()),
        "posterior median (absolute loss)": float(np.median(example_draws)),
        f"posterior {asym_q:.2f}-quantile (asym, c_u=2, c_o=1)": float(
            np.quantile(example_draws, asym_q)
        ),
    }
    bayes_estimators
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Decision 1: Optimal Billing Weight

    Each restaurant pays for the weight we declare. Two real-world frictions make the costs asymmetric:

    - **Undercharging** ($c_u$ per gram below the true weight): we forfeit margin.
    - **Overcharging** ($c_o$ per gram above the true weight): the customer disputes the invoice and we risk the relationship.

    For a single fish, the expected loss of billing weight $a$ is

    $$
    \mathbb{E}[L(a)] = c_u \cdot \mathbb{E}[\max(y - a, 0)] + c_o \cdot \mathbb{E}[\max(a - y, 0)],
    $$

    where the expectation is over the posterior predictive distribution of $y$. We approximate it by averaging over posterior predictive draws.

    Below we evaluate this expected loss over a grid of candidate bills under three cost ratios, for a single test fish.
    """)
    return


@app.cell
def _(oos_pred_weights):
    def expected_asym_loss(draws, a, c_u, c_o):
        return (
            c_u * np.maximum(draws - a, 0.0).mean()
            + c_o * np.maximum(a - draws, 0.0).mean()
        )

    bill_grid = np.linspace(
        oos_pred_weights.min() * 0.9,
        oos_pred_weights.max() * 1.1,
        200,
    )
    cost_ratios = [(1.0, 1.0), (2.0, 1.0), (5.0, 1.0)]

    example_draws_d1 = oos_pred_weights[0]
    loss_curves = {
        (c_u, c_o): np.array(
            [expected_asym_loss(example_draws_d1, b, c_u, c_o) for b in bill_grid]
        )
        for c_u, c_o in cost_ratios
    }

    grid_minima = {
        (c_u, c_o): float(bill_grid[loss_curves[(c_u, c_o)].argmin()])
        for c_u, c_o in cost_ratios
    }
    closed_form = {
        (c_u, c_o): float(np.quantile(example_draws_d1, c_u / (c_u + c_o)))
        for c_u, c_o in cost_ratios
    }
    {"grid argmin": grid_minima, "closed-form quantile": closed_form}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The grid search agrees with the closed-form quantile $q = c_u / (c_u + c_o)$ to numerical precision. As undercharging becomes more costly relative to overcharging, the optimal billed weight shifts upward.

    Now apply this to every fish in the test set under one chosen cost ratio, alongside the naive plug-in choice (posterior mean):
    """)
    return


@app.cell
def _(fish_test, oos_pred_weights):
    c_u_chosen, c_o_chosen = 2.0, 1.0
    optimal_q = c_u_chosen / (c_u_chosen + c_o_chosen)

    naive_bills = oos_pred_weights.mean(axis=1)
    optimal_bills_all = np.quantile(oos_pred_weights, optimal_q, axis=1)
    actual_weights = fish_test["Weight"].to_numpy()

    bill_comparison = pl.DataFrame(
        {
            "fish": np.arange(len(actual_weights)),
            "actual_weight": actual_weights.round(1),
            "naive_bill_mean": naive_bills.round(1),
            f"optimal_bill_q{optimal_q:.2f}": optimal_bills_all.round(1),
            "naive_overcharge": (naive_bills - actual_weights).round(1),
            "optimal_overcharge": (optimal_bills_all - actual_weights).round(1),
        }
    )
    bill_comparison
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Because $c_u = 2 c_o$, the optimal rule is the posterior $\tfrac{2}{3}$-quantile. The optimal bills systematically exceed the naive (mean) bills, trading higher overcharge per fish in exchange for avoiding the costlier undercharge.

    ### Decision 2: Shipping-Tier Classification

    The carrier prices by declared weight tier. Five tiers cover the price book:

    | Tier | Range (g) |
    |---|---|
    | 0 | $w < 250$ |
    | 1 | $250 \le w < 500$ |
    | 2 | $500 \le w < 750$ |
    | 3 | $750 \le w < 1000$ |
    | 4 | $w \ge 1000$ |

    Mismatches between the *declared* tier and the *actual* tier carry asymmetric costs:

    - **Under-declaring** triggers a re-weigh fine plus the higher tier charge. Cost rises sharply with the gap.
    - **Over-declaring** locks us into a more expensive tier; margin is lost but no fine is triggered. Cost rises gently with the gap.
    - **Correct declaration** is free.

    We encode this in a cost matrix $C[\text{declared}, \text{actual}]$ and choose the declared tier that minimises expected cost under the posterior tier distribution.
    """)
    return


@app.cell
def _(oos_pred_weights):
    tier_edges = np.array([0, 250, 500, 750, 1000, np.inf])
    n_tiers = len(tier_edges) - 1

    cost_matrix = np.zeros((n_tiers, n_tiers))
    for declared in range(n_tiers):
        for actual in range(n_tiers):
            if declared == actual:
                cost_matrix[declared, actual] = 0.0
            elif declared < actual:
                cost_matrix[declared, actual] = 100.0 + 50.0 * (actual - declared)
            else:
                cost_matrix[declared, actual] = 10.0 * (declared - actual)

    actual_tier_draws = np.digitize(oos_pred_weights, tier_edges[1:-1])
    tier_probs = np.stack(
        [(actual_tier_draws == k).mean(axis=1) for k in range(n_tiers)],
        axis=1,
    )

    expected_cost = tier_probs @ cost_matrix.T
    optimal_tier = expected_cost.argmin(axis=1)

    cost_matrix
    return cost_matrix, n_tiers, optimal_tier, tier_edges, tier_probs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare the decision-theoretic choice against a naive "plug-in" rule that declares the tier containing the posterior mean weight:
    """)
    return


@app.cell
def _(fish_test, oos_pred_weights, optimal_tier, tier_edges):
    posterior_mean_weight = oos_pred_weights.mean(axis=1)
    naive_tier = np.digitize(posterior_mean_weight, tier_edges[1:-1])
    actual_weights_d2 = fish_test["Weight"].to_numpy()
    actual_tier = np.digitize(actual_weights_d2, tier_edges[1:-1])

    decision_table = pl.DataFrame(
        {
            "fish": np.arange(len(actual_weights_d2)),
            "actual_weight": actual_weights_d2.round(1),
            "actual_tier": actual_tier,
            "naive_plug_in_tier": naive_tier,
            "optimal_tier": optimal_tier,
            "differs": naive_tier != optimal_tier,
        }
    )
    decision_table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The naive plug-in rule ignores the *cost* asymmetry. For fish whose posterior straddles a tier boundary, the optimal rule shifts the declaration upward to avoid the steep under-declaration fine. The deeper the posterior probability mass on the higher tier, the more often the optimal and naive choices disagree.

    ## Exercise: Re-tune the Tier Cost Matrix

    Suppose a regulator letter just made under-declaration much harsher: each under-declaration penalty is now **five times** larger than it was before. The over-declaration penalty is unchanged.

    **Your task:**

    1. Build a new cost matrix that scales the under-declaration entries by 5.
    2. Recompute the optimal declared tier for each test fish.
    3. Report how many fish have a different declared tier than under the original matrix, and in which direction.
    """)
    return


@app.function
def exercise_harsh_costs():
    # YOUR CODE HERE — copy cost_matrix and scale the under-declaration
    # entries (declared < actual) by 5
    harsh_cost_matrix = ...

    # YOUR CODE HERE — recompute expected costs from tier_probs and take
    # the argmin to get each fish's new optimal declared tier
    new_optimal_tier = ...

    # YOUR CODE HERE — count switches vs optimal_tier, and upward moves
    n_changed = ...
    n_upward = ...
    return mo.md(
        f"**{n_changed}** fish change declared tier; **{n_upward}** move upward."
    )


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Hint": mo.md(r"""
        Only the off-diagonal entries with `declared < actual` need to change. Multiply that branch by 5, keep `declared > actual` as-is, leave the diagonal at zero. The expected-cost computation `tier_probs @ new_cost_matrix.T` and the argmin step are unchanged. Use `(new_optimal_tier != optimal_tier).sum()` to count switches and `(new_optimal_tier > optimal_tier).sum()` to confirm switches go upward (toward higher declared tiers).
        """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    run_harsh_costs = mo.ui.run_button(label="▶ Run exercise")
    run_harsh_costs
    return (run_harsh_costs,)


@app.cell(hide_code=True)
def _(run_harsh_costs):
    mo.stop(
        not run_harsh_costs.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_harsh_costs()
    return


@app.cell(hide_code=True)
def _(cost_matrix, n_tiers, optimal_tier, tier_probs):
    def solution_harsh_costs():
        harsh_cost_matrix = cost_matrix.copy()
        for declared in range(n_tiers):
            for actual in range(n_tiers):
                if declared < actual:
                    harsh_cost_matrix[declared, actual] *= 5

        new_expected_cost = tier_probs @ harsh_cost_matrix.T
        new_optimal_tier = new_expected_cost.argmin(axis=1)

        n_changed = int((new_optimal_tier != optimal_tier).sum())
        n_upward = int((new_optimal_tier > optimal_tier).sum())
        return mo.md(
            f"**{n_changed}** fish change declared tier; **{n_upward}** move upward."
        )

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_harsh_costs)}\n```"),
                    mo.lazy(solution_harsh_costs, show_loading_indicator=True),
                    mo.md(
                        "The harsher under-declaration penalty pushes the optimal declared tier upward for borderline fish. All shifts are non-decreasing (`n_upward == n_changed`), because raising under-declaration cost can only make a higher tier more attractive, never a lower one."
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

    ## The Bayesian Workflow

    This is the workflow you have practiced throughout the course; here we apply it to regression models with continuous, count, and Binomial outcomes:

    1. **Specify** the model: choose likelihood, priors, and predictors.
    2. **Prior predictive check**: does the model generate plausible data before seeing real data?
    3. **Fit**: sample from the posterior with `pm.sample()`.
    4. **Diagnose**: check convergence (R-hat, ESS, divergences).
    5. **Posterior predictive check**: does the fitted model reproduce the data?
    6. **Compare**: use LOO-CV to compare candidate models.
    7. **Improve**: refine and iterate.
    8. **Decide**: pair the posterior predictive with a loss function and choose the action that minimises expected loss.

    The workflow carries directly into Session 4.2, where the regression structure changes to share information across grouped observations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generalized Regression

    The fish model used a Normal likelihood because log-weight is approximately continuous and symmetric. More generally, a regression model starts with a **linear predictor**,

    $$
    \eta_i = \beta_0 + \sum_j \beta_j x_{ij},
    $$

    then chooses a likelihood for the outcome and maps the real-valued predictor to that likelihood's valid mean with an inverse link:

    $$
    Y_i \sim \mathcal{D}(\mu_i, \theta), \qquad \mu_i = g^{-1}(\eta_i).
    $$

    | Outcome | Likelihood | Inverse link | Extra variation |
    |---|---|---|---|
    | Continuous, symmetric | Normal | identity | residual $\sigma$ |
    | Counts | Poisson | exponential | none |
    | Overdispersed counts | Negative-Binomial | exponential | $\alpha$ |
    | Approvals out of trials | Binomial | logistic | none |
    | Overdispersed approvals | Beta-Binomial | logistic | $\kappa$ |

    The next two examples keep the regression structure but change the likelihood and link to respect the support of the observed outcome.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Count regression: medication, alcohol, and sneezes

    `poisson_sneeze.csv` records a daily count with two binary predictors. Counts cannot be negative, so an unconstrained linear mean is not a valid Poisson rate. A log link makes the rate positive while retaining additive effects on the log-rate scale.
    """)
    return


@app.cell
def _():
    sneezes = pl.read_csv(data_path / "poisson_sneeze.csv")
    sneeze_design = sneezes.select(["meds", "alcohol"]).to_numpy()
    sneeze_coords = {
        "regressor": ["meds", "alcohol"],
        "obs_idx": np.arange(sneezes.height),
    }
    sneezes.head()
    return sneeze_coords, sneeze_design, sneezes


@app.cell(hide_code=True)
def _(sneezes):
    sneeze_counts = (
        sneezes.group_by(["meds", "alcohol", "nsneeze"])
        .len()
        .sort(["alcohol", "meds", "nsneeze"])
    )
    px.bar(
        sneeze_counts,
        x="nsneeze",
        y="len",
        facet_row="alcohol",
        facet_col="meds",
        labels={"nsneeze": "Sneezes per day", "len": "Days"},
        title="Observed sneeze counts by medication and alcohol",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Poisson regression with a log link

    For a count outcome, the Poisson mean $\mu_i$ must be positive. We put priors on an unconstrained intercept and slopes, then transform their linear predictor with $\exp$:

    $$
    \log(\mu_i) = \beta_0 + \beta_{\mathrm{meds}}\,\mathrm{meds}_i + \beta_{\mathrm{alcohol}}\,\mathrm{alcohol}_i.
    $$

    The coefficients are changes in log expected count; exponentiating a coefficient gives a multiplicative change in expected count.
    """)
    return


@app.cell
def _(sneeze_coords, sneeze_design, sneezes):
    def build_poisson_sneeze_model():
        with pm.Model(coords=sneeze_coords) as model:
            design = pm.Data("design", sneeze_design, dims=("obs_idx", "regressor"))
            intercept = pm.Normal(
                "intercept", mu=np.log(sneezes["nsneeze"].mean()), sigma=1.0
            )
            slopes = pm.Normal("slopes", mu=0.0, sigma=1.0, dims="regressor")
            mu = pm.Deterministic(
                "mu",
                pm.math.exp(intercept + pm.math.dot(design, slopes)),
                dims="obs_idx",
            )
            pm.Poisson(
                "sneeze_count",
                mu=mu,
                observed=sneezes["nsneeze"].to_numpy(),
                dims="obs_idx",
            )
            prior = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
            trace = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(trace)
            pm.sample_posterior_predictive(
                trace, extend_inferencedata=True, random_seed=RANDOM_SEED
            )
        return model, prior, trace

    poisson_model, poisson_prior, poisson_trace = build_poisson_sneeze_model()
    poisson_model
    return poisson_prior, poisson_trace


@app.cell(hide_code=True)
def _(poisson_prior):
    az.plot_dist(
        poisson_prior,
        group="prior_predictive",
        sample_dims=poisson_prior["prior_predictive"]["sneeze_count"].dims,
    )
    return


@app.cell(hide_code=True)
def _(poisson_trace):
    az.plot_ppc_dist(poisson_trace, var_names=["sneeze_count"])
    return


@app.cell(hide_code=True)
def _(sneezes):
    sneeze_group_summary = sneezes.group_by(["meds", "alcohol"]).agg(
        pl.col("nsneeze").mean().alias("mean"),
        pl.col("nsneeze").var().alias("variance"),
    )
    sneeze_group_summary
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Overdispersion and Negative-Binomial regression

    A Poisson model requires $\operatorname{Var}(Y_i) = \operatorname{E}(Y_i)$. Compare the group means and variances above. When the observed variation is systematically larger, the Negative-Binomial likelihood keeps the same log-linked mean model and introduces $\alpha$ to allow extra-Poisson variation.
    """)
    return


@app.cell
def _(sneeze_coords, sneeze_design, sneezes):
    def build_negative_binomial_sneeze_model():
        with pm.Model(coords=sneeze_coords) as model:
            design = pm.Data("design", sneeze_design, dims=("obs_idx", "regressor"))
            intercept = pm.Normal(
                "intercept", mu=np.log(sneezes["nsneeze"].mean()), sigma=1.0
            )
            slopes = pm.Normal("slopes", mu=0.0, sigma=1.0, dims="regressor")
            mu = pm.Deterministic(
                "mu",
                pm.math.exp(intercept + pm.math.dot(design, slopes)),
                dims="obs_idx",
            )
            alpha = pm.Exponential("alpha", lam=1.0)
            pm.NegativeBinomial(
                "sneeze_count",
                mu=mu,
                alpha=alpha,
                observed=sneezes["nsneeze"].to_numpy(),
                dims="obs_idx",
            )
            prior = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
            trace = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(trace, compile_kwargs={"mode": "FAST_COMPILE"})
            pm.sample_posterior_predictive(
                trace, extend_inferencedata=True, random_seed=RANDOM_SEED
            )
        return model, prior, trace

    negative_binomial_model, negative_binomial_prior, negative_binomial_trace = (
        build_negative_binomial_sneeze_model()
    )
    negative_binomial_model
    return (negative_binomial_trace,)


@app.cell(hide_code=True)
def _(negative_binomial_trace):
    az.plot_ppc_dist(negative_binomial_trace, var_names=["sneeze_count"])
    return


@app.cell(hide_code=True)
def _(negative_binomial_trace, poisson_trace):
    az.compare(
        {"Poisson": poisson_trace, "Negative-Binomial": negative_binomial_trace}
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Exercise: improve the sneeze model

    The Negative-Binomial model respects the count outcome and handles extra-Poisson variation. Use the observed data, posterior predictive checks, and LOO workflow to propose and fit one defensible improvement.

    State the additional assumption your model makes, then compare its out-of-sample predictive performance with the current model.
    """)
    return


@app.cell(hide_code=True)
def _():
    sneeze_model_hints = mo.accordion(
        {
            "Hints": mo.md("""
            1. Inspect the medication and alcohol groups together. Ask whether one predictor's effect should be allowed to differ across values of the other.
            2. One candidate improvement adds the product of the two binary predictors to the log-rate linear predictor.
            3. Keep the Negative-Binomial likelihood and its dispersion parameter; compare the improved model with the additive model using LOO.
            """)
        }
    )
    sneeze_model_hints

    return


@app.cell(hide_code=True)
def _():
    def exercise_improved_sneeze_model():
        # YOUR CODE HERE — propose and fit a defensible extension of the
        # Negative-Binomial sneeze model, then compare it with the current model.
        improved_model = ...
        ...


    exercise_improved_sneeze_model

    return (exercise_improved_sneeze_model,)


@app.cell(hide_code=True)
def _():
    run_count_interaction = mo.ui.run_button(label="▶ Run exercise")
    run_count_interaction
    return (run_count_interaction,)


@app.cell(hide_code=True)
def _(exercise_improved_sneeze_model, run_count_interaction):
    mo.stop(
        not run_count_interaction.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_improved_sneeze_model()
    return


@app.cell(hide_code=True)
def _(negative_binomial_trace, sneeze_coords, sneeze_design, sneezes):
    def solution_count_interaction():
        with pm.Model(coords=sneeze_coords) as interaction_model:
            design = pm.Data("design", sneeze_design, dims=("obs_idx", "regressor"))
            intercept = pm.Normal(
                "intercept", mu=np.log(sneezes["nsneeze"].mean()), sigma=1.0
            )
            slopes = pm.Normal("slopes", mu=0.0, sigma=1.0, dims="regressor")
            interaction = pm.Normal("interaction", mu=0.0, sigma=1.0)
            mu = pm.Deterministic(
                "mu",
                pm.math.exp(
                    intercept
                    + pm.math.dot(design, slopes)
                    + interaction * design[:, 0] * design[:, 1]
                ),
                dims="obs_idx",
            )
            alpha = pm.Exponential("alpha", lam=1.0)
            pm.NegativeBinomial(
                "sneeze_count",
                mu=mu,
                alpha=alpha,
                observed=sneezes["nsneeze"].to_numpy(),
                dims="obs_idx",
            )
            interaction_trace = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(interaction_trace, compile_kwargs={"mode": "FAST_COMPILE"})
            pm.sample_posterior_predictive(
                interaction_trace,
                extend_inferencedata=True,
                random_seed=RANDOM_SEED,
            )
        return az.compare(
            {"additive": negative_binomial_trace, "interaction": interaction_trace}
        )

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_count_interaction)}\n```"),
                    mo.lazy(solution_count_interaction, show_loading_indicator=True),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    macron_polling_image_path = data_path.parent.parent / "images" / "macron-polling.gif"
    macron_polling_image = __import__("base64").b64encode(
        macron_polling_image_path.read_bytes()
    ).decode()

    mo.md(f"""
    ### French polling data

    French presidents are elected for five-year terms. Between elections, polling firms gauge presidential popularity, which can shape re-election chances. Survey data are not a direct measurement of public support: polling houses differ in sampling methods, response rates can be low, and each survey can be biased.

    <img src="data:image/gif;base64,{macron_polling_image}" alt="Emmanuel Macron at a podium" style="display: block; max-width: 450px; width: 100%; height: auto; margin: 0 auto;">

    The substantive question is close to a hypothetical referendum: if an election or referendum were held today, how much support would Emmanuel Macron receive? The polls are noisy observations of a president's true, latent popularity—not exact answers to that question.

    `macron_popularity.csv` records the number of respondents who approve (`N_approve`) and the number surveyed (`N_total`) in each historical poll. That makes each row a Binomial sampling problem rather than a percentage to regress as a continuous outcome.
    """)

    return


@app.cell
def _():
    polls = pl.read_csv(data_path / "macron_popularity.csv")
    poll_n = polls["N_total"].to_numpy()
    poll_approve = polls["N_approve"].to_numpy()
    poll_log_unemployment = np.log(polls["unemployment"].to_numpy())
    poll_unemployment_mean = poll_log_unemployment.mean()
    poll_unemployment_sd = poll_log_unemployment.std()
    poll_log_unemployment = (
        poll_log_unemployment - poll_unemployment_mean
    ) / poll_unemployment_sd
    polls.head()
    return poll_approve, poll_log_unemployment, poll_n


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Exercise: French polling

    Build a first Bayesian model for the polling data. Identify the observed response, the number of opportunities for that response, and a probability model that respects their support. Then follow the model-checking workflow you used for the sneeze counts.

    State what one shared approval probability assumes about these polls before interpreting the result.
    """)
    return


@app.cell(hide_code=True)
def _():
    polling_hints = mo.accordion(
        {
            "Hints": mo.md("""
            1. Each row records `N_approve` successes out of `N_total` surveyed respondents.
            2. A Binomial likelihood represents this sampling process. Its approval probability must lie between zero and one.
            3. Use a `Beta` prior with `mu=0.4` and `sigma=0.15` for that shared probability.
            4. Reuse the Poisson workflow: prior predictive draws, posterior sampling, explicit log likelihood, then posterior predictive draws.
            """)
        }
    )
    polling_hints

    return


@app.cell
def _():
    logit_grid = np.linspace(-8, 8, 200)
    logit_fig, logit_ax = plt.subplots(figsize=(7, 4))
    logit_ax.plot(logit_grid, 1 / (1 + np.exp(-logit_grid)), color=PYMC_BLUE)
    logit_ax.set(
        xlabel="Linear predictor",
        ylabel="Approval probability",
        title="Logistic inverse link",
    )
    logit_fig

    return


@app.cell(hide_code=True)
def _():
    def exercise_french_poll_model():
        # YOUR CODE HERE — construct a model for the polling response from the
        # observed successes and survey sizes. Run prior and posterior predictive
        # checks, then summarize the fitted shared approval probability.
        polling_model = ...
        ...


    exercise_french_poll_model

    return (exercise_french_poll_model,)


@app.cell(hide_code=True)
def _():
    run_french_poll_model = mo.ui.run_button(label="▶ Run exercise")
    run_french_poll_model

    return (run_french_poll_model,)


@app.cell(hide_code=True)
def _(exercise_french_poll_model, run_french_poll_model):
    mo.stop(
        not run_french_poll_model.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_french_poll_model()

    return


@app.cell(hide_code=True)
def _(poll_approve, poll_n):
    def solution_french_poll_model():
        with pm.Model(coords={"obs_idx": np.arange(len(poll_n))}) as model:
            approval_probability = pm.Beta(
                "approval_probability", mu=0.4, sigma=0.15
            )
            pm.Binomial(
                "approval_count",
                n=poll_n,
                p=approval_probability,
                observed=poll_approve,
                dims="obs_idx",
            )
            prior = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
            trace = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(trace)
            pm.sample_posterior_predictive(
                trace, extend_inferencedata=True, random_seed=RANDOM_SEED
            )
        return model, prior, trace


    def show_french_poll_solution():
        _, _, trace = solution_french_poll_model()
        return az.summary(trace, var_names=["approval_probability"])


    poll_solution = mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_french_poll_model)}\n```"
                    ),
                    mo.lazy(show_french_poll_solution, show_loading_indicator=True),
                ]
            )
        }
    )
    poll_solution

    return (solution_french_poll_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: explain polling variation

    The first polling model treats every survey as evidence about one shared approval probability. Investigate whether standardized log unemployment explains systematic variation among polls.

    1. Choose a probability model and a link that keep predicted approval probabilities valid.
    2. Fit the extension, check its predictions, and compare it with the shared-probability model using LOO.
    3. State the new modeling assumption and whether the comparison supports adding the predictor.
    """)
    return


@app.cell(hide_code=True)
def _():
    unemployment_model_hints = mo.accordion(
        {
            "Hints": mo.md(r"""
            1. The response is still `N_approve` out of `N_total`; retain the Binomial likelihood.
            2. Let a linear predictor combine a baseline with the standardized `poll_log_unemployment` predictor, then map it to $[0, 1]$ with `pm.math.invlogit`.
            3. Use `Normal(-0.7, 0.5)` for the baseline and `Normal(0, 0.2)` for the unemployment effect.
            4. Reuse the French-poll workflow, including explicit log likelihood and posterior predictive draws, then compare it with the intercept-only model using LOO.
            """)
        }
    )
    unemployment_model_hints

    return


@app.cell(hide_code=True)
def _():
    def exercise_unemployment_poll_model():
        # YOUR CODE HERE — extend the polling model with the available predictor.
        # Keep the response on its valid scale, run the model-checking workflow,
        # and compare the extension with the shared-probability model.
        extended_model = ...
        ...


    exercise_unemployment_poll_model

    return (exercise_unemployment_poll_model,)


@app.cell(hide_code=True)
def _():
    run_unemployment_poll_model = mo.ui.run_button(label="▶ Run exercise")
    run_unemployment_poll_model
    return (run_unemployment_poll_model,)


@app.cell(hide_code=True)
def _(exercise_unemployment_poll_model, run_unemployment_poll_model):
    mo.stop(
        not run_unemployment_poll_model.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_unemployment_poll_model()
    return


@app.cell(hide_code=True)
def _(poll_approve, poll_log_unemployment, poll_n, solution_french_poll_model):
    def solution_unemployment_poll_model():
        _, _, raw_poll_trace = solution_french_poll_model()
        with pm.Model(coords={"obs_idx": np.arange(len(poll_n))}) as model:
            unemployment = pm.Data("unemployment", poll_log_unemployment, dims="obs_idx")
            baseline = pm.Normal("baseline", mu=-0.7, sigma=0.5)
            unemployment_effect = pm.Normal("unemployment_effect", mu=0.0, sigma=0.2)
            approval_probability = pm.Deterministic(
                "approval_probability",
                pm.math.invlogit(baseline + unemployment_effect * unemployment),
                dims="obs_idx",
            )
            pm.Binomial(
                "approval_count", n=poll_n, p=approval_probability,
                observed=poll_approve, dims="obs_idx",
            )
            trace = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(trace)
            pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=RANDOM_SEED)
        return az.compare({"intercept_only": raw_poll_trace, "unemployment": trace})

    mo.accordion({
        "Solution": mo.vstack([
            mo.md(f"```python\n{inspect.getsource(solution_unemployment_poll_model)}\n```"),
            mo.lazy(solution_unemployment_poll_model, show_loading_indicator=True),
        ])
    })

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Beta-Binomial model: extra variation in polling

    A Binomial likelihood fixes its variance once the approval probability and respondent count are known. If polls vary more than that sampling model allows, a Beta-Binomial likelihood gives each poll an approval probability drawn from a common Beta distribution. We retain a shared logit-scale mean and add a concentration parameter $\kappa$.
    """)
    return


@app.cell
def _(poll_approve, poll_n):
    def build_beta_binomial_poll_model():
        with pm.Model(coords={"obs_idx": np.arange(len(poll_n))}) as model:
            baseline = pm.Normal("baseline", mu=-0.7, sigma=0.5)
            approval_probability = pm.Deterministic(
                "approval_probability", pm.math.invlogit(baseline)
            )
            kappa = pm.Exponential("kappa_offset", lam=1.0) + 10.0
            pm.BetaBinomial(
                "approval_count",
                alpha=approval_probability * kappa,
                beta=(1.0 - approval_probability) * kappa,
                n=poll_n,
                observed=poll_approve,
                dims="obs_idx",
            )
            prior = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
            trace = pm.sample(random_seed=RANDOM_SEED)
            pm.compute_log_likelihood(trace, compile_kwargs={"mode": "FAST_COMPILE"})
            pm.sample_posterior_predictive(
                trace, extend_inferencedata=True, random_seed=RANDOM_SEED
            )
        return model, prior, trace

    beta_binomial_poll_model, beta_binomial_poll_prior, beta_binomial_poll_trace = build_beta_binomial_poll_model()
    beta_binomial_poll_model
    return (beta_binomial_poll_trace,)


@app.cell(hide_code=True)
def _(beta_binomial_poll_trace):
    az.plot_ppc_dist(beta_binomial_poll_trace, var_names=["approval_count"])
    return


@app.cell(hide_code=True)
def _(beta_binomial_poll_trace, solution_french_poll_model):
    def compare_beta_binomial_poll_model():
        _, _, raw_poll_trace = solution_french_poll_model()
        return az.compare({
            "intercept-only Binomial": raw_poll_trace,
            "Beta-Binomial": beta_binomial_poll_trace,
        })


    mo.lazy(compare_beta_binomial_poll_model, show_loading_indicator=True)

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Regression Recap and Next Steps

    The fish, sneeze, and polling analyses used the same workflow with likelihoods and inverse links that respect each outcome's support. When observed variation is larger than a likelihood permits, model that dispersion rather than forcing a poor fit.

    Session 4.2 keeps the regression workflow but changes the coefficient structure: partial pooling lets related counties share information while retaining group-specific estimates.
    """)
    return


if __name__ == "__main__":
    app.run()
