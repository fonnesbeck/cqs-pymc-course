import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


with app.setup:
    import marimo as mo
    import inspect
    import base64
    from pathlib import Path
    import numpy as np
    import pymc as pm
    import pymc_extras as pmx
    import arviz as az
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import matplotlib.pyplot as plt
    import warnings
    import xarray as xr
    import pymc.dims as pmd

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
    RANDOM_SEED = 20090425
    RNG = np.random.default_rng(RANDOM_SEED)
    warnings.filterwarnings("ignore", module="mkl_fft")
    warnings.filterwarnings("ignore", category=RuntimeWarning)


@app.cell(hide_code=True)
def header():
    mo.md("""

    # Session 4.2: Hierarchical Models
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Hierarchical or multilevel modeling is another generalization of regression modeling.

    _Multilevel models_ are regression models in which the constituent model parameters are given **probability models**. This implies that model parameters are allowed to **vary by group**.

    Observational units are often naturally **clustered**. Clustering induces dependence between observations, despite random sampling of clusters and random sampling within clusters.

    A _hierarchical model_ is a particular multilevel model where parameters are nested within one another.

    Some multilevel structures are not hierarchical.

    - e.g. "country" and "year" are not nested, but may represent separate, but overlapping, clusters of parameters

    We will motivate this topic using an environmental epidemiology example.
    """)
    return


@app.cell(hide_code=True)
def _():
    def make_radon_html():
        radon_img_path = Path(__file__).parent / "images" / "how_radon_enters.jpg"
        if radon_img_path.exists():
            radon_b64 = base64.b64encode(radon_img_path.read_bytes()).decode()
            return f'<img src="data:image/jpeg;base64,{radon_b64}" width="600">'
        return ""

    radon_html = make_radon_html()

    mo.md(f"""
    ### Example: Radon contamination (Gelman and Hill 2006)

    Radon is a radioactive gas that enters homes through contact points with the ground. It is a carcinogen that is the primary cause of lung cancer in non-smokers. Radon levels vary greatly from household to household.

    {radon_html}

    The EPA did a study of radon levels in 80,000 houses. There are two important predictors:

    - measurement in basement or first floor (radon higher in basements)
    - county uranium level (positive correlation with radon levels)

    We will focus on modeling radon levels in Minnesota.

    The hierarchy in this example is households within county.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Data organization
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First, we import the data from a local file, and extract Minnesota's data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The original data exists as several independent datasets, which we will import, merge, and process here. First is the data on measurements from individual homes from across the United States. We will extract just the subset from Minnesota.
    """)
    return


@app.cell(hide_code=True)
def _():
    data_path = Path(__file__).parent / "data"
    return (data_path,)


@app.cell(hide_code=True)
def _(data_path):
    srrs2 = pl.read_csv(data_path / "srrs2.dat")

    srrs2 = srrs2.rename({col: col.strip() for col in srrs2.columns})
    srrs_mn = srrs2.filter(pl.col("state") == "MN")
    _output = srrs_mn.shape
    srrs_mn
    return (srrs_mn,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, obtain the county-level predictor, uranium, by combining two variables.
    """)
    return


@app.cell(hide_code=True)
def _(data_path, srrs_mn):
    cty = pl.read_csv(data_path / "cty.dat")
    srrs_mn_1 = srrs_mn.with_columns(
        (
            pl.col("stfips").str.strip_chars().cast(pl.Int64) * 1000
            + pl.col("cntyfips").str.strip_chars().cast(pl.Int64)
        ).alias("fips")
    )
    cty_mn = cty.filter(pl.col("st") == "MN").with_columns(
        (1000 * pl.col("stfips") + pl.col("ctfips")).alias("fips")
    )
    return cty_mn, srrs_mn_1


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Use the Polars `join` method to combine home- and county-level information in a single DataFrame.
    """)
    return


@app.cell(hide_code=True)
def _(cty_mn, srrs_mn_1):
    srrs_mn_2 = srrs_mn_1.join(cty_mn.select(["fips", "Uppm"]), on="fips")
    srrs_mn_2 = srrs_mn_2.unique(subset=["idnum"], maintain_order=True)
    return (srrs_mn_2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's encode the county names and make local copies of the variables we will use.
    We also need a lookup table (`dict`) for each unique county, for indexing.
    """)
    return


@app.cell(hide_code=True)
def _(srrs_mn_2):
    srrs_mn_3 = srrs_mn_2.with_columns(
        pl.col("county").map_elements(str.strip, return_dtype=pl.Utf8).alias("county")
    )
    unique_counties = (
        srrs_mn_3.select("county").unique(maintain_order=True).to_series().to_list()
    )
    mn_counties = np.array(unique_counties)
    county_dict = {county: i for i, county in enumerate(unique_counties)}
    county_uranium_df = srrs_mn_3.group_by("county", maintain_order=True).agg(
        pl.col("Uppm").first()
    )
    county_to_uranium = {row[0]: row[1] for row in county_uranium_df.iter_rows()}
    ordered_uranium = [county_to_uranium[county] for county in unique_counties]
    u = np.log(np.array(ordered_uranium))
    srrs_mn_3 = srrs_mn_3.with_columns(
        pl.col("county").replace_strict(county_dict, default=None).alias("county_code")
    )
    srrs_mn_3 = srrs_mn_3.with_columns(
        pl.col("activity").str.strip_chars().cast(pl.Float64).alias("activity")
    )
    county = srrs_mn_3.select("county_code").to_numpy().flatten()
    radon = srrs_mn_3.select("activity").to_numpy().flatten()
    log_radon = np.log(radon + 0.1)
    srrs_mn_3 = srrs_mn_3.with_columns(pl.lit(log_radon).alias("log_radon"))
    floor_measure = srrs_mn_3.select("floor").to_numpy().flatten()
    return county, floor_measure, log_radon, mn_counties, srrs_mn_3, u


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Distribution of radon levels in MN (log scale):
    """)
    return


@app.cell(hide_code=True)
def _(srrs_mn_3):
    _output = px.histogram(
        srrs_mn_3,
        x="log_radon",
        nbins=50,
        labels={"log_radon": "log(radon)"},
        title="Distribution of log(radon) levels in MN",
    ).update_layout(xaxis_title="log(radon)", yaxis_title="frequency")
    _output
    return


@app.cell(hide_code=True)
def _(srrs_mn_3):
    floor_counts = srrs_mn_3.get_column("floor").value_counts().sort("floor")
    _output = px.bar(
        x=["Basement", "Floor"],
        y=floor_counts.get_column("count").to_list(),
        title="Distribution of measurement locations in MN",
        labels={"x": "Measurement location", "y": "frequency"},
    )
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conventional approaches

    The two conventional alternatives to modeling radon exposure represent the two extremes of the bias-variance tradeoff:

    **_Complete pooling_**:

    Treat all counties the same, and estimate a single radon level.

    $$y_i = \alpha + \beta x_i + \epsilon_i$$

    **_No pooling_**:

    Model radon in each county independently.

    $$y_i = \alpha_{j[i]} + \beta x_i + \epsilon_i$$

    where $j = 1,\ldots,85$

    The errors $\epsilon_i$ may represent measurement error, temporal within-house variation, or variation among houses.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > **A note on the model API used here.** PyMC v6 ships a dim-first construction API at `pymc.dims` (commonly imported as `pmd`). Distributions take `dims=...` directly, broadcasting through named dimensions automatically. We use it for the hierarchical models in this notebook because that's where dim-first construction pays off most. The classic `pm.Normal(..., dims=...)` API you may see online still works and is fully supported; the two APIs interoperate within the same `pm.Model` block. We mix them later when we get to LKJ correlated effects (the LKJ prior currently lives only in the classic namespace).

    Here are the point estimates of the slope and intercept for the complete pooling model:
    """)
    return


@app.cell
def _(floor_measure, log_radon):
    @pmx.as_model(coords={"obs_id": np.arange(len(log_radon))})
    def build_pooled():
        floor_ind = pmd.Data("floor_ind", floor_measure, dims="obs_id")
        alpha = pmd.Normal("alpha", mu=0, sigma=10)
        beta = pmd.Normal("beta", mu=0, sigma=10)
        sigma = pmd.HalfNormal("sigma", sigma=2)
        theta = alpha + beta * floor_ind
        pmd.Normal(
            "y",
            mu=theta,
            sigma=sigma,
            observed=pmd.as_xtensor(log_radon, dims=("obs_id",)),
            dims="obs_id",
        )

    pooled_model = build_pooled()
    pooled_model
    return (pooled_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You may be wondering why we are using the `pm.Data` container above even though the variable `floor_ind` is not an observed variable nor a parameter of the model. As you'll see, this will make our lives much easier when we'll plot and diagnose our model.ArviZ will thus include `floor_ind` as a variable in the `constant_data` group of the resulting {ref}`InferenceData <xarray_for_arviz>` object. Moreover, including `floor_ind` in the `InferenceData` object makes sharing and reproducing analysis much easier, all the data needed to analyze or rerun the model is stored there.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Before running the model let's do some **prior predictive checks**.

    Indeed, having sensible priors is not only a way to incorporate scientific knowledge into the model, it can also help and make the MCMC machinery faster -- here we are dealing with a simple linear regression, so no link function comes and distorts the outcome space; but one day this will happen to you and you'll need to think hard about your priors to help your MCMC sampler. So, better to train ourselves when it's quite easy than having to learn when it's very hard.
    """)
    return


@app.cell
def _(pooled_model):
    with pooled_model:
        prior_checks = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    prior_checks
    return (prior_checks,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ArviZ `InferenceData` uses `xarray.Dataset`s under the hood, which give access to several common plotting functions with `.plot`. In this case, we want scatter plot of the mean log radon level (which is stored in variable `a`) for each of the two levels we are considering. If our desired plot is supported by xarray plotting capabilities, we can take advantage of xarray to automatically generate both plot and labels for us. Notice how everything is directly plotted and annotated, the only change we need to do is renaming the y axis label from `a` to `Mean log radon level`.
    """)
    return


@app.cell(hide_code=True)
def _(prior_checks):
    prior = prior_checks.prior.dataset.squeeze(drop=True)

    _output = (
        xr.concat((prior["alpha"], prior["alpha"] + prior["beta"]), dim="location")
        .rename("log_radon")
        .assign_coords(location=["basement", "floor"])
        .plot.scatter(x="location", y="log_radon", edgecolors="none")
    )
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    I'm no radon expert, but before seeing the data, these priors seem to allow for quite a wide range of the mean log radon level, both as measured either in a basement or on a floor. But don't worry, we can always change these priors if sampling gives us hints that they might not be appropriate -- after all, priors are assumptions, not oaths; and as with most assumptions, they can be tested.

    However, we can already think of an improvement: Remember that we stated radon levels tend to be higher in basements, so we could incorporate this prior scientific knowledge into our model by forcing the floor effect (`beta`) to be negative. For now, we will leave the model as is, and trust that the information in the data will be sufficient.

    Speaking of sampling, let's fire up the Bayesian machinery!
    """)
    return


@app.cell
def _(pooled_model):
    with pooled_model:
        pooled_trace = pm.sample(random_seed=RANDOM_SEED)
    pooled_trace
    return (pooled_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    No divergences and a sampling that only took seconds! Here the chains look very good (good R hat, good effective sample size, small sd). The model also estimated a negative floor effect, as we expected.
    """)
    return


@app.cell
def _(pooled_trace):
    _output = az.summary(pooled_trace, round_to=2)
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's plot the expected radon levels in basements (`alpha`) and on floors (`alpha + beta`) in relation to the data used to fit the model:
    """)
    return


@app.cell(hide_code=True)
def _(pooled_trace, srrs_mn_3):
    def plot_pooled_fit():
        xvals = np.linspace(-0.2, 1.2, 100)
        return px.scatter(
            x=srrs_mn_3["floor"].to_numpy(),
            y=np.log(srrs_mn_3["activity"].to_numpy() + 0.1),
            labels={"x": "floor", "y": "log(activity + 0.1)"},
        ).add_scatter(
            x=xvals,
            y=post_mean["beta"].item() * xvals + post_mean["alpha"].item(),
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Posterior Mean",
        )

    post_mean = pooled_trace.posterior.dataset.mean(dim=("chain", "draw"))
    _output = plot_pooled_fit()
    _output
    return (post_mean,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This looks reasonable, though notice that there is a great deal of residual variability in the data.

    Let's now turn our attention to the unpooled model, and see how it fares in comparison.
    """)
    return


@app.cell
def _(county, floor_measure, log_radon, mn_counties):
    coords = {"county": mn_counties, "obs_id": np.arange(len(log_radon))}

    @pmx.as_model(coords=coords)
    def build_unpooled():
        floor_ind = pmd.Data("floor_ind", floor_measure, dims="obs_id")
        county_idx = pmd.Data("county_idx", county, dims="obs_id")
        alpha = pmd.Normal("alpha", mu=0, sigma=10, dims="county")
        beta = pmd.Normal("beta", mu=0, sigma=10)
        sigma = pmd.HalfNormal("sigma", sigma=2)
        theta = alpha[county_idx] + beta * floor_ind
        pmd.Normal(
            "y",
            mu=theta,
            sigma=sigma,
            observed=pmd.as_xtensor(log_radon, dims=("obs_id",)),
            dims="obs_id",
        )

    unpooled_model = build_unpooled()
    unpooled_model
    return coords, unpooled_model


@app.cell
def _(unpooled_model):
    with unpooled_model:
        unpooled_trace = pm.sample(random_seed=RANDOM_SEED)
    unpooled_trace
    return (unpooled_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The sampling was clean here too; Let's look at the expected values for both basement (dimension 0) and floor (dimension 1) in each county:
    """)
    return


@app.cell(hide_code=True)
def _(unpooled_trace):
    _output = az.plot_forest(unpooled_trace, var_names=["alpha"], combined=True)
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To identify counties with high radon levels, we can plot the ordered mean estimates, as well as their 89% ETI:
    """)
    return


@app.cell(hide_code=True)
def _(mn_counties, unpooled_trace):
    unpooled_means = unpooled_trace.posterior.dataset.mean(dim=("chain", "draw"))
    unpooled_eti = az.eti(unpooled_trace).dataset
    unpooled_means_iter = unpooled_means.sortby("alpha")
    unpooled_eti_iter = unpooled_eti.sortby(unpooled_means_iter.alpha)

    def plot_unpooled_means():
        fig, ax = plt.subplots(figsize=(10, 6))
        xticks = np.arange(0, 86, 6)
        unpooled_means_iter.plot.scatter(x="county", y="alpha", ax=ax, alpha=0.8)
        ax.vlines(
            np.arange(mn_counties.shape[0]),
            unpooled_eti_iter.alpha.sel(ci_bound="lower"),
            unpooled_eti_iter.alpha.sel(ci_bound="upper"),
            color="orange",
            alpha=0.6,
        )
        ax.set(ylabel="Radon estimate", ylim=(-2, 4.5))
        ax.set_xticks(xticks)
        ax.set_xticklabels(unpooled_means_iter.county.values[xticks])
        ax.tick_params(rotation=45)
        return fig

    _output = plot_unpooled_means()
    _output
    return (unpooled_means,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we have fit both conventional (_i.e._ non-hierarchcial) models, let's see how their inferences differ. Here are visual comparisons between the pooled and unpooled estimates for a subset of counties representing a range of sample sizes.
    """)
    return


@app.cell(hide_code=True)
def _(post_mean, srrs_mn_3, unpooled_means):
    def plot_county_comparison():
        sample_counties = (
            "LAC QUI PARLE",
            "AITKIN",
            "KOOCHICHING",
            "DOUGLAS",
            "CLAY",
            "STEARNS",
            "RAMSEY",
            "ST LOUIS",
        )
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
        axes = axes.ravel()
        m = unpooled_means["beta"]
        for i, c in enumerate(sample_counties):
            y = srrs_mn_3.filter(pl.col("county") == c)["log_radon"].to_numpy()
            x = srrs_mn_3.filter(pl.col("county") == c)["floor"].to_numpy()
            axes[i].scatter(x + np.random.randn(len(x)) * 0.01, y, alpha=0.4)
            b = unpooled_means["alpha"].sel(county=c)
            xvals = xr.DataArray(np.linspace(0, 1))
            axes[i].plot(xvals, m * xvals + b)
            axes[i].plot(xvals, post_mean["beta"] * xvals + post_mean["alpha"], "r--")
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(["basement", "floor"])
            axes[i].set_ylim(-1, 3)
            axes[i].set_title(c)
            if not i % 2:
                axes[i].set_ylabel("log radon level")
        return fig

    _output = plot_county_comparison()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Neither of these models are satisfactory:

    - If we are trying to identify high-radon counties, pooling is useless -- because, by definition, the pooled model estimates radon at the state-level. In other words, pooling leads to maximal _underfitting_: the variation across counties is not taken into account and only the overall population is estimated.
    - We do not trust extreme unpooled estimates produced by models using few observations. This leads to maximal _overfitting_: only the within-county variations are taken into account and the overall population (i.e the state-level, which tells us about similarities across counties) is not estimated.

    This issue is acute for small sample sizes, as seen above: in counties where we have few floor measurements, if radon levels are higher for those data points than for basement ones (Aitkin, Koochiching, Ramsey), the model will estimate that radon levels are higher in floors than basements for these counties. But we shouldn't trust this conclusion, because both scientific knowledge and the situation in other counties tell us that it is usually the reverse (basement radon > floor radon). So unless we have a lot of observations telling us otherwise for a given county, we should be skeptical and shrink our county-estimates to the state-estimates -- in other words, we should balance between cluster-level and population-level information, and the amount of shrinkage will depend on how extreme and how numerous the data in each cluster are.

    Here is where hierarchical models come into play.
    """)
    return


@app.cell(hide_code=True)
def _():
    def embed_model_images():
        img_dir = Path(__file__).parent / "images"

        def embed_png(name):
            p = img_dir / name
            if p.exists():
                d = base64.b64encode(p.read_bytes()).decode()
                return f'<img src="data:image/png;base64,{d}" width="500">'
            return ""

        return (
            embed_png("pooled_model.png"),
            embed_png("unpooled_model.png"),
            embed_png("partial_pooled_model.png"),
        )

    pooled_img, unpooled_img, partial_img = embed_model_images()

    mo.md(f"""
    ## Multilevel and hierarchical models

    When we pool our data, we imply that they are sampled from the same model. This ignores any variation among sampling units (other than sampling variance) -- we assume that counties are all the same:

    {pooled_img}

    When we analyze data unpooled, we imply that they are sampled independently from separate models. At the opposite extreme from the pooled case, this approach claims that differences between sampling units are too large to combine them -- we assume that counties have no similarity whatsoever:

    {unpooled_img}

    In a hierarchical model, parameters are viewed as a sample from a population distribution of parameters. Thus, we view them as being neither entirely different or exactly the same. This is **_partial pooling_**:

    {partial_img}

    We can use PyMC to easily specify multilevel models, and fit them using Markov chain Monte Carlo.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Partial pooling model

    The simplest partial pooling model for the household radon dataset is one which simply estimates radon levels, without any predictors at any level. A partial pooling model represents a compromise between the pooled and unpooled extremes, essentially a weighted average (based on sample size) of the unpooled county estimates and the pooled estimates.

    $$\hat{\alpha} \approx \frac{(n_j/\sigma_y^2)\bar{y}_j + (1/\sigma_{\alpha}^2)\bar{y}}{(n_j/\sigma_y^2) + (1/\sigma_{\alpha}^2)}$$

    Estimates for counties with smaller sample sizes will shrink towards the state-wide average, while those for counties with larger sample sizes will be closer to the unpooled county estimates.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's start with a very simple partial pooling model, which ignores the effect of floor vs. basement measurement.
    """)
    return


@app.cell
def _(coords, county, log_radon):
    @pmx.as_model(coords=coords)
    def build_partial_pooling():
        county_idx = pmd.Data("county_idx", county, dims="obs_id")
        mu_a = pmd.Normal("mu_a", mu=0.0, sigma=10)
        sigma_a = pmd.HalfNormal("sigma_a", sigma=2)
        alpha = pmd.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")
        sigma_y = pmd.HalfNormal("sigma_y", sigma=2)
        y_hat = alpha[county_idx]
        pmd.Normal(
            "y_like",
            mu=y_hat,
            sigma=sigma_y,
            observed=pmd.as_xtensor(log_radon, dims=("obs_id",)),
            dims="obs_id",
        )

    partial_pooling = build_partial_pooling()
    partial_pooling
    return (partial_pooling,)


@app.cell
def _(partial_pooling):
    with partial_pooling:
        partial_pooling_trace = pm.sample(tune=2000, random_seed=RANDOM_SEED)
    partial_pooling_trace
    return (partial_pooling_trace,)


@app.cell(hide_code=True)
def _(partial_pooling_trace, srrs_mn_3, unpooled_trace):
    N_county = (
        srrs_mn_3.group_by("county")
        .agg(pl.count("idnum"))
        .sort("county")["idnum"]
        .to_numpy()
    )

    def plot_pooling_comparison():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        for ax, trace, level in zip(
            axes,
            (unpooled_trace, partial_pooling_trace),
            ("no pooling", "partial pooling"),
        ):
            post_ds = trace.posterior.dataset.assign_coords(
                {"N_county": ("county", N_county)}
            )
            post_ds.mean(dim=("chain", "draw")).plot.scatter(
                x="N_county", y="alpha", ax=ax, alpha=0.9
            )
            ax.hlines(
                partial_pooling_trace.posterior["alpha"].mean(),
                0.9,
                max(N_county) + 1,
                alpha=0.4,
                ls="--",
                label="Est. population mean",
            )
            hdi = az.eti(trace).dataset.alpha
            ax.vlines(
                N_county,
                hdi.sel(ci_bound="lower"),
                hdi.sel(ci_bound="upper"),
                color="orange",
                alpha=0.5,
            )
            ax.set(
                title=f"{level.title()} Estimates",
                xlabel="Nbr obs in county (log scale)",
                xscale="log",
                ylabel="Log radon",
            )
            ax.legend(fontsize=10)
        return fig

    _output = plot_pooling_comparison()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice the difference between the unpooled and partially-pooled estimates, particularly at smaller sample sizes: As expected, the former are both more extreme and more imprecise. Indeed, in the partially-pooled model, estimates in small-sample-size counties are informed by the population parameters -- hence more precise estimates. Moreover, the smaller the sample size, the more regression towards the overall mean (the dashed gray line) -- hence less extreme estimates. In other words, the model is skeptical of extreme deviations from the population mean in counties where data is sparse. This is known as **shrinkage**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now let's go back and integrate the `floor` predictor, but allowing the intercept to vary by county.

    ## Varying intercept model

    This model allows intercepts to vary across county, according to a random effect.

    $$y_i = \alpha_{j[i]} + \beta x_{i} + \epsilon_i$$

    where

    $$\epsilon_i \sim N(0, \sigma_y^2)$$

    and the intercept random effect:

    $$\alpha_{j[i]} \sim N(\mu_{\alpha}, \sigma_{\alpha}^2)$$

    As with the the “no-pooling” model, we set a separate intercept for each county, but rather than fitting separate least squares regression models for each county, multilevel modeling **shares strength** among counties, allowing for more reasonable inference in counties with little data.
    """)
    return


@app.cell
def _(coords, county, floor_measure, log_radon):
    @pmx.as_model(coords=coords)
    def build_varying_intercept():
        floor_idx = pmd.Data("floor_idx", floor_measure, dims="obs_id")
        county_idx = pmd.Data("county_idx", county, dims="obs_id")
        mu_a = pmd.Normal("mu_a", mu=0.0, sigma=10.0)
        sigma_a = pmd.HalfNormal("sigma_a", sigma=2)
        alpha = pmd.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")
        beta = pmd.Normal("beta", mu=0.0, sigma=10.0)
        sd_y = pmd.HalfNormal("sd_y", sigma=2)
        y_hat = alpha[county_idx] + beta * floor_idx
        pmd.Normal(
            "y_like",
            mu=y_hat,
            sigma=sd_y,
            observed=pmd.as_xtensor(log_radon, dims=("obs_id",)),
            dims="obs_id",
        )

    varying_intercept = build_varying_intercept()
    varying_intercept
    return (varying_intercept,)


@app.cell
def _(varying_intercept):
    with varying_intercept:
        varying_intercept_trace = pm.sample(tune=2000, random_seed=RANDOM_SEED)
    varying_intercept_trace
    return (varying_intercept_trace,)


@app.cell(hide_code=True)
def _(varying_intercept_trace):
    _output = az.plot_forest(
        varying_intercept_trace, var_names=["alpha"], combined=True
    )
    _output
    return


@app.cell(hide_code=True)
def _(varying_intercept_trace):
    _output = az.plot_dist(varying_intercept_trace, var_names=["sigma_a", "beta"])
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The estimate for the `floor` coefficient is approximately -0.66, which can be interpreted as houses without basements having about half ($\exp(-0.66) = 0.52$) the radon levels of those with basements, after accounting for county.
    """)
    return


@app.cell
def _(varying_intercept_trace):
    _output = az.summary(varying_intercept_trace, var_names=["beta"])
    _output
    return


@app.cell(hide_code=True)
def _(varying_intercept_trace):
    def plot_varying_intercept_radon():
        xvals = xr.DataArray(
            [0, 1], dims="Level", coords={"Level": ["Basement", "Floor"]}
        )
        post = varying_intercept_trace.posterior  # alias for readability
        theta = (
            (post.alpha + post.beta * xvals)
            .mean(dim=("chain", "draw"))
            .to_dataset(name="Mean log radon")
        )
        fig, ax = plt.subplots()
        theta.plot.scatter(x="Level", y="Mean log radon", alpha=0.2, color="k", ax=ax)
        ax.plot(xvals, theta["Mean log radon"].T, "k-", alpha=0.2)
        ax.set_title("MEAN LOG RADON BY COUNTY")
        # add lines too
        return fig  # scatter

    _output = plot_varying_intercept_radon()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It is easy to show that the partial pooling model provides more objectively reasonable estimates than either the pooled or unpooled models, at least for counties with small sample sizes.
    """)
    return


@app.cell(hide_code=True)
def _(post_mean, srrs_mn_3, unpooled_means, varying_intercept_trace):
    def plot_partial_pooling_comparison():
        sample_counties = (
            "LAC QUI PARLE",
            "AITKIN",
            "KOOCHICHING",
            "DOUGLAS",
            "CLAY",
            "STEARNS",
            "RAMSEY",
            "ST LOUIS",
        )
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
        axes = axes.ravel()
        m = unpooled_means["beta"]
        for i, c in enumerate(sample_counties):
            y = srrs_mn_3.filter(pl.col("county") == c)["log_radon"].to_numpy()
            x = srrs_mn_3.filter(pl.col("county") == c)["floor"].to_numpy()
            axes[i].scatter(x + np.random.randn(len(x)) * 0.01, y, alpha=0.4)
            b = unpooled_means["alpha"].sel(county=c)
            xvals = xr.DataArray(np.linspace(0, 1))
            axes[i].plot(xvals, m.values * xvals + b.values)
            axes[i].plot(xvals, post_mean["beta"] * xvals + post_mean["alpha"], "r--")
            varying_intercept_trace.posterior.sel(county=c).beta
            post = varying_intercept_trace.posterior.sel(county=c).mean(
                dim=("chain", "draw")
            )
            theta = post.alpha.values + post.beta.values * xvals
            axes[i].plot(xvals, theta, "k:")
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(["basement", "floor"])
            axes[i].set_ylim(-1, 3)
            axes[i].set_title(c)
            if not i % 2:
                axes[i].set_ylabel("log radon level")
        return fig

    _output = plot_partial_pooling_comparison()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Varying intercept and slope model

    The most general model allows both the intercept and slope to vary by county:

    $$y_i = \alpha_{j[i]} + \beta_{j[i]} x_{i} + \epsilon_i$$

    Complete the model inside the `exercise_varying_intercept_slope` scaffold
    below, then click ▶ Run exercise. The notebook continues with a reference
    implementation named `varying_intercept_slope` (from the Solution), so the
    rest of the notebook works whether or not you complete the exercise.

    Plot the model DAG to check your structure — the scaffold already returns
    the model object, which marimo renders as the model graph.
    """)
    return


@app.cell
def _(coords, county, floor_measure, log_radon):
    def exercise_varying_intercept_slope():
        @pmx.as_model(coords=coords)
        def my_varying_model():
            floor_idx = pmd.Data("floor_idx", floor_measure, dims="obs_id")
            county_idx = pmd.Data("county_idx", county, dims="obs_id")
            obs = pmd.as_xtensor(log_radon, dims=("obs_id",))
            # YOUR CODE HERE — hyperpriors for the intercept and slope
            # YOUR CODE HERE — county-level alpha and beta (dims="county")
            # YOUR CODE HERE — sigma_y, the expected value
            #   alpha[county_idx] + beta[county_idx] * floor_idx,
            #   and the Normal likelihood (observed=obs, dims="obs_id")
            ...

        model = my_varying_model()
        return model

    return (exercise_varying_intercept_slope,)


@app.cell(hide_code=True)
def _():
    run_varying_intercept_slope = mo.ui.run_button(label="▶ Run exercise")
    run_varying_intercept_slope
    return (run_varying_intercept_slope,)


@app.cell(hide_code=True)
def _(exercise_varying_intercept_slope, run_varying_intercept_slope):
    mo.stop(
        not run_varying_intercept_slope.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_varying_intercept_slope()
    return


@app.cell(hide_code=True)
def _(coords, county, floor_measure, log_radon):
    def solution_varying_intercept_slope():
        @pmx.as_model(coords=coords)
        def varying_intercept_slope_model():
            floor_idx = pmd.Data("floor_idx", floor_measure, dims="obs_id")
            county_idx = pmd.Data("county_idx", county, dims="obs_id")
            mu_a = pmd.Normal("mu_a", mu=0.0, sigma=10.0)
            sigma_a = pmd.HalfNormal("sigma_a", sigma=2)
            mu_b = pmd.Normal("mu_b", mu=0.0, sigma=10.0)
            sigma_b = pmd.HalfNormal("sigma_b", sigma=2)
            alpha = pmd.Normal("alpha", mu=mu_a, sigma=sigma_a, dims="county")
            beta = pmd.Normal("beta", mu=mu_b, sigma=sigma_b, dims="county")
            sigma_y = pmd.HalfNormal("sigma_y", sigma=2)
            y_hat = alpha[county_idx] + beta[county_idx] * floor_idx
            pmd.Normal(
                "y_like",
                mu=y_hat,
                sigma=sigma_y,
                observed=pmd.as_xtensor(log_radon, dims=("obs_id",)),
                dims="obs_id",
            )

        return varying_intercept_slope_model()

    varying_intercept_slope = solution_varying_intercept_slope()

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_varying_intercept_slope)}\n```"
                    ),
                    varying_intercept_slope,
                    mo.md(
                        "_This model (`varying_intercept_slope`) is sampled below, so the rest of the notebook works whether or not you complete the exercise._"
                    ),
                ]
            ),
        }
    )
    return (varying_intercept_slope,)


@app.cell
def _(varying_intercept_slope):
    with varying_intercept_slope:
        varying_intercept_slope_trace = pm.sample(
            tune=2000, target_accept=0.95, random_seed=RANDOM_SEED
        )
    varying_intercept_slope_trace
    return (varying_intercept_slope_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice that the trace of this model includes divergences, which can be problematic depending on where and how frequently they occur. These can occur in some hierarchical models, and they can be avoided by using the **non-centered parametrization**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Non-centered Parameterization

    The partial pooling models specified above uses a **centered** parameterization of the slope random effect. That is, the individual county effects are distributed around a county mean, with a spread controlled by the hierarchical standard deviation parameter. As the preceding plot reveals, this constraint serves to **shrink** county estimates toward the overall mean, to a degree proportional to the county sample size. This is exactly what we want, and the model appears to fit well--the Gelman-Rubin statistics are exactly 1.

    But, on closer inspection, there are signs of trouble. Specifically, let's look at the trace of the random effects, and their corresponding standard deviation:
    """)
    return


@app.cell(hide_code=True)
def _(varying_intercept_slope_trace):
    def plot_centered_traces():
        # Extract posterior samples for chain 0 using polars
        sigma_b_df = (
            varying_intercept_slope_trace.posterior["sigma_b"]
            .sel(chain=0)
            .to_dataframe()
            .reset_index()
        )
        beta_df = (
            varying_intercept_slope_trace.posterior["beta"]
            .sel(chain=0)
            .to_dataframe()
            .reset_index()
        )
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(sigma_b_df["sigma_b"].to_numpy(), alpha=0.5)
        axs[0].set(ylabel="sigma_b")
        axs[1].plot(beta_df["beta"].to_numpy(), alpha=0.5)
        axs[1].set(ylabel="beta")
        return fig

    _output = plot_centered_traces()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice that when the chain reaches the lower end of the parameter space for $\sigma_b$, it appears to get "stuck" and the entire sampler, including the random slopes `beta`, mixes poorly.

    Jointly plotting the random effect variance and one of the individual random slopes demonstrates what is going on.
    """)
    return


@app.cell(hide_code=True)
def _(varying_intercept_slope_trace):
    def plot_centered_funnel():
        x = (
            varying_intercept_slope_trace.posterior["beta"]
            .sel(county="AITKIN")
            .values.flatten()
        )
        y = varying_intercept_slope_trace.posterior["sigma_b"].values.flatten()
        diverging_mask = (
            varying_intercept_slope_trace.sample_stats["diverging"]
            .values.flatten()
            .astype(bool)
        )
        return (
            go.Figure()
            .add_trace(
                go.Scatter(
                    x=x[~diverging_mask],
                    y=y[~diverging_mask],
                    mode="markers",
                    name="Non-diverging",
                    marker=dict(color="blue", size=8, opacity=0.2),
                )
            )
            .add_trace(
                go.Scatter(
                    x=x[diverging_mask],
                    y=y[diverging_mask],
                    mode="markers",
                    name="Diverging",
                    marker=dict(color="orange", size=8, opacity=0.7),
                )
            )
            .update_layout(
                title="Neal's Funnel",
                xaxis_title="Beta (AITKIN county)",
                yaxis_title="Sigma_b",
                showlegend=True,
                yaxis=dict(range=[0, None]),
            )
        )

    _output = plot_centered_funnel()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When the group variance is small, this implies that the individual random slopes are themselves close to the group mean. This results in a _funnel_-shaped relationship between the samples of group variance and any of the slopes (particularly those with a smaller sample size).

    In itself, this is not a problem, since this is the behavior we expect. However, if the sampler is tuned for the wider (unconstrained) part of the parameter space, it has trouble in the areas of higher curvature. The consequence of this is that the neighborhood close to the lower bound of $\sigma_b$ is sampled poorly; indeed, in our chain it is not sampled at all below 0.1. In addtion, the sampler generates a lot of divergent samples. The result of this will be biased inference.

    Now that we've spotted the problem, what can we do about it? Before we rewrite the model, it's worth trying a smaller change first.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Option A: low-rank mass-matrix adaptation

    PyMC v6's `nutpie` sampler can adapt a **low-rank** mass matrix in addition to the diagonal one. The low-rank correction captures the dominant correlations between parameters, which often gets you through funnel-prone geometries without rewriting the model. You enable it with one keyword:

    ```python
    pm.sample(nuts={"adaptation": "low_rank"})
    ```

    Let's see what it does to the centered parameterization.
    """)
    return


@app.cell
def _(varying_intercept_slope):
    with varying_intercept_slope:
        centered_lowrank_trace = pm.sample(
            tune=2000,
            random_seed=RANDOM_SEED,
            nuts={"adaptation": "low_rank"},
            progressbar=False,
        )
    centered_lowrank_trace
    return (centered_lowrank_trace,)


@app.cell(hide_code=True)
def _(centered_lowrank_trace, varying_intercept_slope_trace):
    centered_divergences = int(
        varying_intercept_slope_trace.sample_stats["diverging"].sum().values
    )
    lowrank_divergences = int(
        centered_lowrank_trace.sample_stats["diverging"].sum().values
    )
    lowrank_compare = pl.DataFrame(
        {
            "adaptation": [
                "default (diagonal)",
                "low_rank",
            ],
            "divergences": [centered_divergences, lowrank_divergences],
        }
    )
    mo.vstack(
        [
            mo.md(
                "**Divergent transitions on the centered model under different nutpie adaptations:**"
            ),
            lowrank_compare,
            mo.md(
                "Low-rank adaptation often eliminates or sharply reduces divergences on funnel-prone hierarchies without changing the model. It costs a bit more per tuning iteration in exchange. When it works, you keep the more natural centered parameterization; when it doesn't (very tight funnels, hundreds of correlated parameters), fall back to reparameterization."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Option B: non-centered parameterization (reparameterize the model)

    Sometimes the geometry is bad enough that no amount of sampler tuning will recover it. The classic fix is to rewrite the random effects so they are drawn from a unit normal and then scaled and shifted by the group-level mean and SD. Notice the random slopes in this version:
    """)
    return


@app.cell
def _(coords, county, floor_measure, log_radon):
    @pmx.as_model(coords=coords)
    def build_noncentered():
        floor_idx = pmd.Data("floor_idx", floor_measure, dims="obs_id")
        county_idx = pmd.Data("county_idx", county, dims="obs_id")
        mu_a = pmd.Normal("mu_a", mu=0.0, sigma=10.0)
        sigma_a = pmd.HalfNormal("sigma_a", sigma=2)
        z_a = pmd.Normal("z_a", mu=0, sigma=1, dims="county")
        alpha = pmd.Deterministic("alpha", mu_a + z_a * sigma_a, dims="county")
        mu_b = pmd.Normal("mu_b", mu=0.0, sigma=10.0)
        sigma_b = pmd.HalfNormal("sigma_b", sigma=2)
        z_b = pmd.Normal("z_b", mu=0, sigma=1, dims="county")
        beta = pmd.Deterministic("beta", mu_b + z_b * sigma_b, dims="county")
        sigma_y = pmd.HalfNormal("sigma_y", sigma=2)
        y_hat = alpha[county_idx] + beta[county_idx] * floor_idx
        pmd.Normal(
            "y_like",
            mu=y_hat,
            sigma=sigma_y,
            observed=pmd.as_xtensor(log_radon, dims=("obs_id",)),
            dims="obs_id",
        )

    varying_intercept_slope_noncentered = build_noncentered()
    varying_intercept_slope_noncentered
    return (varying_intercept_slope_noncentered,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This is a [**non-centered** parameterization](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/). By this, we mean that the random deviates are no longer explicitly modeled as being centered on $\mu_b$. Instead, they are independent standard normals $\upsilon$, which are then scaled by the appropriate value of $\sigma_b$, before being location-transformed by the mean.

    This model samples much better.
    """)
    return


@app.cell
def _(varying_intercept_slope_noncentered):
    with varying_intercept_slope_noncentered:
        noncentered_trace = pm.sample(
            tune=3000,
            target_accept=0.95,
            random_seed=RANDOM_SEED,
        )
    noncentered_trace
    return (noncentered_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice that the bottlenecks in the traces are gone.
    """)
    return


@app.cell(hide_code=True)
def _(noncentered_trace):
    def plot_noncentered_traces():
        # Extract posterior samples for chain 0 using polars
        sigma_b_df = (
            noncentered_trace.posterior["sigma_b"]
            .sel(chain=0)
            .to_dataframe()
            .reset_index()
        )
        beta_df = (
            noncentered_trace.posterior["beta"]
            .sel(chain=0)
            .to_dataframe()
            .reset_index()
        )
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(sigma_b_df["sigma_b"].to_numpy(), alpha=0.5)
        axs[0].set(ylabel="sigma_b")
        axs[1].plot(beta_df["beta"].to_numpy(), alpha=0.5)
        axs[1].set(ylabel="beta")
        return fig

    _output = plot_noncentered_traces()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And correspondingly, the low end of the posterior distribution of the slope random effect variance can now be sampled efficiently.
    """)
    return


@app.cell(hide_code=True)
def _(noncentered_trace):
    def plot_noncentered_funnel():
        x = noncentered_trace.posterior["beta"].sel(county="AITKIN").values.flatten()
        y = noncentered_trace.posterior["sigma_b"].values.flatten()
        diverging_mask = (
            noncentered_trace.sample_stats["diverging"].values.flatten().astype(bool)
        )
        return (
            go.Figure()
            .add_trace(
                go.Scatter(
                    x=x[~diverging_mask],
                    y=y[~diverging_mask],
                    mode="markers",
                    name="Non-diverging",
                    marker=dict(color="blue", size=8, opacity=0.2),
                )
            )
            .add_trace(
                go.Scatter(
                    x=x[diverging_mask],
                    y=y[diverging_mask],
                    mode="markers",
                    name="Diverging",
                    marker=dict(color="orange", size=8, opacity=0.7),
                )
            )
            .update_layout(
                title="Neal's Funnel (non-centered)",
                xaxis_title="Beta (AITKIN county)",
                yaxis_title="Sigma_b",
                showlegend=True,
                yaxis=dict(range=[0, None]),
            )
        )

    _output = plot_noncentered_funnel()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As a result, we are now fully exploring the support of the posterior. This results in less bias in these parameters.
    """)
    return


@app.cell(hide_code=True)
def _(varying_intercept_slope_trace):
    # Compare sigma_b posteriors for centered vs non-centered parameterizations
    _output = az.plot_dist(varying_intercept_slope_trace, var_names=["sigma_b"])
    _output
    return


@app.cell(hide_code=True)
def _(noncentered_trace):
    _output = az.plot_dist(noncentered_trace, var_names=["sigma_b"])
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice that `sigma_b` now has a lot of density near zero, which would indicate that counties don't vary that much in their answer to the `floor` "treatment".

    This was the problem with the original parameterization: the sampler has difficulty with the geometry of the posterior distribution when the values of the slope random effects are so different for standard deviations very close to zero compared to when they are positive. However, even with the non-centered model the sampler is not that comfortable with `sigma_b`: in fact if you look at the estimates with `az.summary` you'll see that the number of effective samples is quite low for `sigma_b`.

    Also note that `sigma_a` is not that big either -- i.e counties do differ in their baseline radon levels, but not by a lot. However we don't have that much of a problem to sample from this distribution because it's much narrower than `sigma_b` and doesn't get dangerously close to 0.
    """)
    return


@app.cell
def _(varying_intercept_slope_trace):
    _output = az.summary(
        varying_intercept_slope_trace, var_names=["sigma_a", "sigma_b"]
    )
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To wrap up this model, let's plot the relationship between radon and floor for each county:
    """)
    return


@app.cell(hide_code=True)
def _(noncentered_trace):
    def plot_noncentered_radon():
        xvals = xr.DataArray(
            [0, 1], dims="Level", coords={"Level": ["Basement", "Floor"]}
        )
        post = noncentered_trace.posterior  # alias for readability
        theta = (
            (post.alpha + post.beta * xvals)
            .mean(dim=("chain", "draw"))
            .to_dataset(name="Mean log radon")
        )
        fig, ax = plt.subplots()
        theta.plot.scatter(x="Level", y="Mean log radon", alpha=0.2, color="k", ax=ax)
        ax.plot(xvals, theta["Mean log radon"].T, "k-", alpha=0.2)
        ax.set_title("MEAN LOG RADON BY COUNTY")
        # add lines too
        return fig  # scatter

    _output = plot_noncentered_radon()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This, while both the intercept and the slope vary by county, there is far less variation in the slope.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > **Note for learners:** This section is an advanced extension. It introduces correlated random effects
    > via an LKJ prior. The rest of the notebook does not depend on this section — you may skip to
    > "Adding group-level predictors" if you prefer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Correlated Random Effects

    In the model above, the varying intercepts and slopes are drawn independently. But it's natural to expect them to be **correlated** — for example, counties with higher baseline radon (higher intercept) might show a stronger floor effect (steeper slope).

    We can model this correlation using a **multivariate normal** prior for the joint distribution of intercepts and slopes:

    $$
    \begin{pmatrix} \alpha_j \\ \beta_j \end{pmatrix} \sim \text{MvNormal}\left(
    \begin{pmatrix} \mu_\alpha \\ \mu_\beta \end{pmatrix},
    \Sigma \right)
    $$

    The covariance matrix $\Sigma$ is decomposed as:

    $$\Sigma = \text{diag}(\sigma) \cdot R \cdot \text{diag}(\sigma)$$

    where $R$ is a **correlation matrix** and $\sigma$ are standard deviations.

    ### The LKJ Prior for Correlation Matrices

    PyMC provides `pm.LKJCholeskyCov` for jointly sampling the correlation matrix and standard deviations:

    ```python
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol", n=2, eta=2.0,
        sd_dist=pm.Exponential.dist(1.0),
    )
    ```

    The `eta` parameter controls the prior on correlations:
    - `eta=1`: uniform over valid correlation matrices
    - `eta=2`: mild preference for weaker correlations (often a good default)
    - `eta>2`: increasingly concentrated near the identity matrix

    Estimating the correlation lets us:
    - Understand **how** group-level parameters relate to each other
    - Make better predictions by leveraging the correlation structure
    - Detect structural patterns in the data (e.g., "counties with high baselines have strong floor effects")

    This extension is most valuable when you have varying intercepts *and* slopes, and enough groups to estimate the correlation reliably.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Implementing correlated intercepts and slopes

    `pymc.dims` covers the continuous likelihoods we have used so far, but specialized matrix priors (LKJ, Wishart) still live in the classic `pm.*` namespace. We can mix them inside the same `pm.Model`: build the cholesky factor with `pm.LKJCholeskyCov`, tag the resulting tensor with dim labels using `pmd.as_xtensor`, and feed it to `pmd.MvNormal`.
    """)
    return


@app.cell
def _(coords, county, floor_measure, log_radon, mn_counties):
    re_coords = {
        "county": mn_counties,
        "obs_id": np.arange(len(log_radon)),
        "param": ["intercept", "slope"],
        "param_": ["intercept", "slope"],
    }

    @pmx.as_model(coords=re_coords)
    def build_correlated_random_effects():
        floor_idx = pmd.Data("floor_idx", floor_measure, dims="obs_id")
        county_idx = pmd.Data("county_idx", county, dims="obs_id")

        sd_dist = pm.HalfNormal.dist(1.0, shape=2)
        chol, corr, sigmas = pm.LKJCholeskyCov(
            "chol_cov", n=2, eta=2.0, sd_dist=sd_dist
        )
        chol_x = pmd.as_xtensor(chol, dims=("param", "param_"))

        mu_re = pmd.Normal("mu_re", mu=0.0, sigma=10, dims="param")
        re = pmd.MvNormal(
            "re",
            mu=mu_re,
            chol=chol_x,
            core_dims=("param", "param_"),
            dims=("county", "param"),
        )

        alpha = re[:, 0]  # intercept column
        beta = re[:, 1]  # slope column

        sigma_y = pmd.HalfNormal("sigma_y", sigma=2)
        y_hat = alpha[county_idx] + beta[county_idx] * floor_idx
        pmd.Normal(
            "y_like",
            mu=y_hat,
            sigma=sigma_y,
            observed=pmd.as_xtensor(log_radon, dims=("obs_id",)),
            dims="obs_id",
        )

    _ = coords  # unused; coords is rebuilt locally with the extra param/param_ axes
    correlated_random_effects = build_correlated_random_effects()
    correlated_random_effects
    return (correlated_random_effects,)


@app.cell
def _(correlated_random_effects):
    with correlated_random_effects:
        correlated_re_trace = pm.sample(
            tune=2000,
            target_accept=0.95,
            random_seed=RANDOM_SEED,
            progressbar=False,
        )
    correlated_re_trace
    return (correlated_re_trace,)


@app.cell(hide_code=True)
def _(correlated_re_trace):
    correlated_summary = az.summary(
        correlated_re_trace,
        var_names=["chol_cov_corr", "chol_cov_stds", "mu_re"],
        round_to=3,
    )
    correlated_summary
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The off-diagonal entries of `chol_cov_corr` give the posterior for the correlation between county intercepts and county slopes. If the credible interval crosses zero, the data don't strongly support correlated random effects on this dataset, and the independent-effects model from the previous section is a reasonable simplification.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Adding group-level predictors

    A primary strength of multilevel models is the ability to handle predictors on multiple levels simultaneously. If we consider the varying-intercepts model above:

    $$y_i = \alpha_{j[i]} + \beta x_{i} + \epsilon_i$$

    we may, instead of a simple random effect to describe variation in the expected radon value, specify another regression model with a county-level covariate. Here, we use the county uranium reading $u_j$, which is thought to be related to radon levels:

    $$\alpha_j = \gamma_0 + \gamma_1 u_j + \zeta_j$$

    $$\zeta_j \sim N(0, \sigma_{\alpha}^2)$$

    Thus, we are now incorporating a house-level predictor (floor or basement) as well as a county-level predictor (uranium).

    Note that the model has both indicator variables for each county, plus a county-level covariate. In classical regression, this would result in collinearity. In a multilevel model, the partial pooling of the intercepts towards the expected value of the group-level linear model avoids this.

    Group-level predictors also serve to reduce group-level variation, $\sigma_{\alpha}$ (here it would be the variation across counties, `sigma_a`). An important implication of this is that the group-level estimate induces stronger pooling -- by definition, a smaller $\sigma_{\alpha}$ means a stronger shrinkage of counties parameters towards the overall state mean.

    This is fairly straightforward to implement in PyMC -- we just add another level:
    """)
    return


@app.cell
def _(coords, county, floor_measure, log_radon, u):
    @pmx.as_model(coords=coords)
    def build_hierarchical_intercept():
        u_county = pmd.Data("u_county", u, dims="county")
        floor_idx = pmd.Data("floor_idx", floor_measure, dims="obs_id")
        county_idx = pmd.Data("county_idx", county, dims="obs_id")
        sigma_a = pmd.HalfCauchy("sigma_a", beta=5)
        gamma_0 = pmd.Normal("gamma_0", mu=0.0, sigma=10.0)
        gamma_1 = pmd.Normal("gamma_1", mu=0.0, sigma=10.0)
        mu_a = pmd.Deterministic("mu_a", gamma_0 + gamma_1 * u_county, dims="county")
        epsilon_a = pmd.Normal("epsilon_a", mu=0, sigma=1, dims="county")
        alpha = pmd.Deterministic("alpha", mu_a + sigma_a * epsilon_a, dims="county")
        beta = pmd.Normal("beta", mu=0.0, sigma=10.0)
        sigma_y = pmd.HalfNormal("sigma_y", sigma=2)
        y_hat = alpha[county_idx] + beta * floor_idx
        pmd.Normal(
            "y_like",
            mu=y_hat,
            sigma=sigma_y,
            observed=pmd.as_xtensor(log_radon, dims=("obs_id",)),
            dims="obs_id",
        )

    hierarchical_intercept = build_hierarchical_intercept()
    hierarchical_intercept
    return (hierarchical_intercept,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Do you see the new level, with `sigma_a` and `gamma`, which is two-dimensional because it contains the linear model for `a_county`?
    """)
    return


@app.cell
def _(hierarchical_intercept):
    with hierarchical_intercept:
        hierarchical_intercept_trace = pm.sample(tune=2000, random_seed=RANDOM_SEED)
    hierarchical_intercept_trace
    return (hierarchical_intercept_trace,)


@app.cell(hide_code=True)
def _(hierarchical_intercept_trace, u):
    def plot_uranium_intercept():
        uranium = u
        post = hierarchical_intercept_trace.posterior.dataset.assign_coords(
            uranium=uranium
        )
        avg_a = post["mu_a"].mean(dim=("chain", "draw")).values[np.argsort(uranium)]
        avg_a_county = post["alpha"].mean(dim=("chain", "draw"))
        avg_a_county_hdi = az.eti(hierarchical_intercept_trace, var_names=["alpha"])[
            "alpha"
        ]
        mu_a_hdi = az.eti(hierarchical_intercept_trace, var_names=["mu_a"])["mu_a"]
        # Calculate ETI for the trend line (mu_a)
        sorted_indices = np.argsort(uranium)
        fig = (
            go.Figure()
            .add_trace(
                go.Scatter(
                    x=np.concatenate(
                        [uranium[sorted_indices], uranium[sorted_indices][::-1]]
                    ),
                    y=np.concatenate(
                        [
                            mu_a_hdi.sel(ci_bound="lower").values[sorted_indices],
                            mu_a_hdi.sel(ci_bound="upper").values[sorted_indices][::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor="rgba(128,128,128,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=True,
                    name="Mean intercept ETI",
                    hoverinfo="skip",
                )
            )
            .add_trace(
                go.Scatter(
                    x=uranium[sorted_indices],
                    y=avg_a,
                    mode="lines",
                    line=dict(dash="dash", color="black", width=2),
                    opacity=0.6,
                    name="Mean intercept",
                )
            )
            .add_trace(
                go.Scatter(
                    x=uranium,
                    y=avg_a_county,
                    mode="markers",
                    marker=dict(color="teal", size=6, opacity=0.8),
                    name="Mean county-intercept",
                )
            )
        )
        for i, u_val in enumerate(uranium):
            fig.add_shape(
                type="line",
                x0=u_val,
                x1=u_val,
                y0=avg_a_county_hdi.sel(ci_bound="lower").values[i],
                y1=avg_a_county_hdi.sel(ci_bound="upper").values[i],
                line=dict(color="orange", width=1.5),
                opacity=0.7,
            )
        fig.update_layout(
            xaxis_title="County-level uranium",
            yaxis_title="Intercept estimate",
            showlegend=True,
            plot_bgcolor="white",
            width=800,
            height=500,
        )
        return fig

    _output = plot_uranium_intercept()
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Uranium is indeed strongly associated with baseline radon levels in each county. The graph above shows the average relationship and its uncertainty: the baseline radon level in an average county as a function of uranium, as well as the 89% ETI of this radon level (dashed line and envelope). The blue points and orange bars represent the relationship between baseline radon and uranium, but now for each county. As you see, the uncertainty is bigger now, because it adds on top of the average uncertainty -- each county has its idyosyncracies after all.

    If we compare the county-intercepts for this model with those of the partial-pooling model without a county-level covariate:The standard errors on the intercepts are narrower than for the partial-pooling model without a county-level covariate.
    """)
    return


@app.cell(hide_code=True)
def _(varying_intercept_trace):
    # Plot forest for both models side by side
    _output = az.plot_forest(
        varying_intercept_trace, var_names=["alpha"], combined=True
    )
    _output
    return


@app.cell(hide_code=True)
def _(hierarchical_intercept_trace):
    _output = az.plot_forest(
        hierarchical_intercept_trace, var_names=["alpha"], combined=True
    )
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We see that the compatibility intervals are narrower for the model including the county-level covariate. This is expected, as the effect of a covariate is to reduce the variation in the outcome variable -- provided the covariate is of predictive value. More importantly, with this model we were able to squeeze even more information out of the data.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Correlations among levels

    In some instances, having predictors at multiple levels can reveal correlation between individual-level variables and group residuals. We can account for this by including the average of the individual predictors as a covariate in the model for the group intercept.

    $$\alpha_j = \gamma_0 + \gamma_1 u_j + \gamma_2 \bar{x} + \zeta_j$$

    These are broadly referred to as **_contextual effects_**.

    To add these effects to our model, let's create a new variable containing the mean of `floor` in each county and add that to our previous model:
    """)
    return


@app.cell(hide_code=True)
def _(srrs_mn_3):
    # Create new variable for mean of floor across counties
    avg_floor_data = (
        srrs_mn_3.group_by("county")
        .agg(pl.col("floor").mean())
        .select("floor")
        .to_numpy()
        .flatten()
    )
    return (avg_floor_data,)


@app.cell
def _(avg_floor_data, coords, county, floor_measure, log_radon, u):
    ctx_coords = {
        **coords,
        "coeff": ["intercept", "uranium", "avg_floor"],
    }

    @pmx.as_model(coords=ctx_coords)
    def build_contextual_effect():
        u_county = pmd.Data("u_county", u, dims="county")
        avg_floor_county = pmd.Data("avg_floor_county", avg_floor_data, dims="county")
        floor_idx = pmd.Data("floor_idx", floor_measure, dims="obs_id")
        county_idx = pmd.Data("county_idx", county, dims="obs_id")
        sigma_a = pmd.HalfCauchy("sigma_a", beta=5)
        gamma = pmd.Normal("gamma", mu=0.0, sigma=10, dims="coeff")
        mu_a = pmd.Deterministic(
            "mu_a",
            gamma[0]  # intercept
            + gamma[1] * u_county  # uranium
            + gamma[2] * avg_floor_county,  # avg_floor
            dims="county",
        )
        epsilon_a = pmd.Normal("epsilon_a", mu=0, sigma=1, dims="county")
        alpha = pmd.Deterministic("alpha", mu_a + sigma_a * epsilon_a, dims="county")
        beta = pmd.Normal("beta", mu=0.0, sigma=10)
        sigma_y = pmd.HalfNormal("sigma_y", sigma=2)
        y_hat = alpha[county_idx] + beta * floor_idx
        pmd.Normal(
            "y_like",
            mu=y_hat,
            sigma=sigma_y,
            observed=pmd.as_xtensor(log_radon, dims=("obs_id",)),
            dims="obs_id",
        )

    contextual_effect = build_contextual_effect()
    contextual_effect
    return (contextual_effect,)


@app.cell
def _(contextual_effect):
    with contextual_effect:
        contextual_effect_trace = pm.sample(tune=2000, random_seed=RANDOM_SEED)
    contextual_effect_trace
    return (contextual_effect_trace,)


@app.cell
def _(contextual_effect_trace):
    _output = az.summary(contextual_effect_trace, var_names="gamma", round_to=2)
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So we might infer from this that counties with higher proportions of houses without basements tend to have higher baseline levels of radon. This seems to be new, as up to this point we saw that `floor` was _negatively_ associated with radon levels. But remember this was at the household-level: radon tends to be higher in houses with basements. But at the county-level it seems that the less basements on average in the county, the more radon. So it's not that contradictory. What's more, the estimate for $\gamma_2$ is quite uncertain and overlaps with zero, so it's possible that the relationship is not that strong. And finally, let's note that $\gamma_2$ estimates something else than uranium's effect, as this is already taken into account by $\gamma_1$ -- it answers the question "once we know uranium level in the county, is there any value in learning about the proportion of houses without basements?".

    All of this is to say that we shouldn't interpret this causally: there is no credible mechanism by which a basement (or absence thereof) _causes_ radon emissions. More probably, our causal graph is missing something: a confounding variable, one that influences both basement construction and radon levels, is lurking somewhere in the dark... Perhaps is it the type of soil, which might influence what type of structures are built _and_ the level of radon? Maybe adding this to our model would help with causal inference.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model Comparison with LOO

    Gelman (2006) used cross-validation tests to check the prediction error of the unpooled, pooled, and partially-pooled models

    **root mean squared cross-validation prediction errors**:

    - unpooled = 0.86
    - pooled = 0.84
    - multilevel = 0.79

    Using cross-validation for a Bayesian model, fitting several copies of the model under different subsets of the data is computationally expensive. In the case of **Leave-One-Out Cross-Validation (LOO)** this involves repeatedly leaving out one observation, fitting the model to the remaining data, and evaluating prediction accuracy on the held-out point. The LOO estimate is:

    $$\text{elpd}_{\text{loo}} = \sum_{i=1}^N \log p(y_i | y_{-i})$$

    where $p(y_i | y_{-i})$ is the posterior predictive density for observation $i$ when trained on all data except $i$.

    However, Vehtari *et al.* (2016) introduced an efficient computation of LOO from the MCMC samples, which are corrected using **Pareto-smoothed importance sampling (PSIS)** to provide an estimate of point-wise out-of-sample prediction accuracy.

    This involves estimating the importance sampling LOO predictive distribution

    $$p(\tilde{y}_i | y_{-i}) \approx \frac{\sum_{s=1}^S w_i(\theta^{(s)}) p(\tilde{y}_i|\theta^{(s)})}{\sum_{s=1}^S w_i(\theta^{(s)})}$$

    where the importance weights are:

    $$w_i(\theta^{(s)}) = \frac{1}{p(y_i | \theta^{(s)})} \propto \frac{p(\theta^{(s)}|y_{-i})}{p(\theta^{(s)}|y)}$$

    The predictive distribution evaluated at the held-out point is then:

    $$p(y_i | y_{-i}) \approx \frac{1}{\frac{1}{S} \sum_{s=1}^S \frac{1}{p(y_i | \theta^{(s)})}}$$

    However, the posterior is likely to have a *smaller variance and thinner tails* than the LOO posteriors, so this approximation induces instability due to the fact that the importance ratios can have high or infinite variance.

    To deal with this instability, a generalized **Pareto distribution** fit to the upper tail of the distribution of the importance ratios can be used to construct a test for a finite importance ratio variance. If the test suggests the variance is infinite then importance sampling is halted.

    Let's compare our pooled, unpooled, and contextual effect models:
    """)
    return


@app.cell
def _(
    contextual_effect,
    contextual_effect_trace,
    pooled_model,
    pooled_trace,
    unpooled_model,
    unpooled_trace,
):
    # Compute log-likelihood for each model (required for LOO)
    with pooled_model:
        pm.compute_log_likelihood(pooled_trace)

    with unpooled_model:
        pm.compute_log_likelihood(unpooled_trace)

    with contextual_effect:
        pm.compute_log_likelihood(contextual_effect_trace)
    return


@app.cell
def _(contextual_effect_trace, pooled_trace, unpooled_trace):
    # Compare models using LOO
    model_comparison = az.compare(
        {
            "pooled": pooled_trace,
            "unpooled": unpooled_trace,
            "contextual_effect": contextual_effect_trace,
        },
    )
    model_comparison
    return (model_comparison,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The comparison table includes:

    - **rank**: Model ranking (0 = best)
    - **elpd_loo**: Expected log pointwise predictive density (higher is better)
    - **p_loo**: Effective number of parameters (complexity penalty)
    - **d_loo**: Difference from best model
    - **weight**: Model weights for ensemble predictions
    - **se/dse**: Standard errors for elpd and differences
    - **warning**: Flags potential PSIS reliability issues
    """)
    return


@app.cell(hide_code=True)
def _(model_comparison):
    _output = az.plot_compare(model_comparison)
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prediction

    There are two types of prediction that can be made in a multilevel model:

    1. a new individual within an existing group
    2. a new individual within a new group

    For example, if we wanted to make a prediction for a new house with no basement in St. Louis and Kanabec counties, we just need to sample from the radon model with the appropriate intercept.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That is,

    $$\tilde{y}_i \sim N(\alpha_{69} + \beta (x_i=1), \sigma_y^2)$$

    Because we judiciously set the county index and floor values as shared variables earlier, we can modify them directly to the desired values (69 and 1 respectively) and sample corresponding posterior predictions, without having to redefine and recompile our model. Using the model just above:
    """)
    return


@app.cell
def _(contextual_effect_trace):
    # Compute posterior predictive for specific houses in ST LOUIS and KANABEC
    ctx_post = contextual_effect_trace.posterior
    alpha_69 = ctx_post["alpha"].sel(county="ST LOUIS").values
    alpha_31 = ctx_post["alpha"].sel(county="KANABEC").values
    beta_post = ctx_post["beta"].values
    sigma_y_post = ctx_post["sigma_y"].values

    # Both houses on first floor (floor=1)
    mu_69 = alpha_69 + beta_post * 1
    mu_31 = alpha_31 + beta_post * 1

    rng = np.random.default_rng(RANDOM_SEED)
    y_pred_69 = mu_69 + sigma_y_post * rng.normal(size=mu_69.shape)
    y_pred_31 = mu_31 + sigma_y_post * rng.normal(size=mu_31.shape)

    stl_pred = y_pred_69, y_pred_31  # (chain, draw) arrays
    return (stl_pred,)


@app.cell(hide_code=True)
def _(stl_pred):
    pp_data = xr.Dataset(
        {
            "ST LOUIS": xr.DataArray(stl_pred[0].flatten(), dims="draw"),
            "KANABEC": xr.DataArray(stl_pred[1].flatten(), dims="draw"),
        }
    )
    _output = az.plot_dist(pp_data, sample_dims="draw")
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Prediction for a house within a new county is a little trickier. It is actually easier to create a new model to work with, **but use the trace from the original model for posterior predictive sampling**.

    How can this work?

    First, consider how posterior predictive sampling works in PyMC: samples are drawn not from the distributions themselves, but from the set of samples in the trace. Therefore, we can take the trace from the original model, and use it to sample posterior predictions from a new model that has the same variables.

    The variables in the new model need only have the same name as the original -- to reinforce this, I will use `pm.Flat` variables as placeholders in this example. The only variables we actually need are the ones that need to be resampled for a new county.

    We don't even need `Data` here; we can use raw data, since we are just creating this model to get posterior predictions for houses in this notional new county.
    """)
    return


@app.cell
def _(contextual_effect_trace):
    @pmx.as_model()
    def build_new_county():
        u_new = np.array([-0.2, 0.3])
        xbar = np.array([0.5, 0.8])
        floor_idx = np.array([1, 0])
        sigma_a = pm.Flat("sigma_a")
        gamma = pm.Flat("gamma", shape=3)
        beta = pm.Flat("beta")
        sigma_y = pm.Flat("sigma_y")
        mu_a_new = pm.Deterministic(
            "mu_a_new", gamma[0] + gamma[1] * u_new + gamma[2] * xbar
        )
        mu_new = pm.Normal("mu_new", mu_a_new, sigma_a)
        y_hat_new = mu_new + beta * floor_idx
        pm.Normal("y_new", mu=y_hat_new, sigma=sigma_y)

    new_county_model = build_new_county()
    with new_county_model:
        pp_new = pm.sample_posterior_predictive(
            contextual_effect_trace, var_names=["y_new"]
        )
    pp_new
    return (pp_new,)


@app.cell(hide_code=True)
def _(pp_new):
    _output = az.plot_dist(pp_new, group="posterior_predictive")
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Benefits of Multilevel Models

    - Accounting for natural hierarchical structure of observational data.

    - Estimation of coefficients for (under-represented) groups.

    - Incorporating individual- and group-level information when estimating group-level coefficients.

    - Allowing for variation among individual-level coefficients across groups.

    As an alternative approach to hierarchical modeling for this problem, radon levels can also be modeled spatially with Gaussian processes, which we cover in Session 5.2.
    """)
    return


if __name__ == "__main__":
    app.run()
