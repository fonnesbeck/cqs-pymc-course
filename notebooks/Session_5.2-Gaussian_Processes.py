import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


with app.setup:
    import marimo as mo
    import inspect
    import warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az
    import pytensor
    import pytensor.tensor as pt
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmap
    from patsy import dmatrix
    from pymc.gp.util import plot_gp_dist

    PYMC_BLUE = "#154A72"
    PYMC_GREEN = "#81C240"
    PYMC_LIGHT_BLUE = "#4A9EDE"
    PYMC_DARK_GREEN = "#40611F"
    RANDOM_SEED = 42
    RNG = np.random.default_rng(RANDOM_SEED)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    data_path = Path(__file__).parent / "data"

    def to_np(expr):
        """Compile a pytensor tensor expression to numpy."""
        return pytensor.function([], expr)()


@app.cell(hide_code=True)
def _():
    mo.md(
        """

        # Session 5.2: Gaussian Processes
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Bayesian non-parametric models are of interest because they provide a flexible and powerful framework for modeling complex data without making strong assumptions about the underlying distribution.

    Gaussian Processes, in particular, are a popular choice for Bayesian non-parametric modeling due to their ability to model complex functions and capture uncertainty. They can be used for regression, classification, and other tasks, and provide a probabilistic framework that allows for various sources of uncertainty to be accounted for. These models are particularly useful in scenarios where the data is sparse, noisy, or exhibits complex patterns that cannot be easily captured by traditional parametric models.

    Use of the term "non-parametric" in the context of Bayesian analysis is something of a misnomer. This is because the first and fundamental step in Bayesian modeling is to specify a *full probability model* for the problem at hand. It is rather difficult to explicitly state a full probability model without the use of probability functions, which are parametric. Bayesian non-parametric methods do not imply that there are no parameters, but rather that the number of parameters grows with the size of the dataset. Often, Bayesian non-parametric models are *infinitely* parametric.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Spline Models

    Before we dive into Gaussian processes, it's worth mentioning another popular class of non-parametric models: spline models. Spline models are a class of regression models that use **piecewise polynomials** to fit the data. They are particularly useful for modeling complex functions and capturing non-linear relationships.

    A spline fit is effectively a sum of multiple individual curves (piecewise polynomials), each fit to a different section of $x$, that are tied together at their boundaries, often called *knots*.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Swing Decisions Data

    To motivate the use of non-parametric models, we will use a subset of the batter grade data, specifically swing decisions from the 2023 season. We will look at the relationship between swing decision scores and age, within a snapshot of a single season (while studiously ignoring the selection bias associated with censored data).
    """)
    return


@app.cell(hide_code=True)
def _():
    swing_decisions = pd.read_csv(
        data_path / "batter_grades_2023.csv", index_col=0
    ).query('(throws=="R") & (n_pa>100)')[
        ["batter_id", "batter", "age", "swing_decision"]
    ]
    swing_decisions.head()
    return (swing_decisions,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If we visualize the data, it is clear that there is a lot of annual variation, but some evidence for a non-linear trend of swing decision grade by batter age.
    """)
    return


@app.cell(hide_code=True)
def _(swing_decisions):
    swing_decisions.plot.scatter(
        x="age",
        y="swing_decision",
        s=10,
        title="Swing Decisions 2023",
        alpha=0.3,
        legend=False,
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Spline model

    We will fit the following model.

    $$G \sim \mathcal{N}(\mu, \sigma)$$
    $$\quad \mu = a + Bw$$
    $$\qquad a \sim \mathcal{N}(0, 5)$$
    $$\qquad w \sim \mathcal{N}(0, 3)$$
    $$\quad \sigma \sim {Exp}(1)$$

    The batter grade $G$ will be modeled as a normal distribution with mean $\mu$ and standard deviation $\sigma$. In turn, the mean will be a linear model composed of a y-intercept $a$ and a spline defined by the basis $B$ multiplied by the model parameter $w$ with a variable for each region of the basis. Both have relatively weak normal priors.

    ### Prepare the spline

    The spline will have 7 *knots*, splitting the year into 8 sections (including the regions covering the years before and after those in which we have data). The knots are the boundaries of the spline, the name owing to how the individual lines will be tied together at these boundaries to make a continuous and smooth curve. The knots will be unevenly spaced over the years such that each region will have the same proportion of data.
    """)
    return


@app.cell(hide_code=True)
def _(swing_decisions):
    num_knots = 7
    knot_list = np.quantile(swing_decisions.age, np.linspace(0, 1, num_knots))
    knot_list
    return (knot_list,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below is a plot of the locations of the knots over the data.
    """)
    return


@app.cell(hide_code=True)
def _(knot_list, swing_decisions):
    knot_ax = swing_decisions.plot.scatter(
        x="age",
        y="swing_decision",
        s=10,
        title="Swing Decisions 2023",
        alpha=0.3,
        legend=False,
    )
    for _knot in knot_list:
        knot_ax.axvline(_knot, color="grey", alpha=0.4)
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can use `patsy` to create the matrix $B$ that will be the b-spline basis for the regression. The degree is set to 3 to create a cubic b-spline.
    """)
    return


@app.cell(hide_code=True)
def _(knot_list, swing_decisions):
    age = swing_decisions.age.unique()
    age.sort()

    B = dmatrix(
        "bs(age, knots=knots, degree=3, include_intercept=True) - 1",
        {"age": age, "knots": knot_list[1:-1]},
    )
    B
    return B, age


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The b-spline basis is plotted below, showing the *domain* of each piece of the spline. The height of each curve indicates how influential the corresponding model covariate (one per spline region) will be on the model's inference of that region. The overlapping regions represent the knots, showing how the smooth transition from one region to the next is formed.
    """)
    return


@app.cell(hide_code=True)
def _(B, age):
    spline_basis_df = (
        pd.DataFrame(B)
        .assign(age=age)
        .melt("age", var_name="spline_i", value_name="swing_decision")
    )

    _colors = plt.cm.magma(np.linspace(0, 0.80, len(spline_basis_df.spline_i.unique())))

    plt.figure()
    for _i, _c in enumerate(_colors):
        _subset = spline_basis_df.query(f"spline_i == {_i}")
        _subset.plot("age", "swing_decision", c=_c, ax=plt.gca(), label=_i)
    plt.legend(title="Spline Index", loc="upper center", fontsize=8, ncol=6)
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the model

    The model is built using PyMC. A graphical diagram shows the organization of the model parameters.
    """)
    return


@app.cell
def _(B, swing_decisions):
    def build_spline_model():
        coords = {
            "splines": np.arange(B.shape[1]),
            "obs": np.arange(swing_decisions.shape[0]),
        }
        with pm.Model(coords=coords) as spline_model:
            player_age_ind = pm.Data(
                "player_age_ind",
                swing_decisions.age.values - swing_decisions.age.min(),
            )
            swing_decision_obs = pm.Data(
                "swing_decision_obs", swing_decisions.swing_decision.values
            )

            a = pm.Normal("a", 0, 5)
            w = pm.Normal("w", mu=0, sigma=3, size=B.shape[1], dims="splines")
            mu = pm.Deterministic(
                "mu",
                a + pm.math.dot(np.asarray(B, order="F"), w.T)[player_age_ind],
            )
            sigma = pm.Exponential("sigma", 1)
            pm.Normal("D", mu=mu, sigma=sigma, observed=swing_decision_obs, dims="obs")
        return spline_model

    spline_model = build_spline_model()
    return (spline_model,)


@app.cell(hide_code=True)
def _(spline_model):
    spline_model
    return


@app.cell
def _(spline_model):
    with spline_model:
        spline_trace = pm.sample(
            draws=1000,
            tune=1000,
            random_seed=RANDOM_SEED,
            chains=2,
        )
    return (spline_trace,)


@app.cell
def _(spline_model, spline_trace):
    with spline_model:
        pm.sample_posterior_predictive(spline_trace, extend_inferencedata=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Parameter Estimates

    Below is a table summarizing the posterior distributions of the model parameters. The posteriors of $a$ and $\sigma$ are quite narrow while those for $w$ are wider. This is likely because all of the data points are used to estimate $a$ and $\sigma$ whereas only a subset are used for each value of $w$. (It could be interesting to model these hierarchically, allowing for the sharing of information and adding regularization across the spline.) The effective sample size and $\widehat{R}$ values all look good, indicating that the model has converged and sampled well from the posterior distribution.
    """)
    return


@app.cell(hide_code=True)
def _(spline_trace):
    az.summary(spline_trace, var_names=["a", "w", "sigma"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Another visualization of the fit spline values is to plot them multiplied against the basis matrix. The knot boundaries are shown as vertical lines again, but now the spline basis is multiplied against the values of $w$ (represented as the rainbow-colored curves). The dot product of $B$ and $w$ (the actual computation in the linear model) is shown in black.
    """)
    return


@app.cell(hide_code=True)
def _(B, age, knot_list, spline_trace):
    wp = spline_trace.posterior["w"].mean(("chain", "draw")).values

    spline_weighted_df = (
        pd.DataFrame(B * wp.T)
        .assign(age=age)
        .melt("age", var_name="spline_i", value_name="swing_decision")
    )
    spline_merged_df = (
        pd.DataFrame(np.dot(B, wp.T))
        .assign(age=age)
        .melt("age", var_name="spline_i", value_name="swing_decision")
    )

    _colors = plt.cm.rainbow(
        np.linspace(0, 1, len(spline_weighted_df.spline_i.unique()))
    )
    plt.figure()
    for _i, _c in enumerate(_colors):
        _subset = spline_weighted_df.query(f"spline_i == {_i}")
        _subset.plot("age", "swing_decision", c=_c, ax=plt.gca(), label=_i)
    spline_merged_df.groupby("age").mean().reset_index().plot(
        "age", "swing_decision", c="black", lw=2, ax=plt.gca()
    )
    plt.legend(title="Spline Index", loc="upper center", fontsize=8, ncol=6)
    for _knot in knot_list:
        plt.gca().axvline(_knot, color="grey", alpha=0.4)
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model predictions

    Lastly, we can generate predictions from the model using the posterior predictive distribution via `sample_posterior_predictive`. There is no additional model fitting here; we are just sampling from the posterior distribution of the model parameters to generate predictions for the outcome variable under the model.
    """)
    return


@app.cell
def _(age, spline_model, spline_trace):
    with spline_model:
        pm.set_data({"player_age_ind": age - age.min()})
        spline_post_pred = pm.sample_posterior_predictive(
            spline_trace, var_names=["mu"], random_seed=RANDOM_SEED
        )
    spline_post_pred
    return (spline_post_pred,)


@app.cell(hide_code=True)
def _(age, knot_list, spline_post_pred, swing_decisions):
    spline_pred_summary = az.summary(
        spline_post_pred,
        group="posterior_predictive",
        ci_kind="hdi",
        ci_prob=0.94,
        round_to="none",
    )
    spline_pred_summary["age"] = age

    pred_ax = swing_decisions.plot.scatter(
        "age",
        "swing_decision",
        color="cornflowerblue",
        s=10,
        title="Swing decision data with posterior predictions",
        ylabel="Swing Decision Grade",
    )
    for _knot in knot_list:
        pred_ax.axvline(_knot, color="grey", alpha=0.4)

    spline_pred_summary.plot("age", "mean", ax=pred_ax, lw=3, color="firebrick")
    pred_ax.fill_between(
        spline_pred_summary.age,
        spline_pred_summary["hdi94_lb"],
        spline_pred_summary["hdi94_ub"],
        color="firebrick",
        alpha=0.4,
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Gaussian Processes

    Splines are relatively straightforward to implement and flexible for modeling non-linear trends, but they have some limitations. For example, the number of knots must be specified in advance, and model output can be sensitive to the choice (and number) of knots. This requires the use of cross-validation or other methods to select the optimal set of knots, which can be computationally expensive. Additionally, splines are not well-suited for modeling more complex functions, especially with interactions between variables.

    Gaussian Processes (GPs) are a more powerful alternative that can model arbitrarily complex functions without the need for specifying the number of parameters *a priori*. GPs are a powerful tool for modeling non-linear trends and capturing complex patterns in the data. They can be used for regression, classification, and other tasks, and provide a probabilistic framework that allows for uncertainty quantification.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Building models with Gaussians

    What if we chose to use **normal distributions** to model our data?

    $$p(x \mid \pi, \Sigma) = (2\pi)^{-k/2}|\Sigma|^{-1/2} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime}\Sigma^{-1}(x-\mu) \right\}$$

    There would not seem to be an advantage to doing this, because normal distributions are not particularly flexible distributions in and of themselves. However, adopting a set of Gaussians (a multivariate normal vector) confers a number of advantages.

    First, the **marginal distribution** of any subset of elements from a multivariate normal distribution is also normal:

    $$p(x,y) = \mathcal{N}\left(\left[\begin{array}{c}{\mu_x} \\ {\mu_y} \end{array}\right], \left[\begin{array}{cc} {\Sigma_x} & {\Sigma_{xy}} \\ {\Sigma_{xy}^T} & {\Sigma_y} \end{array}\right]\right)$$

    $$p(x) = \int p(x,y)\, dy = \mathcal{N}(\mu_x, \Sigma_x)$$

    Also, **conditional distributions** of a subset of a multivariate normal distribution (conditional on the remaining elements) are normal too:

    $$p(x|y) = \mathcal{N}(\underbrace{\mu_x + \Sigma_{xy}\Sigma_y^{-1}(y-\mu_y)}_{\text{conditional mean}}, \, \underbrace{\Sigma_x-\Sigma_{xy}\Sigma_y^{-1}\Sigma_{xy}^T}_{\text{conditional covariance}})$$

    A Gaussian process generalizes the multivariate normal to **infinite dimension**. It is defined as an infinite collection of random variables, any finite subset of which have a Gaussian distribution. Thus, the marginalization property is explicit in its definition. Another way of thinking about an infinite vector is as a *function*.

    When we write a function that takes continuous values as inputs, we are essentially specifying an infinite vector that only returns values (indexed by the inputs) when the function is called upon to do so. By the same token, this notion of an infinite-dimensional Gaussian as a function allows us to work with them computationally: we are never required to store all the elements of the Gaussian process, only to calculate them on demand.

    So, we can describe a Gaussian process as a ***distribution over functions***. Just as a multivariate normal distribution is completely specified by a mean vector and covariance matrix, a GP is fully specified by a **mean function** and a **covariance function**:

    $$p(x) \sim \mathcal{GP}(m(x), k(x,x^{\prime}))$$

    It is the marginalization property that makes working with a Gaussian process feasible: we can marginalize over the infinitely-many variables that we are not interested in, or have not observed.

    For example, one specification of a GP might be as follows:

    $$\begin{aligned}
    m(x) &= 0 \\
    k(x,x^{\prime}) &= \theta_1\exp\left(-\frac{\theta_2}{2}(x-x^{\prime})^2\right)
    \end{aligned}$$

    Here, the covariance function is an **exponential quadratic**, for which values of $x$ and $x^{\prime}$ that are close together result in values of $k$ closer to 1 and those that are far apart return values closer to zero.
    """)
    return


@app.cell(hide_code=True)
def _():
    def exponential_cov(x, y, scale, length_scale):
        """Exponential quadratic covariance between two arrays."""
        return scale * np.exp(-0.5 * np.subtract.outer(x, y) ** 2 / length_scale)

    return (exponential_cov,)


@app.cell(hide_code=True)
def _(exponential_cov):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _xrange = np.linspace(0, 5)
    _ax1.plot(_xrange, exponential_cov(0, _xrange, 1, 1))
    _ax1.set_xlabel("x")
    _ax1.set_ylabel("cov(0, x)")

    _z = np.array([exponential_cov(_xrange, _xprime, 1, 1) for _xprime in _xrange])
    _ax2.imshow(_z, cmap="inferno", interpolation="none", extent=(0, 5, 5, 0))
    _ax2.set_xlabel("x")
    _ax2.set_ylabel("x")

    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The mean function just returns a constant value of zero.

    It may seem odd to simply adopt the zero function to represent the mean function of the Gaussian process. Surely we can do better than that! It turns out that most of the learning in the GP involves the covariance function and its parameters, so very little is gained in specifying a complicated mean function (except where data are very sparse).

    For a finite number of points, the GP becomes a multivariate normal, with the mean and covariance as the mean function and covariance function evaluated at those points.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sampling from a Gaussian Process Prior

    To make this notion of a "distribution over functions" more concrete, let's demonstrate how we obtain realizations from a Gaussian process, which result in an evaluation of a function over a set of points. All we will do here is sample from the *prior* Gaussian process, before any data have been introduced. What we need first is our covariance function, which will be the squared exponential, and a function to evaluate the covariance at given points (resulting in a covariance matrix).

    We are going to generate realizations sequentially, point by point, using the lovely conditioning property of multivariate Gaussian distributions. Here is that conditional:

    $$p(y^*| x^*, y, x) = \mathcal{N}\bigl(\Sigma_{x^*x}\Sigma_y^{-1}y,\,
    \Sigma_{x^*}-\Sigma_{x^*x}\Sigma_y^{-1}\Sigma_{x^*x}^T\bigr)$$
    """)
    return


@app.cell(hide_code=True)
def _(exponential_cov):
    def gp_conditional_demo(x_new, x, y, scale, length_scale):
        B_block = exponential_cov(x_new, x, scale, length_scale)
        C_block = exponential_cov(x, x, scale, length_scale)
        A_block = exponential_cov(x_new, x_new, scale, length_scale)
        mu_block = np.linalg.inv(C_block).dot(B_block.T).T.dot(y)
        sigma_block = A_block - B_block.dot(np.linalg.inv(C_block).dot(B_block.T))
        return mu_block.squeeze(), sigma_block.squeeze()

    def gp_predict_demo(x, data, kernel, scale, length_scale, sigma, t):
        k = [kernel(x, y, scale, length_scale) for y in data]
        Sinv = np.linalg.inv(sigma)
        y_pred = np.dot(k, Sinv).dot(t)
        sigma_new = kernel(x, x, scale, length_scale) - np.dot(k, Sinv).dot(k) + 1e-8
        return y_pred, sigma_new

    return gp_conditional_demo, gp_predict_demo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We will start with a Gaussian process prior with hyperparameters $\theta_0=1$, $\theta_1=0.1$. We will also assume a zero function as the mean, so we can plot a band that represents one standard deviation from the mean.
    """)
    return


@app.cell(hide_code=True)
def _(exponential_cov):
    gp_scale, gp_length_scale = 1, 0.1
    sigma_0 = exponential_cov(0, 0, gp_scale, gp_length_scale)
    xpts = np.arange(-3, 3, step=0.01)
    plt.errorbar(xpts, np.zeros(len(xpts)), yerr=sigma_0, capsize=0)
    plt.ylim(-3, 3)
    plt.gca()
    return gp_length_scale, gp_scale


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The demonstration below builds up a realization sequentially: we start by sampling a single point at $x=1$ from an unconditional Gaussian, then conditionally sample additional points one at a time. As the density of points grows, the result approaches one realization (function) from the prior GP. The error bars show the conditional standard deviation at each point on the grid.
    """)
    return


@app.cell(hide_code=True)
def _(exponential_cov, gp_conditional_demo, gp_length_scale, gp_predict_demo, gp_scale):
    def sample_gp_sequentially(seed=42):
        rng = np.random.default_rng(seed)
        x_pred_grid = np.linspace(-3, 3, 1000)

        x = [1.0]
        sigma_init = exponential_cov(0, 0, gp_scale, gp_length_scale)
        y = [rng.normal(scale=sigma_init)]

        m_, s_ = gp_conditional_demo([-0.7], x, y, gp_scale, gp_length_scale)
        y.append(rng.normal(m_, s_))
        x.append(-0.7)

        x_more = [-2.1, -1.5, 0.3, 1.8, 2.5]
        mu_more, s_more = gp_conditional_demo(x_more, x, y, gp_scale, gp_length_scale)
        y_more = rng.multivariate_normal(mu_more, s_more)
        x += x_more
        y += y_more.tolist()

        sigma_curr = exponential_cov(x, x, gp_scale, gp_length_scale)
        preds = [
            gp_predict_demo(
                i, x, exponential_cov, gp_scale, gp_length_scale, sigma_curr, y
            )
            for i in x_pred_grid
        ]
        y_pred, sigmas = np.transpose(preds)
        return x, y, x_pred_grid, y_pred, sigmas

    x_seq, y_seq, x_grid_seq, y_pred_seq, sigmas_seq = sample_gp_sequentially()

    plt.figure(figsize=(10, 5))
    plt.errorbar(x_grid_seq, y_pred_seq, yerr=sigmas_seq, capsize=0)
    plt.plot(x_seq, y_seq, "ro")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title("One realization of the GP prior, built up sequentially")
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Of course, sampling sequentially is just a heuristic to demonstrate how the covariance structure works. We can just as easily sample several points at once from the joint multivariate Gaussian. As the density of points becomes high, the result approaches one realization (function) from the prior GP. If we run this many times, we get an idea of the types of functions that the GP prior can generate.

    This example is trivial because it is simply a random function drawn from the prior. What we are really interested in is *learning* about an underlying function from information residing in our data. In a parametric setting, we either specify a **likelihood**, which we then maximize with respect to the parameters, or a **full probability model**, for which we calculate the posterior in a Bayesian context. Though the integrals associated with posterior distributions are typically intractable for parametric models, they do not pose a problem with Gaussian processes, as we will see.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Gaussian processes regression

    The following **simulated data** clearly shows some type of non-linear process, corrupted by a certain amount of observation or measurement error, so it should be a reasonable task for a Gaussian process approach.
    """)
    return


@app.cell
def _():
    def simulate_gp_data(seed=1, n=100, l_true=1.0, eta_true=3.0, sigma_true=2.0):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 10, n)
        X = x[:, None]

        cov_func = eta_true**2 * pm.gp.cov.Matern52(1, l_true)
        mean_func = pm.gp.mean.Zero()

        f_true = rng.multivariate_normal(
            to_np(mean_func(X)),
            to_np(cov_func(X)) + 1e-8 * np.eye(n),
            1,
        ).flatten()
        y = f_true + sigma_true * rng.standard_normal(n)
        return x, X, f_true, y

    sim_x, sim_X, sim_f_true, sim_y = simulate_gp_data()
    sim_l_true, sim_eta_true, sim_sigma_true = 1.0, 3.0, 2.0

    _fig = plt.figure(figsize=(12, 5))
    _ax = _fig.gca()
    _ax.plot(sim_X, sim_f_true, "dodgerblue", lw=3, label="True f")
    _ax.plot(sim_X, sim_y, "ok", ms=3, alpha=0.5, label="Data")
    _ax.set_xlabel("X")
    _ax.set_ylabel("y")
    plt.legend()
    plt.gca()
    return sim_eta_true, sim_f_true, sim_l_true, sim_sigma_true, sim_x, sim_y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Marginal Likelihood Implementation

    The simplest case of GP regression is the marginal likelihood model, used in scenarios where the observed data are the sum of a GP and Gaussian noise. Its implementation in PyMC, `gp.Marginal`, has a `marginal_likelihood` method, a `conditional` method, and a `predict` method just like the ones we implemented by hand in the previous section. Given a mean and covariance function, the function $f(x)$ is modeled as

    $$
    f(x) \sim \mathcal{GP}(m(x),\, k(x, x')) \,.
    $$

    The observations $y$ are the unknown function plus noise

    $$
    \begin{aligned}
    \epsilon &\sim N(0, \Sigma) \\
    y &= f(x) + \epsilon
    \end{aligned}
    $$

    The resulting posterior distribution is

    $$
    \mathcal{GP}(m_*(x),\, k_*(x, x'))
    $$

    where

    $$
    \begin{aligned}
    m_*(x) & = k\left(x, \mathbf{x}\right)\left[k\left(\mathbf{x}, \mathbf{x}\right)+\sigma_n^2 \mathbb{I}\right]^{-1} \mathbf{y} \\
    k_*(x, x') & = k\left(x, x'\right)-k\left(x, \mathbf{x}\right)\left[k(\mathbf{x}, \mathbf{x})+\sigma_n^2 \mathbb{I}\right]^{-1} k\left(\mathbf{x}, x\right)
    \end{aligned}
    $$

    ### The Marginal Likelihood

    The marginal likelihood is the normalizing constant for the posterior distribution

    $$p(y|f, X) = \frac{p(y|f,X) \cdot p(f|X)}{p(y|X)}$$

    In other words, it is the integral of the product of the likelihood and prior.

    $$p(y|X) = \int_f p(y|f,X)\, p(f|X)\, df$$

    where for Gaussian processes, we are marginalizing over function values $f$ (instead of parameters $\theta$).

    **GP prior**:

    $$\log p(f|X) = - \frac{k}{2}\log 2\pi - \frac{1}{2}\log|K| -\frac{1}{2}f^TK^{-1}f $$

    **Gaussian likelihood**:

    $$\log p(y|f,X) = - \frac{k}{2}\log 2\pi - \frac{1}{2}\log|\sigma^2 I| -\frac{1}{2}(y-f)^T(\sigma^2 I)^{-1}(y-f) $$

    **Marginal likelihood**:

    $$\log p(y|X) = - \frac{k}{2}\log 2\pi - \frac{1}{2}\log|K + \sigma^2 I| - \frac{1}{2}y^T(K+\sigma^2 I)^{-1}y $$

    Notice that the marginal likelihood includes both a data fit term $- \frac{1}{2}y^T(K+\sigma^2 I)^{-1}y$ and a parameter penalty term $\frac{1}{2}\log|K + \sigma^2 I|$. Hence, the marginal likelihood can help us select an appropriate covariance function, based on its fit to the dataset at hand.

    ### Choosing parameters

    This is relevant because we have to make choices regarding the parameters of our Gaussian process; they were chosen arbitrarily for the random functions we demonstrated above.

    For example, in the squared exponential covariance function, we must choose two parameters:

    $$k(x,x^{\prime}) = \theta_1\exp\left(-\frac{\theta_2}{2}(x-x^{\prime})^2\right)$$

    The first parameter $\theta_1$ is a scale parameter, which allows the function to yield values outside of the unit interval. The second parameter $\theta_2$ is a length scale parameter that determines the degree of covariance between $x$ and $x^{\prime}$; smaller values will tend to smooth the function relative to larger values.

    We can use the **marginal likelihood** to select appropriate values for these parameters, since it trades off model fit with model complexity. Thus, an optimization procedure can be used to select values for $\theta$ that maximize the marginal likelihood.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Covariance functions

    The behavior of individual realizations from the GP is governed by the covariance function. This function controls both the degree of *shrinkage* to the mean function and the *smoothness* of functions sampled from the GP.

    PyMC includes a library of covariance functions to choose from. A flexible choice to start with is the Matèrn covariance.

    $$k_{M}(x) = \frac{\sigma^2}{\Gamma(\nu)2^{\nu-1}} \left(\frac{\sqrt{2 \nu} x}{l}\right)^{\nu} K_{\nu}\left(\frac{\sqrt{2 \nu} x}{l}\right)$$

    where $\Gamma$ is the gamma function and $K$ is a modified Bessel function. The form of covariance matrices sampled from this function is governed by three parameters, each of which controls a property of the covariance.

    - **amplitude or scale** ($\sigma$) controls the scaling of the output along the y-axis. This parameter is just a scalar multiplier, and is therefore usually left out of implementations of the Matèrn function (i.e. set to one).
    - **lengthscale** ($l$) complements the amplitude by scaling realizations on the x-axis. Larger values make points appear closer together.
    - **roughness** ($\nu$) controls the sharpness of ridges in the covariance function, which ultimately affects the roughness (smoothness) of realizations.

    Though in general all the parameters are non-negative real-valued, when $\nu = p + 1/2$ for integer-valued $p$, the function can be expressed partly as a polynomial function of order $p$ and generates realizations that are $p$-times differentiable, so values $\nu \in \{3/2, 5/2\}$ are extremely common.

    To give you an idea about the variety of forms of covariance functions, here is a small selection of available ones.
    """)
    return


@app.cell(hide_code=True)
def _():
    def plot_cov(X, K, stationary=True):
        K = K + 1e-8 * np.eye(X.shape[0])
        x = X.flatten()

        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(121)
        m = ax1.imshow(
            K,
            cmap="inferno",
            interpolation="none",
            extent=(np.min(X), np.max(X), np.max(X), np.min(X)),
        )
        plt.colorbar(m)
        ax1.set_title("Covariance Matrix")
        ax1.set_xlabel("X")
        ax1.set_ylabel("X")

        ax2 = fig.add_subplot(122)
        if not stationary:
            ax2.plot(x, np.diag(K), "k", lw=2, alpha=0.8)
            ax2.set_title("The Diagonal of K")
            ax2.set_ylabel("k(x,x)")
            ax2.set_ylim(0, 125)
        else:
            ax2.plot(x, K[:, 0], "k", lw=2, alpha=0.8)
            ax2.set_title("K as a function of x - x'")
            ax2.set_ylabel("k(x,x')")
            if np.nanmin(K) < -1e-8:
                ax2.set_ylim(-25, 25)
            else:
                ax2.set_ylim(0, 25)
        ax2.set_xlabel("X")

        fig2 = plt.figure(figsize=(14, 4))
        ax = fig2.add_subplot(111)
        rng = np.random.default_rng(2024)
        samples = rng.multivariate_normal(np.zeros(K.shape[0]), K, 5).T
        for i in range(samples.shape[1]):
            ax.plot(x, samples[:, i], color=cmap.inferno(i * 0.2), lw=2)
        ax.set_title("Samples from GP Prior")
        ax.set_xlabel("X")
        ax.set_ylim(-20, 20)
        return fig, fig2

    return (plot_cov,)


@app.cell(hide_code=True)
def _():
    X_grid = np.linspace(0, 2, 200)[:, None]
    return (X_grid,)


@app.cell(hide_code=True)
def _():
    gp_kernel_options = ["ExpQuad", "Matern 1/2", "Cosine", "Linear"]

    expquad_lengthscale_slider = mo.ui.slider(
        0.05, 2.0, value=0.30, step=0.05, label="Lengthscale"
    )
    expquad_amplitude_slider = mo.ui.slider(
        0.1, 5.0, value=1.0, step=0.1, label="Amplitude"
    )

    matern12_lengthscale_slider = mo.ui.slider(
        0.05, 2.0, value=0.30, step=0.05, label="Lengthscale"
    )
    matern12_amplitude_slider = mo.ui.slider(
        0.1, 5.0, value=1.0, step=0.1, label="Amplitude"
    )

    cosine_lengthscale_slider = mo.ui.slider(
        0.05, 2.0, value=0.30, step=0.05, label="Period scale"
    )
    cosine_amplitude_slider = mo.ui.slider(
        0.1, 5.0, value=1.0, step=0.1, label="Amplitude"
    )

    linear_offset_slider = mo.ui.slider(
        -2.0, 2.0, value=1.0, step=0.1, label="Offset c"
    )
    linear_amplitude_slider = mo.ui.slider(
        0.1, 5.0, value=1.0, step=0.1, label="Amplitude"
    )

    product_left_kernel = mo.ui.dropdown(
        gp_kernel_options, value="Cosine", label="Left kernel"
    )
    product_right_kernel = mo.ui.dropdown(
        gp_kernel_options, value="Linear", label="Right kernel"
    )

    sum_left_kernel = mo.ui.dropdown(
        gp_kernel_options, value="Cosine", label="Left kernel"
    )
    sum_right_kernel = mo.ui.dropdown(
        gp_kernel_options, value="Linear", label="Right kernel"
    )
    return (
        cosine_amplitude_slider,
        cosine_lengthscale_slider,
        expquad_amplitude_slider,
        expquad_lengthscale_slider,
        linear_amplitude_slider,
        linear_offset_slider,
        matern12_amplitude_slider,
        matern12_lengthscale_slider,
        product_left_kernel,
        product_right_kernel,
        sum_left_kernel,
        sum_right_kernel,
    )


@app.cell
def _():
    def kernel_covariance(kernel_name, amplitude, lengthscale, offset):
        if kernel_name == "ExpQuad":
            return amplitude**2 * pm.gp.cov.ExpQuad(1, lengthscale), True
        if kernel_name == "Matern 1/2":
            return amplitude**2 * pm.gp.cov.Matern12(1, lengthscale), True
        if kernel_name == "Cosine":
            return amplitude**2 * pm.gp.cov.Cosine(1, lengthscale), True
        if kernel_name == "Linear":
            return amplitude**2 * pm.gp.cov.Linear(1, c=offset), False
        raise ValueError(f"Unknown kernel: {kernel_name}")

    def default_kernel_params(kernel_name):
        if kernel_name == "Linear":
            return 1.0, 1.0, 1.0
        return 1.0, 0.3, 0.0

    return default_kernel_params, kernel_covariance


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exponential quadratic covariance

    $$
    k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{2 \ell^2} \right]
    $$

    The exponential quadratic (or squared exponential, or radial basis function) kernel has become the *de facto* default kernel used in a variety of settings, including GPs. It only has two hyperparameters, yet can be used to approximate almost any non-linear function. You can integrate it against most functions and every function in its prior has infinitely many derivatives.
    """)
    return


@app.cell(hide_code=True)
def _(expquad_amplitude_slider, expquad_lengthscale_slider):
    mo.hstack([expquad_lengthscale_slider, expquad_amplitude_slider], gap=2)
    return


@app.cell(hide_code=True)
def _(
    X_grid,
    expquad_amplitude_slider,
    expquad_lengthscale_slider,
    kernel_covariance,
    plot_cov,
):
    _cov, _stationary = kernel_covariance(
        "ExpQuad",
        expquad_amplitude_slider.value,
        expquad_lengthscale_slider.value,
        0.0,
    )
    plot_cov(X_grid, to_np(_cov(X_grid)), stationary=_stationary)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Matern $\nu=1/2$ covariance

    If you need a "wiggly" function, but not as wiggly as the $\nu=5/2$ Matern, the $\nu=1/2$ Matern (also known as the exponential or Ornstein-Uhlenbeck kernel) might be a good choice.

    $$
    k(x, x') = \mathrm{exp}\left[ - \frac{|x - x'|}{\ell} \right]
    $$
    """)
    return


@app.cell(hide_code=True)
def _(matern12_amplitude_slider, matern12_lengthscale_slider):
    mo.hstack([matern12_lengthscale_slider, matern12_amplitude_slider], gap=2)
    return


@app.cell(hide_code=True)
def _(
    X_grid,
    kernel_covariance,
    matern12_amplitude_slider,
    matern12_lengthscale_slider,
    plot_cov,
):
    _cov, _stationary = kernel_covariance(
        "Matern 1/2",
        matern12_amplitude_slider.value,
        matern12_lengthscale_slider.value,
        0.0,
    )
    plot_cov(X_grid, to_np(_cov(X_grid)), stationary=_stationary)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Cosine covariance

    We sometimes need to model periodic behavior (which, incidentally, can be done without a specialized kernel), in which case the cosine kernel is convenient.

    $$
    k(x, x') = \mathrm{cos}\left( 2 \pi \frac{||x - x'||}{ \ell^2} \right)
    $$
    """)
    return


@app.cell(hide_code=True)
def _(cosine_amplitude_slider, cosine_lengthscale_slider):
    mo.hstack([cosine_lengthscale_slider, cosine_amplitude_slider], gap=2)
    return


@app.cell(hide_code=True)
def _(
    X_grid,
    cosine_amplitude_slider,
    cosine_lengthscale_slider,
    kernel_covariance,
    plot_cov,
):
    _cov, _stationary = kernel_covariance(
        "Cosine",
        cosine_amplitude_slider.value,
        cosine_lengthscale_slider.value,
        0.0,
    )
    plot_cov(X_grid, to_np(_cov(X_grid)), stationary=_stationary)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Linear

    Covariance functions can be extremely simple. For example, a linear covariance function is simply the inner product of the input vectors, optionally offset by a constant:

    $$
    k(x, x') = (x - c)(x' - c)
    $$

    The linear kernel is often used in combination with other kernels in order to achieve some desired properties.

    Unlike the other kernels we have seen, the linear kernel is **non-stationary**. A stationary covariance function depends only on the relative values of the inputs, rather than their absolute location.
    """)
    return


@app.cell(hide_code=True)
def _(linear_amplitude_slider, linear_offset_slider):
    mo.hstack([linear_offset_slider, linear_amplitude_slider], gap=2)
    return


@app.cell(hide_code=True)
def _(
    X_grid, kernel_covariance, linear_amplitude_slider, linear_offset_slider, plot_cov
):
    _cov, _stationary = kernel_covariance(
        "Linear",
        linear_amplitude_slider.value,
        1.0,
        linear_offset_slider.value,
    )
    plot_cov(X_grid, to_np(_cov(X_grid)), stationary=_stationary)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Combining Kernels

    Modeling with a single kernel is useful if your data is all the same type, but what if you have multiple types of features but you still want to model them together?

    It turns out that we can build a kernel over different datatypes by multiplying or adding kernels together. Heuristically, multiplying two kernels acts like an `AND` operation. To that end, the kernel resulting from the product of two covariance functions will have a high value when both of the constituent kernels also have a high value.
    """)
    return


@app.cell(hide_code=True)
def _(product_left_kernel, product_right_kernel):
    mo.hstack([product_left_kernel, product_right_kernel], gap=1)
    return


@app.cell(hide_code=True)
def _(
    X_grid,
    default_kernel_params,
    kernel_covariance,
    plot_cov,
    product_left_kernel,
    product_right_kernel,
):
    _left_cov, _left_stationary = kernel_covariance(
        product_left_kernel.value, *default_kernel_params(product_left_kernel.value)
    )
    _right_cov, _right_stationary = kernel_covariance(
        product_right_kernel.value, *default_kernel_params(product_right_kernel.value)
    )
    plot_cov(
        X_grid,
        to_np((_left_cov * _right_cov)(X_grid)),
        stationary=_left_stationary and _right_stationary,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare this to the result when the kernels are added instead of multiplied. When you add two kernels you are essentially specifying an `OR` operation. So, a summed kernel will have a high value if either of the constituent kernels has a high value.
    """)
    return


@app.cell(hide_code=True)
def _(sum_left_kernel, sum_right_kernel):
    mo.hstack([sum_left_kernel, sum_right_kernel], gap=1)
    return


@app.cell(hide_code=True)
def _(
    X_grid,
    default_kernel_params,
    kernel_covariance,
    plot_cov,
    sum_left_kernel,
    sum_right_kernel,
):
    _left_cov, _left_stationary = kernel_covariance(
        sum_left_kernel.value, *default_kernel_params(sum_left_kernel.value)
    )
    _right_cov, _right_stationary = kernel_covariance(
        sum_right_kernel.value, *default_kernel_params(sum_right_kernel.value)
    )
    plot_cov(
        X_grid,
        to_np((_left_cov + _right_cov)(X_grid)),
        stationary=_left_stationary and _right_stationary,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we have a general idea about covariance functions, let's begin by choosing one for our first model.

    We can use a Matern(5/2) covariance to model our simulated data, and pass this as the `cov_func` argument to the `Marginal` class.
    """)
    return


@app.cell
def _(sim_x, sim_y):
    def build_sim_marginal_model():
        with pm.Model() as sim_model:
            ell = pm.Gamma("ell", alpha=2, beta=1)
            eta = pm.HalfCauchy("eta", beta=5)
            cov_func = eta**2 * pm.gp.cov.Matern52(1, ell)
            mean_func = pm.gp.mean.Constant(c=1)
            gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

            sigma = pm.HalfCauchy("sigma", beta=5)
            gp.marginal_likelihood("obs", X=sim_x.reshape(-1, 1), y=sim_y, sigma=sigma)
        return sim_model, gp

    sim_model, sim_gp = build_sim_marginal_model()
    return sim_gp, sim_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The `.marginal_likelihood` method

    The unknown latent function can be analytically integrated out of the product of the GP prior probability with a normal likelihood. This quantity is called the marginal likelihood.

    $$
    p(y \mid x) = \int p(y \mid f, x) \, p(f \mid x) \, df
    $$

    The log of the marginal likelihood, $p(y \mid x)$, is

    $$
    \log p(y \mid x) = -\frac{1}{2} (\mathbf{y} - \mathbf{m}_x)^{T} (\mathbf{K}_{xx} + \boldsymbol\Sigma)^{-1} (\mathbf{y} - \mathbf{m}_x) - \frac{1}{2}|\mathbf{K}_{xx} + \boldsymbol\Sigma| - \frac{n}{2}\log (2 \pi)
    $$

    $\boldsymbol\Sigma$ is the covariance matrix of the Gaussian noise. Since the Gaussian noise doesn't need to be white to be conjugate, the `marginal_likelihood` method supports either using a white noise term when a scalar is provided, or a noise covariance function when a covariance function is provided.

    The `gp.marginal_likelihood` method implements the quantity given above. Schematically, our model looks like this. Notice that the marginal likelihood is represented by a multivariate normal node under the hood.
    """)
    return


@app.cell(hide_code=True)
def _(sim_model):
    sim_model
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's fit the model using Markov chain Monte Carlo (MCMC).
    """)
    return


@app.cell
def _(sim_model):
    with sim_model:
        sim_post = pm.sample(nuts_sampler="nutpie", chains=2, random_seed=RANDOM_SEED)
    sim_post
    return (sim_post,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can collect the results into a `DataFrame` to display, and compare to the values that we used to simulate the data.
    """)
    return


@app.cell(hide_code=True)
def _(sim_eta_true, sim_l_true, sim_post, sim_sigma_true):
    sim_summary = az.summary(
        sim_post, var_names=["ell", "eta", "sigma"], round_to=2, kind="stats"
    )
    sim_summary["True value"] = [sim_l_true, sim_eta_true, sim_sigma_true]
    sim_summary
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The `.conditional` distribution

    In addition to fitting the model, we would like to be able to generate predictions. This implies sampling from the posterior predictive distribution, which if you recall is just some linear algebra:

    $$\begin{aligned}
    m^*(x^*) &= k(x^*,x)^T[k(x,x) + \sigma^2 I]^{-1}y \\
    k^*(x^*) &= k(x^*,x^*) + \sigma^2 - k(x^*,x)^T[k(x,x) + \sigma^2 I]^{-1}k(x^*,x)
    \end{aligned}$$

    PyMC allows for predictive sampling after the model is fit, using the recorded values of the model parameters to generate samples. The `conditional` method implements the predictive GP above, called with a grid of points over which to generate realizations.

    The `.conditional` method has an optional flag for `pred_noise`, which defaults to `False`. When `pred_noise=False`, the `conditional` method produces the predictive distribution for the underlying function represented by the GP. When `pred_noise=True`, it produces the predictive distribution for the GP plus noise.

    We can define a grid of new values from $x=0$ to $x=20$, then add the GP conditional to the model, given the new X values.
    """)
    return


@app.cell
def _(sim_gp, sim_model):
    sim_X_new = np.linspace(0, 20, 600)[:, None]
    with sim_model:
        sim_gp.conditional("f_pred", sim_X_new)
    sim_X_new.shape
    return (sim_X_new,)


@app.cell
def _(sim_model, sim_post):
    with sim_model:
        sim_pred_samples = pm.sample_posterior_predictive(
            sim_post.sel(draw=slice(0, 20)), var_names=["f_pred"]
        )
    return (sim_pred_samples,)


@app.cell(hide_code=True)
def _(sim_pred_samples):
    sim_pred_samples
    return


@app.cell(hide_code=True)
def _(sim_X_new, sim_f_true, sim_pred_samples, sim_x, sim_y):
    _fig = plt.figure(figsize=(12, 5))
    _ax = _fig.gca()

    _f_pred_samples = az.extract(
        sim_pred_samples, group="posterior_predictive", var_names=["f_pred"]
    )
    plot_gp_dist(_ax, samples=_f_pred_samples.T, x=sim_X_new)

    _ax.plot(sim_x, sim_f_true, "dodgerblue", lw=3, label="True f")
    _ax.plot(sim_x, sim_y, "ok", ms=3, alpha=0.5, label="Observed data")

    _ax.set_xlabel("X")
    _ax.set_ylim([-13, 13])
    _ax.set_title("Posterior distribution over $f(x)$ at the observed values")
    _ax.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Real-world example: Fastball spin rates

    That was contrived data; let's try applying Gaussian processes to a baseball problem. Consider a time series of game-averaged fastball spin rates for a single pitcher.
    """)
    return


@app.cell(hide_code=True)
def _():
    spin_rates = pd.read_csv(
        data_path / "fastball_spin_rates.csv",
        index_col=0,
        parse_dates=["game_date"],
    )
    spin_rates.head()
    return (spin_rates,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Consider Michael Kopech:
    """)
    return


@app.cell(hide_code=True)
def _(spin_rates):
    kopech_fb_spin = (
        spin_rates.assign(day_of_year=spin_rates.game_date.dt.day_of_year)
        .loc["Kopech, Michael"]
        .copy()
    )
    kopech_fb_spin.plot.scatter(
        x="game_date",
        y="avg_spin_rate",
        title="Kopech Fastball Spin Rate",
        ylabel="Spin Rate (rpm)",
        figsize=(10, 4),
    )
    plt.gca()
    return (kopech_fb_spin,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can build a similar model to what we used for the simulated data.
    """)
    return


@app.cell
def _(kopech_fb_spin, spin_rates):
    def build_spin_rate_model():
        with pm.Model() as spin_rate_model:
            ell = pm.LogNormal("ell", 0, 1)
            eta = pm.LogNormal("eta", 0, 1)

            mean_fn = pm.gp.mean.Constant(spin_rates.avg_spin_rate.mean())
            cov_fn = (eta**2) * pm.gp.cov.ExpQuad(1, ell)

            sigma = pm.HalfNormal("sigma", 5)
            gp = pm.gp.Marginal(mean_func=mean_fn, cov_func=cov_fn)
            gp.marginal_likelihood(
                "spin_rates",
                X=kopech_fb_spin.day_of_year.values.reshape(-1, 1),
                y=kopech_fb_spin.avg_spin_rate.values,
                sigma=sigma,
            )
        return spin_rate_model, gp

    spin_rate_model, spin_rate_gp = build_spin_rate_model()
    return spin_rate_gp, spin_rate_model


@app.cell
def _(spin_rate_model):
    with spin_rate_model:
        spin_rate_trace = pm.sample(
            draws=1000,
            tune=2000,
            chains=2,
            random_seed=RANDOM_SEED,
        )
    return (spin_rate_trace,)


@app.cell(hide_code=True)
def _(spin_rate_trace):
    az.plot_trace(spin_rate_trace, var_names=["ell", "eta", "sigma"])
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now, let's generate estimates for the 2023 season.
    """)
    return


@app.cell
def _(kopech_fb_spin, spin_rate_gp, spin_rate_model, spin_rate_trace):
    spin_X_pred = np.arange(
        kopech_fb_spin.day_of_year.min(), kopech_fb_spin.day_of_year.max()
    ).reshape(-1, 1)
    with spin_rate_model:
        spin_rate_gp.conditional("spin_rate_pred", spin_X_pred)
        spin_rate_samples = pm.sample_posterior_predictive(
            spin_rate_trace.sel(draw=slice(0, 5)), var_names=["spin_rate_pred"]
        )
    return spin_X_pred, spin_rate_samples


@app.cell(hide_code=True)
def _(kopech_fb_spin, spin_X_pred, spin_rate_samples):
    _ax = kopech_fb_spin.plot.scatter(
        x="day_of_year", y="avg_spin_rate", c="k", s=50, figsize=(10, 4)
    )
    for _x in az.extract(
        spin_rate_samples, group="posterior_predictive", var_names="spin_rate_pred"
    ).values.T:
        _ax.plot(spin_X_pred, _x, alpha=0.5, c="r")
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can also use the `.predict` method on the GP to get the posterior mean and variance at a single posterior draw, giving a closed-form mean and 2σ band.
    """)
    return


@app.cell
def _(kopech_fb_spin, spin_X_pred, spin_rate_gp, spin_rate_model, spin_rate_trace):
    with spin_rate_model:
        _mu, _var = spin_rate_gp.predict(
            spin_X_pred,
            point=az.extract(
                spin_rate_trace.posterior.sel(draw=[0], chain=[0])
            ).squeeze(),
            diag=True,
        )
    _sd = np.sqrt(_var)

    _fig = plt.figure(figsize=(12, 5))
    _ax = _fig.gca()
    _ax.plot(spin_X_pred, _mu, "r", lw=2, label="mean and 2σ region")
    _ax.plot(spin_X_pred, _mu + 2 * _sd, "r", lw=1)
    _ax.plot(spin_X_pred, _mu - 2 * _sd, "r", lw=1)
    _ax.fill_between(
        spin_X_pred.flatten(),
        _mu - 2 * _sd,
        _mu + 2 * _sd,
        color="r",
        alpha=0.5,
    )
    kopech_fb_spin.plot.scatter(x="day_of_year", y="avg_spin_rate", c="k", s=50, ax=_ax)
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Multi-output GPs

    We can generalize this model to accommodate multiple pitchers using a **multi-output Gaussian process**. This lets us simultaneously learn and infer many outputs that share the same source of uncertainty from their inputs. There are several approaches to multi-output GPs; we will focus on the **Intrinsic Coregionalization Model** (ICM), which uses the Hadamard product (*i.e.* element-wise product) between a `Coregion` kernel and an arbitrary input kernel.

    $$ K_{ICM} = B \otimes K_{ExpQuad} $$

    where $B(o,o')$ is the output kernel and $K_{ExpQuad}(x,x')$ is an input kernel, and

    $$ B = WW^T + \mathrm{diag}(\kappa). $$

    The `Coregion` kernel is a **coregionalization** kernel: it expresses the idea that the outputs are related to each other. Coregionalization models capture the dependency between multiple variables by modeling their covariance structure, assuming the processes generating these variables are related and that their variations are not independent.

    Rather than just picking on Michael Kopech, let's look at the five pitchers with the most game appearances in the 2023 season.
    """)
    return


@app.cell(hide_code=True)
def _(spin_rates):
    n_outputs = 5
    top_pitchers = (
        spin_rates.groupby("pitcher_name").size().nlargest(n_outputs).reset_index()
    )
    top_pitchers = top_pitchers.reset_index().rename(
        columns={"index": "output_idx", 0: "games"}
    )

    _fig, _ax = plt.subplots(1, 1, figsize=(14, 6))
    for _pitcher in top_pitchers["pitcher_name"]:
        _pdata = spin_rates.assign(day_of_year=spin_rates.game_date.dt.day_of_year).loc[
            _pitcher
        ]
        _ax.scatter(_pdata["day_of_year"], _pdata["avg_spin_rate"], label=_pitcher)
    _ax.set_xlabel("Day of year")
    _ax.set_ylabel("Average spin rate (rpm)")
    _ax.legend(loc="upper center")
    plt.gca()
    return n_outputs, top_pitchers


@app.cell(hide_code=True)
def _(spin_rates, top_pitchers):
    analysis_subset = (
        spin_rates.assign(day_of_year=spin_rates.game_date.dt.day_of_year)
        .reset_index()
        .merge(top_pitchers, on="pitcher_name", how="right")
    )
    multi_X = analysis_subset[["day_of_year", "output_idx"]].values
    multi_y = analysis_subset["avg_spin_rate"].values
    return analysis_subset, multi_X, multi_y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here is a convenience function for taking the Hadamard product of the coregionalization kernel and an arbitrary covariance function:
    """)
    return


@app.cell
def _():
    def get_icm(input_dim, kernel, W=None, kappa=None, B=None, active_dims=None):
        """Hadamard product of a Coregion kernel and an input covariance function."""
        coreg = pm.gp.cov.Coregion(
            input_dim=input_dim, W=W, kappa=kappa, B=B, active_dims=active_dims
        )
        return kernel * coreg

    return (get_icm,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The model is essentially the same as the single-pitcher GP, except for the more complex covariance function that captures the dependence between pitchers (*i.e.* the assumption that they are governed by a similar underlying process).
    """)
    return


@app.cell
def _(get_icm, multi_X, multi_y, n_outputs):
    def build_multi_spin_rate_model():
        with pm.Model() as multi_spin_rate_model:
            ell = pm.Gamma("ell", alpha=2, beta=0.5)
            eta = pm.Gamma("eta", alpha=3, beta=1)
            cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=2, ls=ell, active_dims=[0])

            W = pm.Normal(
                "W",
                mu=0,
                sigma=3,
                shape=(n_outputs, 2),
                initval=RNG.standard_normal((n_outputs, 2)),
            )
            kappa = pm.Gamma("kappa", alpha=1.5, beta=1, shape=n_outputs)
            coreg_B = pm.Deterministic("B", pt.dot(W, W.T) + pt.diag(kappa))
            cov_icm = get_icm(input_dim=2, kernel=cov, B=coreg_B, active_dims=[1])

            gp = pm.gp.Marginal(cov_func=cov_icm)
            sigma = pm.HalfNormal("sigma", sigma=3)
            gp.marginal_likelihood("f", multi_X, multi_y, sigma=sigma)
        return multi_spin_rate_model, gp

    multi_spin_rate_model, multi_gp = build_multi_spin_rate_model()
    return multi_gp, multi_spin_rate_model


@app.cell
def _(multi_spin_rate_model):
    with multi_spin_rate_model:
        multi_trace = pm.sample(
            nuts_sampler="nutpie",
            chains=2,
            target_accept=0.9,
            random_seed=RANDOM_SEED,
        )
    multi_trace
    return (multi_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Again, we specify a grid of inputs to predict over — every day of the season for each of the five pitchers — and add the GP conditional to the model:
    """)
    return


@app.cell(hide_code=True)
def _(analysis_subset, n_outputs):
    multi_days_pred = np.arange(
        analysis_subset.day_of_year.min(), analysis_subset.day_of_year.max()
    )
    multi_pitcher_ind = np.repeat(np.arange(n_outputs), len(multi_days_pred))
    multi_X_new = np.column_stack(
        (np.tile(multi_days_pred, n_outputs), multi_pitcher_ind)
    )
    return multi_X_new, multi_days_pred, multi_pitcher_ind


@app.cell
def _(multi_X_new, multi_gp, multi_spin_rate_model, multi_trace):
    _multi_posterior = az.extract(
        multi_trace,
        group="posterior",
        var_names=["ell", "eta", "W", "kappa", "sigma"],
        combined=True,
    ).isel(sample=slice(0, 100))
    _multi_points = [
        {
            _name: _multi_posterior[_name].isel(sample=_sample_idx).values
            for _name in _multi_posterior.data_vars
        }
        for _sample_idx in range(_multi_posterior.sizes["sample"])
    ]

    with multi_spin_rate_model:
        if "preds" not in multi_spin_rate_model.named_vars:
            multi_gp.conditional("preds", multi_X_new)
        multi_gp_samples = pm.sample_posterior_predictive(
            _multi_points,
            var_names=["preds"],
            random_seed=RANDOM_SEED,
            backend="c",
        )
    return (multi_gp_samples,)


@app.cell(hide_code=True)
def _(
    analysis_subset,
    multi_X_new,
    multi_days_pred,
    multi_gp_samples,
    multi_pitcher_ind,
    n_outputs,
    top_pitchers,
):
    multi_f_pred = az.extract(
        multi_gp_samples, group="posterior_predictive", var_names="preds"
    ).values.T  # (sample, point)
    _ = multi_pitcher_ind

    _fig, _axes = plt.subplots(n_outputs, 1, figsize=(12, 15))
    _M = len(multi_days_pred)
    for _idx, _pitcher in enumerate(top_pitchers["pitcher_name"]):
        _ax = _axes[_idx]
        plot_gp_dist(
            _ax,
            multi_f_pred[:, _M * _idx : _M * (_idx + 1)],
            multi_X_new[_M * _idx : _M * (_idx + 1), 0],
            palette="Blues",
            fill_alpha=0.1,
            samples_alpha=0.1,
        )
        _cond = analysis_subset["pitcher_name"] == _pitcher
        _ax.scatter(
            analysis_subset.loc[_cond, "day_of_year"],
            analysis_subset.loc[_cond, "avg_spin_rate"],
            color="r",
        )
        _ax.set_title(_pitcher)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Latent Variable Implementation

    The `gp.Latent` class is a more general implementation of a GP. It is called "Latent" because the underlying function values are treated as latent variables. It has a `prior` method and a `conditional` method. Given a mean and covariance function, the function $f(x)$ is modeled as

    $$
    f(x) \sim \mathcal{GP}(m(x),\, k(x, x')) \,.
    $$

    ## `.prior`

    Since the output is now considered latent, we are no longer using a Gaussian likelihood (or at least, we don't have to), so instead we are using the GP as a prior.

    With some data set of finite size, the `prior` method places a multivariate normal prior distribution on the vector of function values, $\mathbf{f}$,

    $$
    \mathbf{f} \sim \text{MvNormal}(\mathbf{m}_{x},\, \mathbf{K}_{xx}) \,,
    $$

    where the vector $\mathbf{m}$ and the matrix $\mathbf{K}_{xx}$ are the mean vector and covariance matrix evaluated over the inputs $x$.

    It is useful to reparameterize the prior on `f` under the hood by rotating it with the Cholesky factor of its covariance matrix. This helps to reduce covariances in the posterior of the transformed random variable, `v`. The reparameterized model is

    $$
    \begin{aligned}
    \mathbf{v} &\sim \text{N}(0, 1) \\
    \mathbf{L} &= \text{Cholesky}(\mathbf{K}_{xx}) \\
    \mathbf{f} &= \mathbf{m}_{x} + \mathbf{Lv}
    \end{aligned}
    $$

    In PyMC, this reparameterization can be disabled by setting the optional flag in the `prior` method, `reparameterize=False`. The default is `True`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Robust regression

    The following is an example showing how to specify a simple model with a GP prior using the `gp.Latent` class. So we can verify that the inference we perform is correct, the data set is made using a draw from a GP. This will be identical to the first example, except that the noise is Student-T distributed.
    """)
    return


@app.cell
def _():
    def simulate_robust_data(seed=RANDOM_SEED + 7, n=100):
        rng = np.random.default_rng(seed)
        X = np.linspace(0, 10, n)[:, None]

        ls_true = 1.0
        eta_true = 3.0
        cov_func = eta_true**2 * pm.gp.cov.Matern52(1, ls_true)
        mean_func = pm.gp.mean.Zero()

        f_true = rng.multivariate_normal(
            to_np(mean_func(X)), to_np(cov_func(X)) + 1e-8 * np.eye(n), 1
        ).flatten()

        sigma_true = 2.0
        nu_true = 3.0
        y = f_true + sigma_true * rng.standard_t(nu_true, size=n)
        return X, f_true, y, ls_true, eta_true, sigma_true, nu_true

    (
        robust_X,
        robust_f_true,
        robust_y,
        _robust_ls_true,
        _robust_eta_true,
        _robust_sigma_true,
        _robust_nu_true,
    ) = simulate_robust_data()

    _fig = plt.figure(figsize=(12, 5))
    _ax = _fig.gca()
    _ax.plot(robust_X, robust_f_true, "dodgerblue", lw=3, label="True f")
    _ax.plot(robust_X, robust_y, "ok", ms=3, label="Data")
    _ax.set_xlabel("X")
    _ax.set_ylabel("y")
    _ax.legend()
    plt.gca()
    return robust_X, robust_f_true, robust_y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's the model in PyMC. We use a $\text{Gamma}(2, 1)$ prior over the lengthscale parameter, and weakly informative $\text{HalfCauchy}(2)$ priors over the covariance function scale and noise scale. A $\text{Gamma}(2, 0.1)$ prior is assigned to the degrees of freedom parameter of the noise. Finally, a GP prior is placed on the unknown function.
    """)
    return


@app.cell
def _(robust_X, robust_y):
    def build_robust_model():
        with pm.Model() as robust_model:
            ls = pm.Gamma("ls", alpha=2, beta=1)
            eta = pm.HalfCauchy("eta", beta=2)
            cov = eta**2 * pm.gp.cov.Matern52(1, ls)
            gp = pm.gp.Latent(cov_func=cov)

            f = gp.prior("f", X=robust_X)

            sigma = pm.HalfCauchy("sigma", beta=2)
            nu = pm.Gamma("nu", alpha=2, beta=0.1)
            pm.StudentT("y", mu=f, lam=1.0 / sigma, nu=nu, observed=robust_y)
        return robust_model

    robust_model = build_robust_model()
    return (robust_model,)


@app.cell
def _(robust_model):
    with robust_model:
        robust_trace = pm.sample(
            nuts_sampler="nutpie",
            chains=2,
            random_seed=RANDOM_SEED,
        )
    robust_trace
    return (robust_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Below are the posteriors of the covariance function hyperparameters.
    """)
    return


@app.cell(hide_code=True)
def _(robust_trace):
    az.plot_trace(robust_trace, var_names=["eta", "sigma", "ls", "nu"])
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(robust_X, robust_f_true, robust_trace, robust_y):
    _fig = plt.figure(figsize=(12, 5))
    _ax = _fig.gca()

    plot_gp_dist(_ax, robust_trace.posterior["f"].values[0], robust_X)

    _ax.plot(robust_X, robust_f_true, "dodgerblue", lw=3, label="True f")
    _ax.plot(robust_X, robust_y, "ok", ms=3, alpha=0.5, label="Observed data")

    _ax.set_xlabel("X")
    _ax.set_ylabel("True f(x)")
    _ax.set_title("Posterior distribution over $f(x)$ at the observed values")
    _ax.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Latent Poisson GP for the coal-mining disasters

    The exact `gp.Marginal` examples above assume Gaussian observation noise. Counts need a different likelihood: annual disaster totals are non-negative integers, so a Gaussian likelihood can predict impossible negative counts.

    For annual UK coal-mining disasters (1851–1961), model the **log disaster rate** with a latent Gaussian process and connect it to the observed counts with a Poisson likelihood:

    $$f(x) \sim \mathcal{GP}(0, k(x, x')), \qquad y_i \sim \mathrm{Poisson}\bigl(\exp(f(x_i))\bigr).$$

    Use a Matérn-5/2 covariance and put 94% prior mass on lengthscales between **2 and 10 years**. That range says policy shifts, labor reforms, and reporting changes can alter the disaster rate over a few years, but the model should not chase one-year noise.
    """)
    return


@app.cell(hide_code=True)
def _():
    # fmt: off
    disasters_array = np.array([
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4,
        5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2,
        2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2,
        0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
        3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
    ])
    # fmt: on
    disaster_years = np.arange(1851, 1962, dtype=np.float64)

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.bar(disaster_years, disasters_array, color="#154A72", alpha=0.75, width=0.8)
    _ax.set_title("UK coal-mining disasters (1851–1961)")
    _ax.set_xlabel("Year")
    _ax.set_ylabel("Disaster count")
    plt.tight_layout()
    plt.gca()
    return disaster_years, disasters_array


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(r"""
    1. Build a `pm.gp.Latent` model for `disasters_array` using `disaster_years[:, None]` as the input.
    2. Use `pm.find_constrained_prior` to choose a `Gamma` prior for the Matérn-5/2 lengthscale with 94% mass in `[2, 10]` years. (`pm.find_constrained_prior` is PyMC's analogue of the `pz.maxent` elicitation you used in Session 1.2 — it solves for the distribution parameters that put a target probability mass between bounds.)
    3. Put a weakly-informative prior on the GP amplitude, exponentiate the latent GP to get a positive `rate`, and use a `Poisson` likelihood.
    4. Sample the model and plot the posterior mean disaster rate with a 94% HDI over the observed counts.
    5. Optional: run a posterior predictive check. Does a Poisson likelihood reproduce the observed count dispersion?
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Hint": mo.md(r"""
    The full `find_constrained_prior` call for the lengthscale:

    ```python
    ell_params = pm.find_constrained_prior(
        pm.Gamma,
        lower=2,
        upper=10,
        init_guess={"alpha": 2, "beta": 0.5},
        mass=0.94,
    )
    ```
    """)
        }
    )
    return


@app.cell
def _(disaster_years, disasters_array):
    def exercise_disaster_gp():
        # YOUR CODE HERE — elicit the lengthscale prior (see hint)
        ell_params = ...

        with pm.Model():
            # YOUR CODE HERE — Gamma lengthscale from ell_params, and a
            # HalfNormal amplitude
            ell = ...
            eta = ...
            cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
            gp = pm.gp.Latent(cov_func=cov)
            f = gp.prior("f", X=disaster_years[:, None])
            # YOUR CODE HERE — Poisson likelihood on disasters_array with
            # an exponential link: mu = pm.math.exp(f)
            disaster_trace = pm.sample(random_seed=RANDOM_SEED)
        return az.summary(disaster_trace, var_names=["ell", "eta"])

    return (exercise_disaster_gp,)


@app.cell(hide_code=True)
def _():
    run_disaster_gp = mo.ui.run_button(label="▶ Run exercise")
    run_disaster_gp
    return (run_disaster_gp,)


@app.cell(hide_code=True)
def _(exercise_disaster_gp, run_disaster_gp):
    mo.stop(
        not run_disaster_gp.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_disaster_gp()
    return


@app.cell(hide_code=True)
def _(disaster_years, disasters_array):
    def solution_disaster_gp():
        ell_params = pm.find_constrained_prior(
            pm.Gamma,
            lower=2,
            upper=10,
            init_guess={"alpha": 2, "beta": 0.5},
            mass=0.94,
        )
        with pm.Model() as disaster_model:
            ell = pm.Gamma("ell", **ell_params)
            eta = pm.HalfNormal("eta", sigma=2)
            cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
            gp = pm.gp.Latent(cov_func=cov)
            f = gp.prior("f", X=disaster_years[:, None])
            rate = pm.Deterministic("rate", pm.math.exp(f))
            pm.Poisson("y", mu=rate, observed=disasters_array)
            disaster_trace = pm.sample(random_seed=RANDOM_SEED)

        summary = az.summary(disaster_trace, var_names=["ell", "eta"])

        rate_samples = az.extract(disaster_trace, var_names="rate").values
        rate_mean = rate_samples.mean(axis=1)
        rate_hdi = az.hdi(disaster_trace["posterior"], prob=0.94)["rate"].to_numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(disaster_years, disasters_array, color="#154A72", alpha=0.35, width=0.8)
        ax.plot(
            disaster_years,
            rate_mean,
            color="firebrick",
            lw=2,
            label="Posterior mean rate",
        )
        ax.fill_between(
            disaster_years, rate_hdi[:, 0], rate_hdi[:, 1], color="firebrick", alpha=0.2
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Disaster count / rate")
        ax.legend()
        fig.tight_layout()
        return mo.vstack([summary, fig])

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_disaster_gp)}\n```"),
                    mo.lazy(solution_disaster_gp, show_loading_indicator=True),
                    mo.md(r"""
    For the optional PPC, simulate replicated counts and compare a count-dispersion statistic such as variance / mean:

    ```python
    with disaster_model:
        disaster_ppc = pm.sample_posterior_predictive(
            disaster_trace,
            var_names=["y"],
            random_seed=RANDOM_SEED,
        )

    obs_ratio = disasters_array.var() / disasters_array.mean()
    sim_counts = az.extract(disaster_ppc, group="posterior_predictive", var_names="y").values
    sim_ratios = sim_counts.var(axis=0) / sim_counts.mean(axis=0)
    ```

    If the observed ratio is far in the tail of `sim_ratios`, the Poisson variance assumption is too tight. The usual next move is a `NegativeBinomial` likelihood with the same latent GP rate and an additional dispersion parameter.
    """),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Faster Gaussian processes

    One of the major constraints that limits the utility of Gaussian processes in practice is the inversion of $K$ when calculating the posterior covariance. Since it is evaluated at every observed data point, its execution time is $\mathcal{O}(n^3)$, which makes exact Gaussian processes impractical for larger datasets.

    $$ k^*(x^*) = k(x^*,x^*)+\sigma^2 - k(x^*,x)^T \underbrace{[k(x,x) + \sigma^2 I]^{-1}}_{😧}k(x^*,x) $$

    An approach for dealing with this computational complexity is to look for an approximation to accelerate training and prediction. For Gaussian processes, this can be accomplished by employing a **sparse approximation** to the Gram matrix that places $m \ll n$ *inducing points* along the range of the input variables, and uses this to estimate the full covariance matrix for the observed points. This reduces the time complexity to $\mathcal{O}(nm^2)$.

    Having chosen a subset of $m$ points, we can decompose the covariance matrix $K$ into blocks as follows:

    $$
    K=\left(\begin{array}{cc}
    K_{m m} & K_{m(n-m)} \\
    K_{(n-m) m} & K_{(n-m)(n-m)}
    \end{array}\right)
    $$

    From this, we can use the [Nyström method](https://en.wikipedia.org/wiki/Low-rank_matrix_approximations#Nystr%C3%B6m_approximation) to approximate the eigenvalues and eigenvectors of the full covariance matrix. This results in the following approximation:

    $$
    \tilde{K} = K_{n m} K_{m m}^{-1} K_{m n}
    $$

    The `gp.MarginalApprox` class implements sparse GP approximations. It works identically to `gp.Marginal`, except it additionally requires the locations of the inducing points (denoted `Xu`).

    The downside of sparse approximations is that they **reduce the expressiveness** of the GP. Reducing the dimension of the covariance matrix effectively reduces the number of covariance matrix eigenvectors that can be used to fit the data.

    A choice that needs to be made is where to place the inducing points. One option is to use a subset of the inputs. Another possibility is to use K-means. The location of the inducing points can also be an unknown and optimized as part of the model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Dense dataset

    For the following examples, we use the same data set as was used in the `gp.Marginal` example, but with more data points.
    """)
    return


@app.cell
def _():
    def simulate_dense_data(seed=RANDOM_SEED, n=2000):
        rng = np.random.default_rng(seed)
        X = 10 * np.sort(rng.uniform(size=n))[:, None]

        ls_true = 1.0
        eta_true = 3.0
        cov_func = eta_true**2 * pm.gp.cov.Matern52(1, ls_true)
        mean_func = pm.gp.mean.Zero()

        f_true = rng.multivariate_normal(
            to_np(mean_func(X)), to_np(cov_func(X)) + 1e-8 * np.eye(n), 1
        ).flatten()

        sigma_true = 2.0
        y = f_true + sigma_true * rng.standard_normal(n)
        return X, f_true, y

    dense_X, dense_f_true, dense_y = simulate_dense_data()

    _fig = plt.figure(figsize=(12, 5))
    _ax = _fig.gca()
    _ax.plot(dense_X, dense_f_true, "dodgerblue", lw=3, label="True f")
    _ax.plot(dense_X, dense_y, "ok", ms=3, alpha=0.5, label="Data")
    _ax.set_xlabel("X")
    _ax.set_ylabel("The true f(x)")
    _ax.legend()
    plt.gca()
    return dense_X, dense_f_true, dense_y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The inducing points don't need to coincide with observations — they're auxiliary variables introduced solely to compress the GP. The sparse GP will:

    1. Learn latent function values at the 20 inducing locations.
    2. Use the kernel to propagate information from inducing points to observations.
    3. Interpolate smoothly from inducing points to any prediction location.

    ### The approximation, visualized

    The sparse GP replaces the full covariance $K_{nn}$ with the low-rank reconstruction $K_{nm} K_{mm}^{-1} K_{mn}$. Smooth GPs have covariance matrices that are **inherently low-rank** — nearby points are highly correlated, distant points contribute almost nothing — so the approximation is tight.
    """)
    return


@app.cell
def _(dense_X):
    def plot_sparse_covariance_approx():
        idx = np.sort(RNG.choice(dense_X.shape[0], 200, replace=False))
        X_viz = dense_X[idx]
        cov_viz = 3.0**2 * pm.gp.cov.Matern52(1, 1.0)
        K_full = cov_viz(X_viz).eval()

        Xu_poor = np.linspace(dense_X.min(), dense_X.max(), 4)[:, None]
        Xu_enough = pm.gp.util.kmeans_inducing_points(20, dense_X)
        configs = [
            ("Too few inducing points (m = 4)", Xu_poor),
            (f"Enough inducing points (m = {len(Xu_enough)})", Xu_enough),
        ]

        approxes, errors = [], []
        for _, Xu in configs:
            Kmm = cov_viz(Xu).eval()
            Knm = cov_viz(X_viz, Xu).eval()
            K_approx = Knm @ np.linalg.solve(Kmm + 1e-8 * np.eye(Kmm.shape[0]), Knm.T)
            approxes.append(K_approx)
            errors.append(K_full - K_approx)

        vmax = float(max(K_full.max(), max(a.max() for a in approxes)))
        vemax = float(max(np.abs(e).max() for e in errors))

        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for row, ((label, _Xu), K_approx, err) in enumerate(
            zip(configs, approxes, errors)
        ):
            im_full = axes[row, 0].imshow(
                K_full, cmap="viridis", vmin=0, vmax=vmax, origin="lower"
            )
            axes[row, 0].set_title("Full covariance $K_{nn}$", fontsize=11)
            im_approx = axes[row, 1].imshow(
                K_approx, cmap="viridis", vmin=0, vmax=vmax, origin="lower"
            )
            axes[row, 1].set_title(
                f"{label}\n$K_{{nm}} K_{{mm}}^{{-1}} K_{{mn}}$", fontsize=11
            )
            im_err = axes[row, 2].imshow(
                err, cmap="RdBu_r", vmin=-vemax, vmax=vemax, origin="lower"
            )
            axes[row, 2].set_title("Approximation error", fontsize=11)
            for ax in axes[row]:
                ax.set_xlabel("obs index")
            axes[row, 0].set_ylabel("obs index")
            fig.colorbar(im_full, ax=axes[row, 0], shrink=0.8)
            fig.colorbar(im_approx, ax=axes[row, 1], shrink=0.8)
            fig.colorbar(im_err, ax=axes[row, 2], shrink=0.8)

        plt.tight_layout()
        return fig

    plot_sparse_covariance_approx()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can specify a sparse marginal likelihood model via `MarginalApprox`, where the approximation method can be chosen. We will use the **fully independent training conditional (FITC)** algorithm, with the critical approximation being the imposition of a conditional independence assumption on the joint prior over training and test cases. We will initialize 20 inducing points with the **K-means** algorithm.
    """)
    return


@app.cell
def _(dense_X, dense_y):
    def build_sparse_model():
        with pm.Model() as sparse_model:
            ls = pm.Gamma("ls", alpha=2, beta=1)
            eta = pm.HalfCauchy("eta", beta=5)
            cov = eta**2 * pm.gp.cov.ExpQuad(1, ls)
            gp = pm.gp.MarginalApprox(cov_func=cov, approx="FITC")

            Xu = pm.gp.util.kmeans_inducing_points(20, dense_X)

            sigma = pm.HalfCauchy("sigma", 5)
            gp.marginal_likelihood(
                "obs", X=dense_X, Xu=Xu, y=dense_y, sigma=sigma, jitter=1e-5
            )
        return sparse_model, gp, Xu

    sparse_model, sparse_gp, sparse_Xu = build_sparse_model()
    return sparse_Xu, sparse_gp, sparse_model


@app.cell
def _(sparse_model):
    with sparse_model:
        sparse_trace = pm.sample(
            nuts_sampler="nutpie",
            chains=2,
            random_seed=RANDOM_SEED,
        )
    sparse_trace
    return (sparse_trace,)


@app.cell(hide_code=True)
def _(sparse_trace):
    az.plot_trace(sparse_trace, var_names=["eta", "ls"])
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(sparse_gp, sparse_model, sparse_trace):
    sparse_X_new = np.linspace(-1, 11, 200)[:, None]
    with sparse_model:
        sparse_gp.conditional("f_pred", sparse_X_new)
        sparse_pred_samples = pm.sample_posterior_predictive(
            sparse_trace.sel(draw=slice(0, 5)), var_names=["f_pred"]
        )
    return sparse_X_new, sparse_pred_samples


@app.cell(hide_code=True)
def _(dense_X, dense_f_true, dense_y, sparse_X_new, sparse_Xu, sparse_pred_samples):
    _fig = plt.figure(figsize=(12, 5))
    _ax = _fig.gca()

    _f_pred_samples = az.extract(
        sparse_pred_samples,
        group="posterior_predictive",
        var_names=["f_pred"],
    )
    plot_gp_dist(_ax, _f_pred_samples.T, sparse_X_new)

    _ax.plot(dense_X, dense_y, "ok", ms=3, alpha=0.5, label="Observed data")
    _ax.plot(dense_X, dense_f_true, "dodgerblue", lw=3, label="True f")
    _ax.plot(
        sparse_Xu,
        10 * np.ones(sparse_Xu.shape[0]),
        "co",
        ms=10,
        label="Inducing point locations",
    )

    _ax.set_xlabel("X")
    _ax.set_ylim([-13, 13])
    _ax.set_title("Posterior distribution over $f(x)$ at the observed values")
    _ax.legend()
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hilbert Space Approximate Gaussian Processes (HSGP)

    ### The computational wall

    Exact GP regression inverts an $n \times n$ covariance matrix on every likelihood evaluation: $\mathcal{O}(n^3)$. With more than 2000 swing-decision observations, that is billions of operations per MCMC step — effectively impossible with `gp.Marginal`. Sparse GPs dropped us to $\mathcal{O}(nm^2)$ but still leave us choosing and placing inducing points. HSGP does better.

    ### The idea

    The **Hilbert-space Gaussian process** (HSGP, Solin & Särkkä 2020; Riutort-Mayol et al. 2023) replaces the covariance matrix with a low-rank decomposition using fixed **spectral** basis functions:

    $$f(x) \approx \sum_{j=1}^{m} \phi_j(x)\, \beta_j(\ell).$$

    - The **basis functions** $\phi_j(x)$ are sinusoidal-like and depend only on the input domain — *not* on the kernel, and *not* on inducing points we have to place.
    - The **coefficients** $\beta_j$ are determined by the kernel's **spectral density** — loosely, the kernel's power at each spatial frequency (formally, the Fourier transform of $k$). A long lengthscale concentrates power at low frequencies; a short one spreads it to higher ones. HSGP truncates this spectrum at a finite basis.

    Cost drops from $\mathcal{O}(n^3)$ to $\mathcal{O}(nm + m)$ — *linear* in $n$. The approximation is excellent whenever $m$ covers the frequencies the kernel actually cares about, and the input domain isn't too much smaller than the lengthscale.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How does it work?

    In HSGP, the covariance function is viewed as a mathematical operator acting on a high-dimensional space (Hilbert space). We represent the covariance function as a series expansion of eigenvalues and eigenfunctions of the Laplace operator. These eigenfunctions capture the smoothness properties of the underlying function being modeled in the form of a set of basis functions.

    $$f \sim \mathcal{GP}\bigl(0,\, k(x, x'; \ell)\bigr) \;\longrightarrow\; f \approx \phi(x)\,\beta(\ell)$$

    where the basis functions $\phi$ depend only on the input domain and the coefficients $\beta$ depend only on the kernel hyperparameters. This separation is what makes HSGP a *parametric* model — once the basis is precomputed, prediction is the same as any other PyMC regression. There is no `.conditional` step required: the HSGP is a drop-in component you can place inside any larger model with any likelihood function.
    """)
    return


@app.cell(hide_code=True)
def _():
    def plot_basis_functions():
        pymc_blue = "#154A72"
        pymc_green = "#81C240"
        pymc_light_blue = "#4A9EDE"
        pymc_dark_green = "#40611F"

        L = 5.0
        x_bf = np.linspace(-L, L, 300)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        colors_bf = [pymc_blue, pymc_green, pymc_light_blue, pymc_dark_green, "#C2C240"]
        for j, color in zip(range(1, 6), colors_bf):
            phi_j = np.sin(np.pi * j * (x_bf + L) / (2 * L)) / np.sqrt(L)
            ax1.plot(x_bf, phi_j, color=color, lw=1.5, alpha=0.85, label=f"φ_{j}")
        ax1.set_title("First 5 basis functions", fontsize=13)
        ax1.set_xlabel("x")
        ax1.legend(fontsize=9)

        rng_bf = np.random.default_rng(42)
        weights = rng_bf.normal(0, 1, 20) * np.exp(-0.3 * np.arange(20))
        f_approx = np.zeros_like(x_bf)
        for j in range(20):
            phi_j = np.sin(np.pi * (j + 1) * (x_bf + L) / (2 * L)) / np.sqrt(L)
            f_approx += weights[j] * phi_j
        ax2.plot(x_bf, f_approx, color=pymc_blue, lw=2.5)
        ax2.set_title("Weighted sum of 20 basis → smooth function", fontsize=13)
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        plt.tight_layout()
        return fig

    plot_basis_functions()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How `L` and `c` affect the basis

    The basis vectors above are sinusoids on an interval $[-L, L]$, which means they're forced to *pinch to zero* at the boundary. If $L$ sits right at the edge of the data, the approximation degrades there. PyMC builds $L$ from the data half-range $S = \max|x|$ and a multiplier $c$:

    $$L = c \cdot S.$$

    The plot below shows the first few basis vectors for three choices of $L$ on a domain $x \in [-5, 5]$ ($S = 5$). Watch what happens at the edges, and what happens to the *first* eigenvector as $L$ grows.
    """)
    return


@app.cell
def _():
    def plot_hsgp_basis_L_effect():
        pymc_blue = "#154A72"
        pymc_green = "#81C240"
        pymc_light_blue = "#4A9EDE"
        pymc_dark_green = "#40611F"

        x = np.linspace(-5, 5, 1000)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        plt.subplots_adjust(wspace=0.05)
        ylim = 0.55

        L_options = [5.0, 6.0, 20.0]
        m_options = [3, 3, 5]
        S = 5.0

        palette = [
            pymc_blue,
            pymc_green,
            pymc_light_blue,
            pymc_dark_green,
            "#C2C240",
        ]

        for i, ax in enumerate(axes):
            L = np.array([L_options[i]])
            m = [m_options[i]]
            eigvals = pm.gp.hsgp_approx.calc_eigenvalues(L, m)
            phi = pm.gp.hsgp_approx.calc_eigenvectors(x[:, None], L, eigvals, m).eval()
            for j in range(phi.shape[1]):
                ax.plot(
                    x,
                    phi[:, j],
                    color=palette[j % len(palette)],
                    lw=1.5,
                    label=rf"$\phi_{{{j + 1}}}$",
                )
            ax.set_ylim(-ylim, ylim)
            ax.set_xticks(np.arange(-5, 6, 5))
            if i > 0:
                ax.set_yticks([])
            ax.text(
                -4.9,
                -0.45,
                f"L = {L_options[i]}\nc = {L_options[i] / S:.1f}",
                fontsize=12,
            )
            ax.set_xlabel("x (centered)")

        axes[1].set_title("Effect of $L$ on the HSGP basis vectors", fontsize=13)
        axes[0].legend(loc="upper right", fontsize=8)
        return fig

    plot_hsgp_basis_L_effect()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - **Left ($L = S$, $c = 1$).** The basis pinches to zero exactly at the data edges. The HSGP approximation becomes poor as you approach $x = \pm 5$, and how quickly depends on the lengthscale.
    - **Middle ($c = 1.2$).** The Riutort-Mayol *et al.* recommended minimum. The pinch is pushed just outside the data range, which is usually enough.
    - **Right ($c = 4$).** Larger $c$ accommodates longer lengthscales — but the *period* of every basis function grows, so you need more of them (larger $m$) to recover short-lengthscale behavior. Notice the first eigenvector flattens almost completely; in this regime it can become unidentifiable with the model intercept, which is why we set `drop_first=True` whenever we have a separate intercept term.

    To summarize:

    - Increasing $m$ helps the HSGP approximate GPs with **smaller lengthscales**, at the cost of computation.
    - Increasing $c$ (or $L$) helps the HSGP approximate GPs with **larger lengthscales**, but may require larger $m$ to compensate.
    - Always consider the locations where you'll need to make predictions — they shouldn't sit near the boundary pinch either.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The two knobs

    **`m`** — number of basis functions. More = better approximation at short lengthscales. Typical: 20–200 in 1-D. If the kernel's lengthscale is much longer than the data spacing, small `m` is fine.

    **`c`** — boundary extension. Basis functions live on an interval extended by factor `c` beyond the data range, so edge effects don't bite. Typical: 1.3–4.0.

    Rather than guess, use **`pm.gp.hsgp_approx.approx_hsgp_hyperparams`**: give it the data's x-range and a plausible range of lengthscales, and it returns recommended `m` and `c` from the Riutort-Mayol et al. (2023) approximation-error bounds. We'll use it below.

    ### Centered vs non-centered parameterization

    We use `parametrization="noncentered"` by default. To see what this controls, here is essentially what `pm.gp.HSGP` does under the hood. `prior_linearized` returns the eigenvector basis `phi` and the square root of the power spectrum at the eigenvalues, `sqrt_psd`:

    ```python
    phi, sqrt_psd = gp.prior_linearized(Xs=Xs)

    # non-centered (default — better for tight posteriors / weak data)
    beta = pm.Normal("beta", size=gp._m_star)
    f = pm.Deterministic("f", phi @ (beta * sqrt_psd))

    # centered
    beta = pm.Normal("beta", sigma=sqrt_psd, size=gp._m_star)
    f = pm.Deterministic("f", phi @ beta)
    ```

    The same non-centered reparameterization trick used to tame hierarchical funnels — decorrelates the basis coefficients from their scale so NUTS doesn't stall. It matters most when the noise level is uncertain or data is sparse.

    **`drop_first=True`** removes the first (constant) basis function from the HSGP expansion. We include an explicit `intercept` term in the model, so keeping the constant basis would create collinearity. Always pair `drop_first=True` with a separate intercept.

    ### Choosing `m` and `c` for our problem

    In practice, you need to choose $c$ large enough to handle the largest lengthscales your prior allows, and $m$ large enough to accommodate the smallest. Our prior puts 90% mass on $\ell \in [1, 15]$ years (player age), so we pick $c$ and $m$ such that the entire prior range falls inside a valid region. Riutort-Mayol *et al.* give heuristics for the range of lengthscales that are accurately reproduced for given values of $m$ and $c$, summarized in the curves below.
    """)
    return


@app.cell(hide_code=True)
def _(swing_decisions):
    def plot_hsgp_parameter_curves():
        from matplotlib.ticker import MultipleLocator

        c_list = np.array([1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
        ell_grid = np.linspace(1, 25, 500)
        ages_arr = swing_decisions["age"].to_numpy().astype(float)
        S = (ages_arr - ages_arr.mean()).max()

        fig, ax = plt.subplots(figsize=(9, 4.5))
        cmap_curves = plt.cm.cividis
        colors = np.arange(len(c_list)) / len(c_list)

        for i, c_val in enumerate(c_list):
            m_curve = 2.65 * (c_val / ell_grid) * S
            valid = c_val >= (4.1 * (ell_grid / S))
            m_curve[~valid] = np.nan
            ax.semilogy(
                ell_grid,
                m_curve,
                color=cmap_curves(colors[i]),
                label=f"c = {c_val}",
                lw=2,
            )

        ax.grid(True, alpha=0.4)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_title("Matérn-5/2 HSGP approximation parameter curves", fontsize=12)
        ax.set_ylim(10, 1000)
        ax.set_xlabel("lengthscale ($\\ell$)")
        ax.set_ylabel("number of basis functions ($m$)")
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        return fig

    plot_hsgp_parameter_curves()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each curve marks where the HSGP approximation is reliable for that value of $c$. **The right value depends on your prior over $\ell$.** Our prior puts 90% of its mass on $\ell \in [1, 15]$ years (player age), so we pick $c$ and $m$ such that the entire prior range falls inside a valid region — and `approx_hsgp_hyperparams` does exactly that calculation for us, which is what the model below uses. Smaller $m$ is better for speed; $c$ has no effect on cost.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Swing decision data

    Let's revisit the swing decision dataset — which we earlier modeled with a spline — and fit an HSGP to it. There are well over 2000 data points, which would make this slow to fit with an exact GP.
    """)
    return


@app.cell(hide_code=True)
def _(swing_decisions):
    swing_decisions.plot.scatter(
        x="age",
        y="swing_decision",
        s=10,
        title="Swing Decisions 2023",
        alpha=0.3,
        legend=False,
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _(swing_decisions):
    ages_unique = np.sort(swing_decisions.age.unique())
    age_ind_obs = swing_decisions.age.values - ages_unique.min()
    swing_decision_obs_arr = swing_decisions.swing_decision.values
    ages_unique
    return age_ind_obs, ages_unique, swing_decision_obs_arr


@app.cell
def _(age_ind_obs, ages_unique, swing_decision_obs_arr):
    def build_hsgp_swing_decision_model():
        import warnings

        m_opt, c_opt = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
            x_range=[float(ages_unique.min()), float(ages_unique.max())],
            lengthscale_range=[1.0, 15.0],
            cov_func="matern52",
        )
        print(f"approx_hsgp_hyperparams recommends: m = {m_opt}, c = {c_opt:.2f}")

        coords = {"age": ages_unique}
        with pm.Model(coords=coords) as swing_decision_model:
            eta = pm.Exponential("eta", lam=1)

            ell_params = pm.find_constrained_prior(
                pm.Lognormal,
                lower=1,
                upper=15,
                init_guess={"mu": 1, "sigma": 1},
                mass=0.9,
            )
            ell = pm.Lognormal("ell", **ell_params)

            cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                gp = pm.gp.HSGP(
                    m=[m_opt],
                    c=c_opt,
                    cov_func=cov,
                    parametrization="noncentered",
                    drop_first=True,
                )
            f = gp.prior("f", X=ages_unique[:, None].astype(float), dims="age")

            intercept = pm.Normal(
                "intercept", mu=float(swing_decision_obs_arr.mean()), sigma=5
            )
            mu = pm.Deterministic("mu", intercept + f[age_ind_obs])
            sigma = pm.HalfNormal("sigma", sigma=10)
            pm.Normal("y", mu=mu, sigma=sigma, observed=swing_decision_obs_arr)
        return swing_decision_model, gp

    swing_decision_model, swing_decision_gp = build_hsgp_swing_decision_model()
    return (swing_decision_model,)


@app.cell
def _(swing_decision_model):
    with swing_decision_model:
        swing_decision_trace = pm.sample(
            draws=2000,
            tune=3000,
            target_accept=0.97,
            nuts_sampler="nutpie",
            chains=4,
            random_seed=RANDOM_SEED,
        )
    swing_decision_trace
    return (swing_decision_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The resulting diagnostics look pretty good. We use ArviZ's energy plot to check for any HMC pathology, and a trace plot for the hyperparameters.
    """)
    return


@app.cell(hide_code=True)
def _(swing_decision_trace):
    az.plot_energy(swing_decision_trace)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(swing_decision_trace):
    az.plot_trace(swing_decision_trace, var_names=["eta", "ell", "sigma", "intercept"])
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(ages_unique, swing_decision_trace, swing_decisions):
    _fig, _ax = plt.subplots(1, 1, figsize=(18, 10))

    _f = (
        (
            swing_decision_trace.posterior["intercept"]
            + swing_decision_trace.posterior["f"]
        )
        .sel(chain=0)
        .values
    )  # (draw, age)
    plot_gp_dist(ax=_ax, samples=_f, x=ages_unique)
    swing_decisions.plot.scatter(
        x="age", y="swing_decision", s=10, alpha=0.3, legend=False, ax=_ax
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### When NOT to reach for HSGP

    The success above is real, but HSGP carries restrictions worth knowing before you reach for it on the next problem:

    1. **Stationary kernels only.** HSGP works through the kernel's *spectral density*, which means it is compatible with any covariance class that implements `power_spectral_density` — the Matérn family, exponentiated quadratic, etc. Non-stationary or composite-non-stationary kernels are out of scope. The `Periodic` kernel is a special case handled by `pm.gp.HSGPPeriodic`.
    2. **Does not scale well with input dimension.** HSGP is excellent in 1-D (time series) and tolerable in 2-D (spatial point processes). Beyond three input dimensions, the basis-function count grows quickly and the approximation loses its computational edge.
    3. **May struggle with rapidly-varying processes.** If the function changes very quickly relative to the extent of the domain, you may need an impractically large $m$ to capture the high-frequency structure.
    4. **Small data: prefer the exact GP.** For a few hundred observations, exact `pm.gp.Marginal` is simple, fast enough, and removes one source of potential error.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Multi-input GPs

    We have already seen multi-output GPs in the context of the ICM model, where several related outputs are modeled simultaneously. Let's now consider multi-input Gaussian processes, where multiple variables are used to predict an outcome of interest. For example, a spatial process is often modeled as a GP, and the covariance function is chosen to reflect the spatial structure of the process. The spatial structure is often modeled using a covariance function that decreases with distance, such as the Matern covariance function.

    #### Example: Called strike probability

    An obvious baseball application is the prediction of pitch outcomes based on pitch location. We will build a simple called strike probability model.
    """)
    return


@app.cell(hide_code=True)
def _():
    pitch_data = pd.read_csv(data_path / "taken_pitches_walker.csv")
    called_strike = pitch_data[["bats", "location_x", "location_z", "is_strike"]].copy()
    called_strike.head()
    return (called_strike,)


@app.cell(hide_code=True)
def _(called_strike):
    _fig, _ax = plt.subplots(figsize=(5, 6))
    for _flag, _color, _label in [(0, "C0", "ball"), (1, "C1", "strike")]:
        _sub = called_strike[called_strike.is_strike == _flag]
        _ax.scatter(
            _sub.location_x, _sub.location_z, c=_color, s=10, label=_label, alpha=0.6
        )
    _ax.set_xlabel("location_x")
    _ax.set_ylabel("location_z")
    _ax.legend(title="is_strike")
    _rect = plt.Rectangle((-0.7, 1.4), 1.4, 2, edgecolor="black", facecolor="none")
    _ax.add_patch(_rect)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(called_strike):
    strike_X = called_strike[["location_x", "location_z"]].values
    strike_y = called_strike.is_strike.values.astype(int)
    strike_X.shape, strike_y.shape
    return strike_X, strike_y


@app.cell
def _(strike_X, strike_y):
    def build_called_strike_model():
        with pm.Model() as called_strike_model:
            ls = pm.Gamma("ls", alpha=2, beta=2, shape=2)
            eta = pm.HalfNormal("eta", sigma=2)
            cov = eta**2 * pm.gp.cov.Matern52(2, ls=ls)

            gp = pm.gp.HSGP(m=[25, 25], c=4.0, cov_func=cov)
            f = gp.prior("f", X=strike_X)

            strike_prob = pm.math.invlogit(f)
            pm.Bernoulli("strike", p=strike_prob, observed=strike_y)
        return called_strike_model, gp

    called_strike_model, called_strike_gp = build_called_strike_model()
    return called_strike_gp, called_strike_model


@app.cell
def _(called_strike_model):
    with called_strike_model:
        strike_trace = pm.sample(
            nuts_sampler="nutpie",
            chains=2,
            random_seed=RANDOM_SEED,
        )
    strike_trace
    return (strike_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A key step for generating a visualization is creating a 2D grid to predict over, which here we can use PyMC's `cartesian` function to do.
    """)
    return


@app.cell
def _(called_strike_gp, called_strike_model, strike_trace):
    x_pred_strike = np.linspace(-1.5, 1.5, 100)
    z_pred_strike = np.linspace(0.5, 4.5, 100)
    X_pred_strike = pm.math.cartesian(x_pred_strike[:, None], z_pred_strike[:, None])

    with called_strike_model:
        grid_pred = called_strike_gp.conditional("grid", X_pred_strike)
        pm.Deterministic("p_strike", pm.math.invlogit(grid_pred))
        strike_preds = pm.sample_posterior_predictive(
            strike_trace, var_names=["p_strike"]
        )
    return X_pred_strike, strike_preds


@app.cell(hide_code=True)
def _(X_pred_strike, strike_preds):
    plt.figure(figsize=(5, 6))
    plt.scatter(
        X_pred_strike[:, 0],
        X_pred_strike[:, 1],
        s=30,
        c=strike_preds.posterior_predictive["p_strike"]
        .mean(dim=("chain", "draw"))
        .to_numpy(),
        marker="s",
        cmap="coolwarm",
    )
    plt.colorbar()
    _rect = plt.Rectangle((-0.7, 1.4), 1.4, 2, edgecolor="black", facecolor="none")
    plt.gca().add_patch(_rect)
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Other GP Packages

    There are a variety of Python libraries aside from PyMC that implement Gaussian processes. Here are a few:

    - GPy
    - GPflow
    - GPyTorch
    - scikit-learn
    - PyStan
    - Tensorflow Probability

    For example, here is a GPyTorch implementation of a multi-output GP for estimating TrackMan biases in pitch velocity across venues. If you are familiar with Torch, the interface is the same.

    ```python
    class HadamardRandomEffectsModel(gpytorch.models.ExactGP):
        def __init__(self, train_time, train_pitcher_indices, train_venue_indices, train_targets,
                     num_pitchers, num_venues, likelihood):

            super().__init__(
                train_inputs=(train_time, train_pitcher_indices, train_venue_indices),
                train_targets=train_targets,
                likelihood=likelihood
            )

            # Parameters for pitcher effects
            self.mu = torch.nn.Parameter(torch.randn(1))
            self.tau = torch.nn.Parameter(torch.randn(1))

            self.pitcher_effects = gpytorch.distributions.MultivariateNormal(
                torch.ones(num_pitchers) * self.mu,
                gpytorch.lazy.DiagLazyTensor(torch.ones(num_pitchers) * self.tau.pow(2.)))

            # Parameters for venue trends
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.MaternKernel(nu=2.5),
                grid_size=100, num_dims=1)
            self.venue_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_venues, rank=1)
            self.trend_covar_module = gpytorch.kernels.RBFKernel()

        def forward(self, times, pitcher_indices, venue_indices):
            time_mean = self.mean_module(times)
            covar_x = self.covar_module(times)
            covar_v = self.venue_covar_module(venue_indices)
            covar_t = self.trend_covar_module(times)
            time_covar = (covar_x.mul(covar_v) + covar_t).evaluate_kernel()

            return gpytorch.distributions.MultivariateNormal(
                time_mean + self.pitcher_effects[pitcher_indices.squeeze(-1)],
                time_covar
            )

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = HadamardRandomEffectsModel(X, pitcher_ind, venue_ind, y, P, V, likelihood)
    ```

    Note that here we are not "fully Bayesian", as we are not placing priors on the parameters of the model. This is a common approach in machine learning, where the focus is on predictive performance rather than inference.

    But, the nice thing about using GPyTorch is that it is easy to fit the model using a fast GPU.

    ```python
    model = model.cuda()
    likelihood = likelihood.cuda()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Optimize!
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(X, pitcher_ind, venue_ind)
        loss = -mll(output, y)
        loss.backward()
        if not i % 10:
            print(f'Iter {i}/{training_iterations} - Loss: {loss.item()}')
        optimizer.step()
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Want to learn more?

    - [Rasmussen, C. E. & Williams, C. K. I. (2005). Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning series). The MIT Press.](http://www.amazon.com/books/dp/026218253X)
    - [Quinonero-Candela, J. & Rasmussen, C. E. (2005). A Unifying View of Sparse Approximate Gaussian Process Regression. Journal of Machine Learning Research 6, 1939–1959.](http://www.jmlr.org/papers/v6/quinonero-candela05a.html)
    - [Duvenaud, D. The Kernel Cookbook: Advice on Covariance functions](https://www.cs.toronto.edu/~duvenaud/cookbook/index.html)
    - [Riutort-Mayol, G. & Burkner, P. &  Andersen, M. & Solin, A. & Vehtari, A. (2022) Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic programming](https://arxiv.org/abs/2004.11408)
    """)
    return


if __name__ == "__main__":
    app.run()
