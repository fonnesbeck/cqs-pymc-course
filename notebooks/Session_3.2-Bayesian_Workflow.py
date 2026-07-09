import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
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

    # Bayesian workflow for golf putting

    We use a data set from "Statistics: A Bayesian Perspective" (Berry, 1996). The dataset describes the outcome of professional golfers putting from a number of distances, and is small enough that we can just print and load it inline, instead of doing any special `csv` reading.

    This marimo version runs in the workshop pixi environment; the original PyMC Examples notebook notes that `xarray-einstats` is an extra dependency.
    """)
    return


@app.cell(hide_code=True)
def _():
    import io

    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import pymc as pm
    import pytensor.tensor as pt
    import scipy
    from scipy import stats
    import xarray as xr

    from xarray_einstats.stats import XrContinuousRV

    return XrContinuousRV, az, io, np, pl, plt, pm, pt, scipy, stats, xr


@app.cell(hide_code=True)
def _(az):
    RANDOM_SEED = 8927
    az.style.use("arviz-variat")
    return (RANDOM_SEED,)


@app.cell(hide_code=True)
def _(io, pl):
    # golf putting data from berry (1996)
    golf_data = """distance tries successes
    2 1443 1346
    3 694 577
    4 455 337
    5 353 208
    6 272 149
    7 256 136
    8 240 111
    9 217 69
    10 200 67
    11 237 75
    12 202 52
    13 192 46
    14 174 54
    15 167 28
    16 201 27
    17 195 31
    18 191 33
    19 147 20
    20 152 24"""


    golf_data = pl.read_csv(io.StringIO(golf_data), separator=" ", schema_overrides={"distance": pl.Float64})

    BALL_RADIUS = (1.68 / 2) / 12
    CUP_RADIUS = (4.25 / 2) / 12
    return BALL_RADIUS, CUP_RADIUS, golf_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We start plotting the data to get a better idea of how it looks. The hidden cell contains the plotting code
    """)
    return


@app.cell(hide_code=True)
def _(plt, stats):
    # Cell tags: hide-input
    def plot_golf_data(golf_data, ax=None, color='k'):
        """Utility function to standardize a pretty plotting of the golf data."""
        if ax is None:
            _, ax = plt.subplots()
        bg_color = ax.get_facecolor()
        distance = golf_data["distance"].to_numpy()
        tries = golf_data["tries"].to_numpy()
        successes = golf_data["successes"].to_numpy()
        rv = stats.beta(successes, tries - successes)
        ax.vlines(distance, *rv.interval(0.68), label=None, color=color)
        ax.plot(distance, successes / tries, 'o', mec=color, mfc=bg_color, label=None)
        ax.set_xlabel('Distance from hole')
        ax.set_ylabel('Percent of putts made')
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(left=0)
        ax.grid(True, axis='y', alpha=0.7)
        return ax

    return (plot_golf_data,)


@app.cell(hide_code=True)
def _(golf_data, plot_golf_data):
    _ax = plot_golf_data(golf_data)
    _ax.set_title('Overview of data from Berry (1996)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After plotting, we see that generally golfers are less accurate from further away. Note that this data is pre-aggregated: we may be able to do more interesting work with granular putt-by-putt data. This data set appears to have been binned to the nearest foot.

    We might think about doing prediction with this data: fitting a curve to this data would allow us to make reasonable guesses at intermediate distances, as well as perhaps to extrapolate to longer distances.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logit model

    First we will fit a traditional logit-binomial model. We model the number of successes directly, with

    $$
    a, b \sim \mathcal{N}(0, 1) \\
    p(\text{success}) = \operatorname{logit}^{-1}(a \cdot \text{distance} + b) \\
    \text{num. successes} \sim \operatorname{Binomial}(\text{tries}, p(\text{success}))
    $$

    Here is how to write that model in PyMC. We wrap model construction in a function so repeated names like `distance`, `tries`, and `successes` stay local without resorting to underscore-prefixed variables. We also use `pm.Data` to let us swap out the data later, when we will work with a newer data set.
    """)
    return


@app.cell
def _(golf_data, pm):
    def build_logit_model(golf_data):
        with pm.Model() as model:
            distance = pm.Data("distance", golf_data["distance"].to_numpy(), dims="obs_id")
            tries = pm.Data("tries", golf_data["tries"].to_numpy(), dims="obs_id")
            successes = pm.Data(
                "successes", golf_data["successes"].to_numpy(), dims="obs_id"
            )
            a = pm.Normal("a")
            b = pm.Normal("b")
            pm.Binomial(
                "success",
                n=tries,
                p=pm.math.invlogit(a * distance + b),
                observed=successes,
                dims="obs_id",
            )
        return model


    logit_model = build_logit_model(golf_data)
    logit_model
    return (logit_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have some intuition that $a$ should be negative, and also that $b$ should be positive (since when $\text{distance} = 0$, we expect to make nearly 100% of putts). We are not putting that into the model, though. We are using this as a baseline, and we may as well wait and see if we need to add stronger priors.
    """)
    return


@app.cell
def _(RANDOM_SEED, az, logit_model, pm):
    with logit_model:
        logit_trace = pm.sample(1000, tune=1000, target_accept=0.9, random_seed=RANDOM_SEED)

    az.summary(logit_trace)
    return (logit_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see $a$ and $b$ have the signs we expected. There were no bad warnings emitted from the sampler. Looking at the summary, the number of effective samples is reasonable, and the rhat is close to 1. This is a small model, so we are not being too careful about inspecting the fit.

    We plot 50 posterior draws of $p(\text{success})$ along with the expected value. Also, we draw 500 points from the posterior predictive to plot:
    """)
    return


@app.cell
def _(RANDOM_SEED, logit_model, logit_trace, pm):
    # Draw posterior predictive samples
    with logit_model:
        pm.sample_posterior_predictive(logit_trace, extend_inferencedata=True, random_seed=RANDOM_SEED)
    return


@app.cell
def _(BALL_RADIUS, CUP_RADIUS, az, golf_data, logit_trace, np, scipy, xr):
    logit_post = az.extract(logit_trace, num_samples=400)
    # hard to plot more than 400 sensibly
    logit_ppc = az.extract(logit_trace, group='posterior_predictive', num_samples=400)
    _const_data = logit_trace['constant_data']
    logit_ppc_success = logit_ppc / _const_data['tries']
    _t_ary = np.linspace(CUP_RADIUS - BALL_RADIUS, golf_data["distance"].max(), 200)
    t = xr.DataArray(_t_ary, coords=[('distance', _t_ary)])
    logit_post['expit'] = scipy.special.expit(logit_post['a'] * t + logit_post['b'])
    return logit_post, logit_ppc_success, t


@app.cell(hide_code=True)
def _(golf_data, logit_post, logit_ppc_success, plot_golf_data, t):
    _ax = plot_golf_data(golf_data)
    _ax.plot(t, logit_post['expit'].T, lw=1, color='C1', alpha=0.5)
    _ax.plot(t, logit_post['expit'].mean(dim='sample'), color='C2')
    _ax.plot(golf_data["distance"].to_numpy(), logit_ppc_success, 'k.', alpha=0.01)
    _ax.set_title('Logit mean and posterior predictive')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The fit is ok, but not great! It is a good start for a baseline, and lets us answer curve-fitting type questions. We may not trust much extrapolation beyond the end of the data, especially given how the curve does not fit the last four values very well. For example, putts from 50 feet are expected to be made with probability:
    """)
    return


@app.cell
def _(logit_post, scipy):
    prob_at_50 = scipy.special.expit(logit_post["a"] * 50 + logit_post["b"]).mean(dim="sample").item()
    print(f"{100 * prob_at_50:.5f}%")
    return (prob_at_50,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Warning: summarize predictions, not parameters.**
    >
    > We might be tempted to reduce the posterior to a single fitted line by first
    > averaging the parameters `a` and `b`, then plugging those averages into the
    > 50-foot prediction:
    >
    > ```python
    > # Tempting, but wrong for posterior prediction
    > scipy.special.expit(
    >     logit_trace.posterior["a"].mean(dim=("chain", "draw")) * 50
    >     + logit_trace.posterior["b"].mean(dim=("chain", "draw"))
    > )
    > ```
    >
    > The calculation above does the Bayesian version instead: compute the 50-foot
    > make probability for every posterior draw, then average those probabilities.
    >
    > ```python
    > # Right: transform each posterior draw, then summarize
    > scipy.special.expit(
    >     logit_trace.posterior["a"] * 50 + logit_trace.posterior["b"]
    > ).mean(dim=("chain", "draw"))
    > ```
    >
    > These are not generally equal because the inverse-logit transform is nonlinear:
    >
    > $$
    > \mathbb{E}[f(\theta)] \ne f(\mathbb{E}[\theta]).
    > $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geometry-based model

    As a second pass at modelling this data, both to improve fit and to increase confidence in extrapolation, we think about the geometry of the situation. We suppose professional golfers can hit the ball in a certain direction, with some small(?) error. Specifically, the angle the ball actually travels is normally distributed around 0, with some variance that we will try to learn.

    Then the ball goes in whenever the error in angle is small enough that the ball still hits the cup. This is intuitively nice! A longer putt will admit a smaller error in angle, and so a lower success rate than for shorter putts.

    I am skipping a derivation of the probability of making a putt given the accuracy variance and distance to the hole, but it is a fun exercise in geometry, and turns out to be

    $$
    p(\text{success} | \sigma_{\text{angle}}, \text{distance}) = 2 \Phi\left( \frac{ \arcsin \left((R - r) / \text{distance}\right)}{\sigma_{\text{angle}}}\right) - 1,
    $$

    where $\Phi$ is the normal cumulative density function, $R$ is the radius of the cup (turns out 2.125 inches), and $r$ is the radius of the golf ball (around 0.84 inches).

    To get a feeling for this model, move the slider below and watch how the predicted putting curve changes as $\sigma_{\text{angle}}$ changes.
    """)
    return


@app.cell(hide_code=True)
def _(BALL_RADIUS, CUP_RADIUS, XrContinuousRV, np, stats):
    def forward_angle_model(variances_of_shot, t):
        norm_dist = XrContinuousRV(stats.norm, 0, variances_of_shot)
        return 2 * norm_dist.cdf(np.arcsin((CUP_RADIUS - BALL_RADIUS) / t)) - 1

    return (forward_angle_model,)


@app.cell(hide_code=True)
def _(mo):
    sigma_angle_slider = mo.ui.slider(
        start=0.005,
        stop=0.20,
        step=0.005,
        value=0.02,
        label="sigma_angle (radians)",
    )
    return (sigma_angle_slider,)


@app.cell(hide_code=True)
def _(mo, sigma_angle_slider):
    mo.vstack([
        mo.md("**Choose the angular standard deviation used in the geometry model**:"),
        mo.hstack(
            [
                sigma_angle_slider,
                mo.md(f"`{sigma_angle_slider.value:.3f}` radians"),
            ],
            justify="start",
            align="center",
            gap=1,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(forward_angle_model, golf_data, plot_golf_data, sigma_angle_slider, t):
    _ax = plot_golf_data(golf_data)
    sigma_angle = sigma_angle_slider.value
    predicted_success = forward_angle_model(sigma_angle, t)
    _ax.plot(t, predicted_success, color="C1", lw=2.5)
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This looks like a promising approach! A variance of 0.02 radians looks like it will be close to the right answer. The model also predicted that putts from 0 feet all go in, which is a nice side effect. We might think about whether a golfer misses putts symmetrically. It is plausible that a right handed putter and a left handed putter might have a different bias to their shots.
    ### Fitting the model

    PyMC has $\Phi$ implemented, but it is pretty hidden (`pm.distributions.dist_math.normal_lcdf`), and it is worthwhile to implement it ourselves anyways, using an identity with the [error function](https://en.wikipedia.org/wiki/Error_function).
    """)
    return


@app.cell
def _(BALL_RADIUS, CUP_RADIUS, golf_data, pm, pt):
    def phi(x):
        """Calculates the standard normal cumulative distribution function."""
        return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2.0))


    def build_angle_model(golf_data):
        with pm.Model() as model:
            distance = pm.Data("distance", golf_data["distance"].to_numpy(), dims="obs_id")
            tries = pm.Data("tries", golf_data["tries"].to_numpy(), dims="obs_id")
            successes = pm.Data(
                "successes", golf_data["successes"].to_numpy(), dims="obs_id"
            )
            variance_of_shot = pm.HalfNormal("variance_of_shot")
            p_goes_in = pm.Deterministic(
                "p_goes_in",
                2
                * phi(pt.arcsin((CUP_RADIUS - BALL_RADIUS) / distance) / variance_of_shot)
                - 1,
                dims="obs_id",
            )
            pm.Binomial(
                "success", n=tries, p=p_goes_in, observed=successes, dims="obs_id"
            )
        return model


    angle_model = build_angle_model(golf_data)
    angle_model
    return angle_model, phi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior Predictive Checks

    We often wish to sample from the prior, especially if we have some idea of what the observations would look like, but not a lot of intuition for the prior parameters. We have an angle-based model here, but it might not be intuitive if the *variance* of the angle is given, how that effects the accuracy of a shot. Let's check!

    Sometimes a custom visualization or dashboard is useful for a prior predictive check. Here, we plot our prior distribution of putts from 20 feet away.
    """)
    return


@app.cell
def _(RANDOM_SEED, angle_model, pm):
    with angle_model:
        angle_trace = pm.sample_prior_predictive(500, random_seed=RANDOM_SEED)
    return (angle_trace,)


@app.cell(hide_code=True)
def _(RANDOM_SEED, XrContinuousRV, angle_trace, np, plt, stats, xr):
    angle_prior = angle_trace.prior.ds.squeeze()
    _angle_of_shot = XrContinuousRV(stats.norm, 0, angle_prior['variance_of_shot']).rvs(random_state=RANDOM_SEED)
    _distance = 20
    _end_positions = xr.Dataset({'endx': _distance * np.cos(_angle_of_shot), 'endy': _distance * np.sin(_angle_of_shot)})
    _fig, _ax = plt.subplots()
    for draw in _end_positions['draw']:  # radians
        end = _end_positions.sel(draw=draw)  # feet
        _ax.plot([0, end['endx']], [0, end['endy']], 'k-o', lw=1, mfc='w', alpha=0.5)
    _ax.plot(0, 0, 'go', label='Start', mfc='g', ms=20)
    _ax.plot(_distance, 0, 'ro', label='Goal', mfc='r', ms=20)
    _ax.set_title(f'Prior distribution of putts from {_distance}ft away')
    _ax.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a little funny! Most obviously, it should probably be not this common to putt the ball *backwards*. This also leads us to worry that we are using a normal distribution to model an angle. The [von Mises](https://en.wikipedia.org/wiki/Von_Mises_distribution) distribution may be appropriate here. Also, the golfer needs to stand somewhere, so perhaps adding some bounds to the von Mises would be appropriate. We will find that this model learns from the data quite well, though, and these additions are not necessary.
    """)
    return


@app.cell
def _(RANDOM_SEED, angle_model, angle_trace, az, pm):
    with angle_model:
        angle_trace.update(pm.sample(1000, tune=1000, target_accept=0.85, random_seed=RANDOM_SEED))

    angle_post = az.extract(angle_trace)
    return (angle_post,)


@app.cell(hide_code=True)
def _(
    angle_post,
    forward_angle_model,
    golf_data,
    logit_post,
    plot_golf_data,
    t,
):
    _ax = plot_golf_data(golf_data)
    angle_post['expit'] = forward_angle_model(angle_post['variance_of_shot'], t)
    _ax.plot(t, angle_post['expit'][:, ::100], lw=1, color='C1', alpha=0.1)
    _ax.plot(t, angle_post['expit'].mean(dim='sample'), label='Geometry-based model')
    _ax.plot(t, logit_post['expit'].mean(dim='sample'), label='Logit-binomial model')
    _ax.set_title('Comparing the fit of geometry-based and logit-binomial model')
    _ax.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This new model appears to fit much better, and by modelling the geometry of the situation, we may have a bit more confidence in extrapolating the data. This model suggests that a 50 foot putt has much higher chance of going in:
    """)
    return


@app.cell
def _(angle_post, forward_angle_model, np, prob_at_50):
    angle_prob_at_50 = forward_angle_model(angle_post["variance_of_shot"], np.array([50]))
    print(f"{100 * angle_prob_at_50.mean().item():.2f}% vs {100 * prob_at_50:.5f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also recreate our prior predictive plot, giving us some confidence that the prior was not leading to unreasonable situations in the posterior distribution: the variance in angle is quite small!
    """)
    return


@app.cell(hide_code=True)
def _(RANDOM_SEED, XrContinuousRV, angle_post, np, plt, stats, xr):
    _angle_of_shot = XrContinuousRV(stats.norm, 0, angle_post['variance_of_shot']).rvs(random_state=RANDOM_SEED)
    _distance = 20
    _end_positions = xr.Dataset({'endx': _distance * np.cos(_angle_of_shot.data), 'endy': _distance * np.sin(_angle_of_shot.data)})  # radians
    _fig, _ax = plt.subplots()  # feet
    for _x, y in zip(_end_positions.endx, _end_positions.endy):
        _ax.plot([0, _x], [0, y], 'k-o', lw=1, mfc='w', alpha=0.5)
    _ax.plot(0, 0, 'go', label='Start', mfc='g', ms=20)
    _ax.plot(_distance, 0, 'ro', label='Goal', mfc='r', ms=20)
    _ax.set_title(f'Prior distribution of putts from {_distance}ft away')
    _ax.set_xlim(-21, 21)
    _ax.set_ylim(-21, 21)
    _ax.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## New Data!

    Mark Broadie used new summary data on putting to fit a new model. We will use this new data to refine our model:
    """)
    return


@app.cell(hide_code=True)
def _(io, pl):
    #  golf putting data from Broadie (2018)
    new_golf_data = """distance tries successes
    0.28 45198 45183
    0.97 183020 182899
    1.93 169503 168594
    2.92 113094 108953
    3.93 73855 64740
    4.94 53659 41106
    5.94 42991 28205
    6.95 37050 21334
    7.95 33275 16615
    8.95 30836 13503
    9.95 28637 11060
    10.95 26239 9032
    11.95 24636 7687
    12.95 22876 6432
    14.43 41267 9813
    16.43 35712 7196
    18.44 31573 5290
    20.44 28280 4086
    21.95 13238 1642
    24.39 46570 4767
    28.40 38422 2980
    32.39 31641 1996
    36.39 25604 1327
    40.37 20366 834
    44.38 15977 559
    48.37 11770 311
    52.36 8708 231
    57.25 8878 204
    63.23 5492 103
    69.18 3087 35
    75.19 1742 24"""

    new_golf_data = pl.read_csv(io.StringIO(new_golf_data), separator=" ")
    return (new_golf_data,)


@app.cell(hide_code=True)
def _(
    BALL_RADIUS,
    CUP_RADIUS,
    angle_trace,
    forward_angle_model,
    golf_data,
    new_golf_data,
    np,
    plot_golf_data,
    xr,
):
    _ax = plot_golf_data(new_golf_data)
    plot_golf_data(golf_data, ax=_ax, color='C1')
    _t_ary = np.linspace(CUP_RADIUS - BALL_RADIUS, new_golf_data["distance"].max(), 200)
    t_1 = xr.DataArray(_t_ary, coords=[('distance', _t_ary)])
    _ax.plot(t_1, forward_angle_model(angle_trace.posterior['variance_of_shot'], t_1).mean(('chain', 'draw')))
    _ax.set_title('Comparing the new data set to the old data set, and\nconsidering the old model fit')
    return (t_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This new data set represents ~200 times the number of putt attempts as the old data, and includes putts up to 75ft, compared to 20ft for the old data set. It also seems that the new data represents a different population from the old data: while the two have different bins, the new data suggests higher success for most data. This may be from a different method of collecting the data, or golfers improving in the intervening years.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting the model on the new data

    Since we think these may be two different populations, the easiest solution would be to refit our model. This goes worse than earlier: there are divergences, and it takes much longer to run. This may indicate a problem with the model: Andrew Gelman calls this the "folk theorem of statistical computing".
    """)
    return


@app.cell
def _(RANDOM_SEED, angle_model, new_golf_data, pm):
    with angle_model:
        pm.set_data(
            {
                "distance": new_golf_data["distance"].to_numpy(),
                "tries": new_golf_data["tries"].to_numpy(),
                "successes": new_golf_data["successes"].to_numpy(),
            }
        )
        new_angle_trace = pm.sample(1000, tune=1500, random_seed=RANDOM_SEED)
    return (new_angle_trace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:** As you will see in the plot below, this model fits the new data quite badly. In this case, all the divergences
    > and convergence warnings have no other solution than using a different model that can actually explain the data.
    """)
    return


@app.cell(hide_code=True)
def _(
    angle_post,
    az,
    forward_angle_model,
    golf_data,
    new_angle_trace,
    new_golf_data,
    plot_golf_data,
    t_1,
):
    _ax = plot_golf_data(new_golf_data)
    plot_golf_data(golf_data, ax=_ax, color='C1')
    new_angle_post = az.extract(new_angle_trace)
    _ax.plot(t_1, forward_angle_model(angle_post['variance_of_shot'], t_1).mean(dim='sample'), label='Trained on original data')
    _ax.plot(t_1, forward_angle_model(new_angle_post['variance_of_shot'], t_1).mean(dim='sample'), label='Trained on new data')
    _ax.set_title('Retraining the model on new data')
    _ax.legend()
    return (new_angle_post,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A model incorporating distance to hole

    We might assume that, in addition to putting in the right direction, a golfer may need to hit the ball the right distance. Specifically, we assume:

    1. If a putt goes short *or* more than 3 feet past the hole, it will not go in.
    2. The golfer's intended stopping point is 1 foot past the hole. This is a fixed model assumption, not a parameter learned from the data.
    3. The distance the ball goes, $u$, is distributed according to
    $$
    u \sim \mathcal{N}\left(\text{distance} + 1, \sigma_{\text{distance}} (\text{distance} + 1)\right),
    $$
    where we learn $\sigma_{\text{distance}}$.

    The 1-foot value determines where the distribution of roll distances is centered relative to the hole. The makeable distance window is
    $$
    \text{distance} < u < \text{distance} + 3.
    $$
    So if the mean roll distance is `distance + 1`, the model assumes the golfer aims inside that window, one foot beyond the hole. The unknown parameter $\sigma_{\text{distance}}$ controls how tightly actual roll distances cluster around that intended stopping point.

    In code this fixed offset is `AIM_PAST_HOLE`. Let $a$ be that fixed offset. It appears twice in the standardized probability calculation:
    $$
    P(\text{good distance}) =
    \Phi\left(\frac{3 - a}{(\text{distance} + a)\sigma_{\text{distance}}}\right)
    -
    \Phi\left(\frac{-a}{(\text{distance} + a)\sigma_{\text{distance}}}\right).
    $$

    The numerator shifts the lower and upper bounds of the makeable window around the assumed target. The denominator says distance control gets harder for longer putts because the standard deviation scales with the intended roll distance.

    Could we model the aim point instead of fixing it? Yes, but then we would need a prior for it. With only aggregate make/miss data, the aim point and distance-control variance can trade off against each other, so fixing the aim point keeps this example focused on learning the two variance terms.
    """)
    return


@app.cell
def _(BALL_RADIUS, CUP_RADIUS, new_golf_data, phi, pm, pt):
    AIM_PAST_HOLE = 1.0  # assumed intended stopping point, in feet past the hole
    DISTANCE_TOLERANCE = 3.0


    def build_distance_angle_model(new_golf_data):
        with pm.Model() as model:
            distance = pm.Data(
                "distance", new_golf_data["distance"].to_numpy(), dims="obs_id"
            )
            tries = pm.Data("tries", new_golf_data["tries"].to_numpy(), dims="obs_id")
            successes = pm.Data(
                "successes", new_golf_data["successes"].to_numpy(), dims="obs_id"
            )
            variance_of_shot = pm.HalfNormal("variance_of_shot")
            variance_of_distance = pm.HalfNormal("variance_of_distance")
            p_good_angle = pm.Deterministic(
                "p_good_angle",
                2
                * phi(pt.arcsin((CUP_RADIUS - BALL_RADIUS) / distance) / variance_of_shot)
                - 1,
                dims="obs_id",
            )
            p_good_distance = pm.Deterministic(
                "p_good_distance",
                phi(
                    (DISTANCE_TOLERANCE - AIM_PAST_HOLE)
                    / ((distance + AIM_PAST_HOLE) * variance_of_distance)
                )
                - phi(-AIM_PAST_HOLE / ((distance + AIM_PAST_HOLE) * variance_of_distance)),
                dims="obs_id",
            )
            pm.Binomial(
                "success",
                n=tries,
                p=p_good_angle * p_good_distance,
                observed=successes,
                dims="obs_id",
            )
        return model


    distance_angle_model = build_distance_angle_model(new_golf_data)
    distance_angle_model
    return AIM_PAST_HOLE, DISTANCE_TOLERANCE, distance_angle_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This model still has only 2 dimensions to fit because `AIM_PAST_HOLE` and `DISTANCE_TOLERANCE` are fixed assumptions. They are good candidates for sensitivity analysis: changing `AIM_PAST_HOLE` changes the center of the roll-distance distribution, and changing `DISTANCE_TOLERANCE` changes how far past the hole we still count as makeable. We might also think about adding explicit correlations: it is plausible that less control over angle would correspond to less control over distance, or that longer putts lead to more variance in the angle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting the distance angle model
    """)
    return


@app.cell
def _(RANDOM_SEED, distance_angle_model, pm):
    with distance_angle_model:
        distance_angle_trace = pm.sample(1000, tune=1000, target_accept=0.85,  random_seed=RANDOM_SEED)
    return (distance_angle_trace,)


@app.cell(hide_code=True)
def _(
    AIM_PAST_HOLE,
    BALL_RADIUS,
    CUP_RADIUS,
    DISTANCE_TOLERANCE,
    XrContinuousRV,
    az,
    distance_angle_trace,
    forward_angle_model,
    new_angle_post,
    new_golf_data,
    np,
    plot_golf_data,
    stats,
    t_1,
):
    def forward_distance_angle_model(variance_of_shot, variance_of_distance, t):
        rv = XrContinuousRV(stats.norm, 0, 1)
        angle_prob = 2 * rv.cdf(np.arcsin((CUP_RADIUS - BALL_RADIUS) / t) / variance_of_shot) - 1
        distance_prob_one = rv.cdf((DISTANCE_TOLERANCE - AIM_PAST_HOLE) / ((t + AIM_PAST_HOLE) * variance_of_distance))
        distance_prob_two = rv.cdf(-AIM_PAST_HOLE / ((t + AIM_PAST_HOLE) * variance_of_distance))
        distance_prob = distance_prob_one - distance_prob_two
        return angle_prob * distance_prob
    _ax = plot_golf_data(new_golf_data)
    distance_angle_post = az.extract(distance_angle_trace)
    _ax.plot(t_1, forward_angle_model(new_angle_post['variance_of_shot'], t_1).mean(dim='sample'), label='Just angle')
    _ax.plot(t_1, forward_distance_angle_model(distance_angle_post['variance_of_shot'], distance_angle_post['variance_of_distance'], t_1).mean(dim='sample'), label='Distance and angle')
    _ax.set_title('Comparing fits of models on new data')
    _ax.legend()
    return distance_angle_post, forward_distance_angle_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This new model looks better, and fit much more quickly with fewer sampling problems compared to the old model.There is some mismatch between 10 and 40 feet, but it seems generally good. We can come to this same conclusion by taking posterior predictive samples, and looking at the residuals. Here, we see that the fit is being driven by the first 4 bins, which contain ~40% of the data.
    """)
    return


@app.cell
def _(RANDOM_SEED, distance_angle_model, distance_angle_trace, pm):
    with distance_angle_model:
        pm.sample_posterior_predictive(distance_angle_trace, extend_inferencedata=True, random_seed=RANDOM_SEED)
    distance_angle_ppc_ready = True
    return


@app.cell(hide_code=True)
def _(distance_angle_trace, new_golf_data, plt):
    _const_data = distance_angle_trace.constant_data
    _pp = distance_angle_trace.posterior_predictive
    _residuals = 100 * ((_const_data['successes'] - _pp['success']) / _const_data['tries']).mean(('chain', 'draw'))
    _fig, _ax = plt.subplots()
    _ax.plot(new_golf_data["distance"].to_numpy(), _residuals, 'o-')
    _ax.axhline(y=0, linestyle='dashed', linewidth=1)
    _ax.set_xlabel('Distance from hole')
    _ax.set_ylabel('Absolute error in expected\npercent of success')
    _ax.set_title('Residuals of new model')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A new model

    It is reasonable to stop here, but to improve the fit *everywhere* we may want a different likelihood than the `Binomial`. The `Binomial` is fully pinned down by `n` and `p`, so the short-putt bins — with tens of thousands of attempts each — dominate the fit and leave no slack for the rest of the curve. The fix is to let each bin carry a little independent extra error.

    ### Exercise: An overdispersed likelihood

    Start from `build_distance_angle_model`, keep the geometry (`p = p_good_angle * p_good_distance`) exactly as-is, and change **only the likelihood** so each data point can deviate a bit more from the geometric prediction. Three ways to do it:

    1. **Reparametrized `Binomial` / Normal on counts.** Treat the success *count* as approximately Normal with mean `n·p` and variance `n·p·(1 − p)`, then add an error term that is independent of `n`.
    2. **`BetaBinomial`.** Replace `Binomial` with its overdispersed cousin (the per-bin error stays roughly proportional to `n`).
    3. **Normal on the proportion.** Model the observed proportion `successes / tries` as Normal with mean `p` and variance `p(1 − p)/n + ε²`. No custom distribution required.

    Re-fit on `new_golf_data`, then compare the residuals against the distance + angle `Binomial` model. Which approaches actually loosen the high-`n` bins?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Hint": mo.md(r"""
        Keep `variance_of_shot`, `variance_of_distance`, and the `p_good_angle * p_good_distance` product unchanged — only the **likelihood line** changes. Add one extra parameter and pick an approach:

        - **Counts as Normal (approach 1):** `dispersion = pm.HalfNormal("dispersion")`, then observe the *counts* with `pm.Normal(mu=tries * p, sigma=tries * pm.math.sqrt(p * (1 - p) / tries + dispersion**2), observed=successes)`. This is the count-scale twin of approach 3.
        - **BetaBinomial (approach 2):** add a concentration `kappa = pm.HalfNormal("kappa", 500)` and use `pm.BetaBinomial(alpha=p * kappa, beta=(1 - p) * kappa, n=tries, observed=successes)`. Large `kappa` ≈ `Binomial`; clip `p` to `(1e-6, 1 - 1e-6)` so the Beta parameters stay positive.
        - **Proportion as Normal (approach 3):** add a `pm.Data` holding `successes / tries`, then `pm.Normal(mu=p, sigma=pm.math.sqrt(p * (1 - p) / tries + dispersion**2), observed=obs_prop)`.

        Approaches 1 and 3 are algebraically the same model written on the count vs proportion scale, so they fit identically.
        """)
        }
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Solution": mo.md(r"""
        ```python
        # The geometry (p = p_good_angle * p_good_distance) is identical to
        # build_distance_angle_model; only the likelihood differs between approaches.

        def _geometry(distance, variance_of_shot, variance_of_distance):
            p_good_angle = (
                2 * phi(pt.arcsin((CUP_RADIUS - BALL_RADIUS) / distance) / variance_of_shot) - 1
            )
            p_good_distance = phi(
                (DISTANCE_TOLERANCE - AIM_PAST_HOLE) / ((distance + AIM_PAST_HOLE) * variance_of_distance)
            ) - phi(-AIM_PAST_HOLE / ((distance + AIM_PAST_HOLE) * variance_of_distance))
            return p_good_angle * p_good_distance
        # --- Approach 1: Normal on counts (per-trial error independent of n; same model as approach 3) ---
        def build_count_normal_model(golf_data):
            with pm.Model() as model:
                distance = pm.Data("distance", golf_data["distance"].to_numpy(), dims="obs_id")
                tries = pm.Data("tries", golf_data["tries"].to_numpy(), dims="obs_id")
                successes = pm.Data("successes", golf_data["successes"].to_numpy(), dims="obs_id")
                variance_of_shot = pm.HalfNormal("variance_of_shot")
                variance_of_distance = pm.HalfNormal("variance_of_distance")
                dispersion = pm.HalfNormal("dispersion")
                p = _geometry(distance, variance_of_shot, variance_of_distance)
                pm.Normal(
                    "success",
                    mu=tries * p,
                    sigma=tries * pm.math.sqrt(p * (1 - p) / tries + dispersion**2),
                    observed=successes,
                    dims="obs_id",
                )
            return model
        # --- Approach 2: BetaBinomial (overdispersed Binomial) ---
        def build_betabinomial_model(golf_data):
            with pm.Model() as model:
                distance = pm.Data("distance", golf_data["distance"].to_numpy(), dims="obs_id")
                tries = pm.Data("tries", golf_data["tries"].to_numpy(), dims="obs_id")
                successes = pm.Data("successes", golf_data["successes"].to_numpy(), dims="obs_id")
                variance_of_shot = pm.HalfNormal("variance_of_shot")
                variance_of_distance = pm.HalfNormal("variance_of_distance")
                kappa = pm.HalfNormal("kappa", 500)  # concentration; large kappa -> Binomial
                p = pm.math.clip(
                    _geometry(distance, variance_of_shot, variance_of_distance), 1e-6, 1 - 1e-6
                )
                pm.BetaBinomial(
                    "success",
                    alpha=p * kappa,
                    beta=(1 - p) * kappa,
                    n=tries,
                    observed=successes,
                    dims="obs_id",
                )
            return model
        # --- Approach 3: Normal on the proportion (the version carried forward below) ---
        def build_disp_distance_angle_model(golf_data):
            with pm.Model() as model:
                distance = pm.Data("distance", golf_data["distance"].to_numpy(), dims="obs_id")
                tries = pm.Data("tries", golf_data["tries"].to_numpy(), dims="obs_id")
                successes = pm.Data("successes", golf_data["successes"].to_numpy(), dims="obs_id")
                obs_prop = pm.Data(
                    "obs_prop", (golf_data["successes"] / golf_data["tries"]).to_numpy(), dims="obs_id"
                )
                variance_of_shot = pm.HalfNormal("variance_of_shot")
                variance_of_distance = pm.HalfNormal("variance_of_distance")
                dispersion = pm.HalfNormal("dispersion")
                p = _geometry(distance, variance_of_shot, variance_of_distance)
                pm.Normal(
                    "p_success",
                    mu=p,
                    sigma=pm.math.sqrt(p * (1 - p) / tries + dispersion**2),
                    observed=obs_prop,
                    dims="obs_id",
                )
            return model
        # Fit any of them on the new data, e.g. the proportion-Normal model carried forward:
        with build_disp_distance_angle_model(new_golf_data):
            disp_distance_angle_trace = pm.sample(1000, tune=1000, random_seed=RANDOM_SEED)
            pm.sample_posterior_predictive(
                disp_distance_angle_trace, extend_inferencedata=True, random_seed=RANDOM_SEED
            )
        ```

        Approaches 1 and 3 are the same model written on the count vs proportion scale, so they fit identically and both loosen the high-`n` bins (the dispersion `ε` adds error proportional to `n` on the count scale). A *fixed* additive count variance — the literal "error independent of `n`" — barely moves the huge-`n` bins, which is why it is not enough on its own. The `BetaBinomial` also adds genuine overdispersion. This notebook carries the proportion-Normal model forward in the cells below.
        """)
        }
    )
    return




@app.cell(hide_code=True)
def _(
    AIM_PAST_HOLE,
    BALL_RADIUS,
    CUP_RADIUS,
    DISTANCE_TOLERANCE,
    new_golf_data,
    phi,
    pm,
    pt,
):
    def build_disp_distance_angle_model(new_golf_data):
        with pm.Model() as model:
            distance = pm.Data(
                "distance", new_golf_data["distance"].to_numpy(), dims="obs_id"
            )
            tries = pm.Data("tries", new_golf_data["tries"].to_numpy(), dims="obs_id")
            successes = pm.Data(
                "successes", new_golf_data["successes"].to_numpy(), dims="obs_id"
            )
            obs_prop = pm.Data(
                "obs_prop",
                (new_golf_data["successes"] / new_golf_data["tries"]).to_numpy(),
                dims="obs_id",
            )
            variance_of_shot = pm.HalfNormal("variance_of_shot")
            variance_of_distance = pm.HalfNormal("variance_of_distance")
            dispersion = pm.HalfNormal("dispersion")
            p_good_angle = pm.Deterministic(
                "p_good_angle",
                2
                * phi(pt.arcsin((CUP_RADIUS - BALL_RADIUS) / distance) / variance_of_shot)
                - 1,
                dims="obs_id",
            )
            p_good_distance = pm.Deterministic(
                "p_good_distance",
                phi(
                    (DISTANCE_TOLERANCE - AIM_PAST_HOLE)
                    / ((distance + AIM_PAST_HOLE) * variance_of_distance)
                )
                - phi(-AIM_PAST_HOLE / ((distance + AIM_PAST_HOLE) * variance_of_distance)),
                dims="obs_id",
            )
            p = p_good_angle * p_good_distance
            pm.Normal(
                "p_success",
                mu=p,
                sigma=pt.sqrt(p * (1 - p) / tries + dispersion**2),
                observed=obs_prop,
                dims="obs_id",
            )
        return model


    disp_distance_angle_model = build_disp_distance_angle_model(new_golf_data)
    disp_distance_angle_model
    return (disp_distance_angle_model,)


@app.cell(hide_code=True)
def _(RANDOM_SEED, disp_distance_angle_model, pm):
    with disp_distance_angle_model:
        disp_distance_angle_trace = pm.sample(1000, tune=1000, random_seed=RANDOM_SEED)
        pm.sample_posterior_predictive(disp_distance_angle_trace, extend_inferencedata=True, random_seed=RANDOM_SEED)
    return (disp_distance_angle_trace,)


@app.cell(hide_code=True)
def _(
    az,
    disp_distance_angle_trace,
    distance_angle_post,
    forward_distance_angle_model,
    new_golf_data,
    plot_golf_data,
    t_1,
):
    _ax = plot_golf_data(new_golf_data, ax=None)
    disp_distance_angle_post = az.extract(disp_distance_angle_trace)
    _ax.plot(t_1, forward_distance_angle_model(distance_angle_post['variance_of_shot'], distance_angle_post['variance_of_distance'], t_1).mean(dim='sample'), label='Distance and angle')
    _ax.plot(t_1, forward_distance_angle_model(disp_distance_angle_post['variance_of_shot'], disp_distance_angle_post['variance_of_distance'], t_1).mean(dim='sample'), label='Dispersed model')
    _ax.set_title('Comparing dispersed model with binomial distance/angle model')
    _ax.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This new model does better between 10 and 30 feet, as we can also see using the residuals plot - note that this model does marginally worse for very short putts:
    """)
    return


@app.cell(hide_code=True)
def _(az, disp_distance_angle_trace, distance_angle_trace, new_golf_data, plt):
    _const_data = distance_angle_trace.constant_data
    old_pp = az.extract(distance_angle_trace, group='posterior_predictive')
    old_residuals = 100 * ((_const_data['successes'] - old_pp) / _const_data['tries']).mean(dim='sample')
    _pp = az.extract(disp_distance_angle_trace, group='posterior_predictive')
    _residuals = 100 * (_const_data['successes'] / _const_data['tries'] - _pp).mean(dim='sample')
    _fig, _ax = plt.subplots()
    _ax.plot(new_golf_data["distance"].to_numpy(), _residuals, label='Dispersed model')
    _ax.plot(new_golf_data["distance"].to_numpy(), old_residuals, label='Distance and angle model')
    _ax.legend()
    _ax.axhline(y=0, linestyle='dashed', linewidth=1)
    _ax.set_xlabel('Distance from hole')
    _ax.set_ylabel('Absolute error in expected\npercent of success')
    _ax.set_title('Residuals of dispersed model vs distance/angle model')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Beyond prediction

    We want to use Bayesian analysis because we care about quantifying uncertainty in our parameters. We have a beautiful geometric model that not only gives us predictions, but gives us posterior distributions over our parameters. We can use this to back out how where our putts may end up, if not in the hole!

    First, we can try to visualize how 20,000 putts from a professional golfer might look. We:

    1. Set the number of trials to 5
    2. For each *joint* posterior sample of `variance_of_shot` and `variance_of_distance`,
       draw an angle and a distance from normal distribution 5 times.
    3. Plot the point, unless it would have gone in the hole

    The function `simulate_from_distance`
    """)
    return


@app.cell(hide_code=True)
def _(
    AIM_PAST_HOLE,
    BALL_RADIUS,
    CUP_RADIUS,
    DISTANCE_TOLERANCE,
    XrContinuousRV,
    np,
    plt,
    stats,
    xr,
):
    def simulate_from_distance(trace, distance_to_hole, trials=5):
        _variance_of_shot = trace.posterior['variance_of_shot']
        _variance_of_distance = trace.posterior['variance_of_distance']
        theta = XrContinuousRV(stats.norm, 0, _variance_of_shot).rvs(size=trials, dims='trials')
        _distance = XrContinuousRV(stats.norm, distance_to_hole + AIM_PAST_HOLE, (distance_to_hole + AIM_PAST_HOLE) * _variance_of_distance).rvs(size=trials, dims='trials')
        final_position = xr.concat((_distance * np.cos(theta), _distance * np.sin(theta)), dim='axis').assign_coords(axis=['x', 'y'])
        made_it = np.abs(theta) < np.arcsin((CUP_RADIUS - BALL_RADIUS) / distance_to_hole)
        made_it = made_it * (final_position.sel(axis='x') > distance_to_hole) * (final_position.sel(axis='x') < distance_to_hole + DISTANCE_TOLERANCE)
        dims = [dim for dim in final_position.dims if dim != 'axis']
        final_position = final_position.where(~made_it).stack(idx=dims).dropna(dim='idx')
        total_simulations = made_it.size
        _, _ax = plt.subplots()
        _ax.plot(0, 0, 'k.', lw=1, mfc='black', ms=250 / distance_to_hole)
        _ax.plot(*final_position, '.', alpha=0.1, mfc='r', ms=250 / distance_to_hole, mew=0.5)
        _ax.plot(distance_to_hole, 0, 'ko', lw=1, mfc='black', ms=350 / distance_to_hole)
        _ax.set_facecolor('#e6ffdb')
        _ax.set_title(f'Final position of {total_simulations:,d} putts from {distance_to_hole}ft.\n({100 * made_it.mean().item():.1f}% made)')
        return _ax

    return (simulate_from_distance,)


@app.cell
def _(distance_angle_trace, simulate_from_distance):
    simulate_from_distance(distance_angle_trace, distance_to_hole=50)
    return


@app.cell
def _(distance_angle_trace, simulate_from_distance):
    simulate_from_distance(distance_angle_trace, distance_to_hole=7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can then use this to work out how many putts a player may need to take from a given distance. This can influence strategic decisions like trying to reach the green in fewer shots, which may lead to a longer first putt, vs. a more conservative approach. We do this by simulating putts until they have all gone in, updating each missed putt's absolute x/y position before the next shot.

    Note that this is again something we might check experimentally. In particular, a highly unscientific search around the internet finds claims that professionals only 3-putt from 20-25ft around 3% of the time. Our model puts the chance of 3 or more putts from 22.5 feet at 2.8%, which seems suspiciously good.
    """)
    return


@app.cell(hide_code=True)
def _(AIM_PAST_HOLE, BALL_RADIUS, CUP_RADIUS, DISTANCE_TOLERANCE, RANDOM_SEED, az, np):
    def expected_num_putts(trace, distance_to_hole, trials=100000, random_seed=RANDOM_SEED):
        rng = np.random.default_rng(random_seed)
        combined_trace = az.extract(trace)
        n_samples = combined_trace.sizes['sample']
        idxs = rng.integers(0, n_samples, trials)
        _variance_of_shot = combined_trace['variance_of_shot'].isel(sample=idxs).to_numpy()
        _variance_of_distance = combined_trace['variance_of_distance'].isel(sample=idxs).to_numpy()
        hole_x = distance_to_hole * np.ones(trials)
        hole_y = np.zeros(trials)
        ball_x = np.zeros(trials)
        ball_y = np.zeros(trials)
        n_shots = []
        while ball_x.size > 0:
            dx = hole_x - ball_x
            dy = hole_y - ball_y
            remaining_distance = np.sqrt(dx**2 + dy**2)
            ux = dx / remaining_distance
            uy = dy / remaining_distance
            theta = rng.normal(0, _variance_of_shot)
            roll_distance = rng.normal(
                remaining_distance + AIM_PAST_HOLE,
                (remaining_distance + AIM_PAST_HOLE) * _variance_of_distance,
            )
            forward_distance = roll_distance * np.cos(theta)
            final_x = ball_x + roll_distance * (np.cos(theta) * ux - np.sin(theta) * uy)
            final_y = ball_y + roll_distance * (np.sin(theta) * ux + np.cos(theta) * uy)
            angle_window = np.arcsin(
                (CUP_RADIUS - BALL_RADIUS)
                / remaining_distance.clip(min=CUP_RADIUS - BALL_RADIUS)
            )
            made_it = (
                (np.abs(theta) < angle_window)
                & (forward_distance > remaining_distance)
                & (forward_distance < remaining_distance + DISTANCE_TOLERANCE)
            )
            keep = ~made_it
            ball_x = final_x[keep]
            ball_y = final_y[keep]
            hole_x = hole_x[keep]
            hole_y = hole_y[keep]
            _variance_of_shot = _variance_of_shot[keep]
            _variance_of_distance = _variance_of_distance[keep]
            n_shots.append(made_it.sum())
        return np.array(n_shots) / trials

    return (expected_num_putts,)


@app.cell(hide_code=True)
def _(disp_distance_angle_trace, expected_num_putts, np, plt):
    distances = (10, 20, 40, 80)
    _fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10))
    for _distance, _ax in zip(distances, axes.ravel()):
        made = 100 * expected_num_putts(disp_distance_angle_trace, _distance)
        _x = np.arange(1, 1 + len(made), dtype=int)
        _ax.vlines(np.arange(1, 1 + len(made)), 0, made, linewidths=50)
        _ax.set_title(f'{_distance} feet')
        _ax.set_ylabel('Percent of attempts')
        _ax.set_xlabel('Number of putts')
    _ax.set_xticks(_x)
    _ax.set_ylim(0, 100)
    _ax.set_xlim(0, 5.6)
    _fig.suptitle('Simulated number of putts from\na few distances')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Berry, D. A. (1996). *Statistics: A Bayesian Perspective*.
    - Gelman, A. [Golf putting case study](https://mc-stan.org/users/documentation/case-studies/golf.html).
    - Broadie, M. (2018). Golf putting summary data used in the Stan case study.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## License notice

    This notebook is adapted from the PyMC Examples gallery and follows its MIT License terms. Cite the original PyMC Examples notebook and the Stan golf putting case study when reusing it.
    """)
    return


if __name__ == "__main__":
    app.run()
