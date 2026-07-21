import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


with app.setup:
    import marimo as mo
    import pathlib as _pathlib
    import inspect
    import warnings
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pymc_extras.statespace as pmss
    import pytensor.tensor as pt
    from pymc_extras.statespace import structural as st

    PROJECT_ROOT = _pathlib.Path(__file__).resolve().parents[0]
    DATA_DIR = PROJECT_ROOT / "data"
    IMAGE_DIR = _pathlib.Path(__file__).parent / "images"
    RANDOM_SEED = 42
    SEED = sum(map(ord, "Layoff Revenue ITS"))
    rng = np.random.default_rng(SEED)
    plt.rcParams["figure.figsize"] = (10, 4)
    data_path = DATA_DIR
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    def compile_component(component):
        """Draw unconditional trajectories from a structural component."""
        return pmss.compile_statespace(component.build(verbose=False))

    def eti_band(est):
        """Return (lower, upper) ETI bounds as ndarrays. Robust to whether
        `est` is a DataArray or a Dataset."""
        h = az.eti(est)
        if hasattr(h, "data_vars"):
            h = h[next(iter(h.data_vars))]
        arr = h.to_numpy() if hasattr(h, "to_numpy") else np.asarray(h)
        return arr[..., 0], arr[..., 1]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Session 5.1: State-Space Time Series

    | Content |
    |---|
    | Motivation: what state-space models are, what they cost, what you get |
    | Position & velocity: the smallest non-trivial SSM |
    | Structural "lego blocks": the components you'll actually use |
    | Case study: load the layoff series, build the structural model |
    | **Exercise**: wire up priors and sample |
    | Latent-state flavors and decomposition |
    | **Exercise**: forecast under a hypothetical second layoff |
    | Extension: modeling headcount jointly (restricted BVAR) |
    | Where to next |

    The first three rows tour the framework. The remainder is an
    **interrupted time series (ITS)** case study: a real application of
    the same machinery to a short monthly business series with one (or
    several) discrete events.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Motivation

    A **state-space model** (SSM) describes a system with two linear equations:

    $$
    \begin{aligned}
    x_t &= T\, x_{t-1} + R\, \varepsilon_t, \quad &\varepsilon_t &\sim N(0, Q) \\
    y_t &= Z\, x_t + \eta_t, \quad &\eta_t &\sim N(0, H)
    \end{aligned}
    $$

    - $x_t$ is a **latent state**: the hidden story of the system (trend,
      seasonal position, velocity, whatever you encode).
    - $T$ is the **transition**: how the latent state evolves between steps.
    - $\varepsilon_t$ are **shocks** with covariance $Q$: everything
      unmodeled (news, events, innovations) pushing the state around.
    - $R$ is the **selection** matrix: it picks which states the shocks
      enter (e.g., the level gets innovations but the slope is deterministic).
    - $y_t$ is the **data you actually see**; $Z$ is the **design** matrix
      mapping latent state to observation, and $\eta_t$ is measurement
      noise with covariance $H$.

    **The canonical picture.** A car drives down the road and you watch it
    with a noisy GPS. The latent state is (position, velocity). The car
    enters a tunnel and you lose GPS for ten minutes. Where was the car
    during the tunnel? The SSM tells you: the dynamics keep propagating
    the state forward whether you observe or not, and the estimate
    uncertainty widens, then tightens back up when the car pops out the
    other side. The Kalman filter is what does this inference.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _plot_tunnel(flavor="filtered"):
        # Synthetic 1D position+velocity trajectory with a "tunnel" gap
        # in observations. Forward Kalman gives predicted + filtered;
        # backward RTS pass gives smoothed.
        rng_local = np.random.default_rng(7)
        n = 100
        tunnel = (40, 65)
        sigma_v, sigma_obs = 0.05, 0.8

        true_v = np.zeros(n)
        true_x = np.zeros(n)
        true_v[0] = 0.5
        for tt in range(1, n):
            true_v[tt] = true_v[tt - 1] + sigma_v * rng_local.standard_normal()
            true_x[tt] = true_x[tt - 1] + true_v[tt - 1]

        obs = true_x + sigma_obs * rng_local.standard_normal(n)
        observed = np.ones(n, dtype=bool)
        observed[tunnel[0] : tunnel[1]] = False

        T = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.array([[0.0, 0.0], [0.0, sigma_v**2]])
        Z = np.array([[1.0, 0.0]])
        H = np.array([[sigma_obs**2]])

        # Forward pass: predicted (before update) and filtered (after update)
        x_pred = np.zeros((n, 2))
        P_pred = np.zeros((n, 2, 2))
        x_filt = np.zeros((n, 2))
        P_filt = np.zeros((n, 2, 2))
        x_filt[0] = [0.0, 0.5]
        P_filt[0] = np.diag([1.0, 0.25])
        x_pred[0] = x_filt[0]
        P_pred[0] = P_filt[0]

        for tt in range(1, n):
            x_pred[tt] = T @ x_filt[tt - 1]
            P_pred[tt] = T @ P_filt[tt - 1] @ T.T + Q
            if observed[tt]:
                innov = obs[tt] - (Z @ x_pred[tt])[0]
                S = (Z @ P_pred[tt] @ Z.T + H)[0, 0]
                K = (P_pred[tt] @ Z.T / S).flatten()
                x_filt[tt] = x_pred[tt] + K * innov
                P_filt[tt] = P_pred[tt] - np.outer(K, Z @ P_pred[tt])
            else:
                x_filt[tt], P_filt[tt] = x_pred[tt], P_pred[tt]

        # Backward RTS smoother
        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()
        for tt in range(n - 2, -1, -1):
            C = P_filt[tt] @ T.T @ np.linalg.inv(P_pred[tt + 1])
            x_smooth[tt] = x_filt[tt] + C @ (x_smooth[tt + 1] - x_pred[tt + 1])
            P_smooth[tt] = P_filt[tt] + C @ (P_smooth[tt + 1] - P_pred[tt + 1]) @ C.T

        flavors = {
            "predicted": (x_pred, P_pred, "C3"),
            "filtered": (x_filt, P_filt, "C0"),
            "smoothed": (x_smooth, P_smooth, "C2"),
        }
        x_arr, P_arr, color = flavors[flavor]
        std = np.sqrt(P_arr[:, 0, 0])
        ts = np.arange(n)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axvspan(
            tunnel[0], tunnel[1], color="gray", alpha=0.2, label="tunnel (no GPS)"
        )
        ax.plot(ts, true_x, color="black", lw=1, label="true position")
        ax.scatter(
            ts[observed],
            obs[observed],
            s=12,
            color="C3",
            alpha=0.6,
            label="noisy GPS",
        )
        ax.plot(ts, x_arr[:, 0], color=color, lw=1.2, label=f"{flavor} estimate")
        ax.fill_between(
            ts,
            x_arr[:, 0] - 2 * std,
            x_arr[:, 0] + 2 * std,
            alpha=0.25,
            color=color,
            label=f"{flavor} ±2σ",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("position")
        ax.set_title(f"Tunnel example — {flavor} estimate")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig

    _plot_tunnel()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Concretely, the matrices for this example are

    $$
    T = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad R = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad Z = \begin{bmatrix} 1 & 0 \end{bmatrix}
    $$

    so the system reads

    $$
    \begin{bmatrix} x_t \\ v_t \end{bmatrix} = T \begin{bmatrix} x_{t-1} \\ v_{t-1} \end{bmatrix} + R\, \varepsilon_t, \qquad y_t = Z \begin{bmatrix} x_t \\ v_t \end{bmatrix} + \eta_t.
    $$

    $T$ encodes "position increases by velocity, velocity persists"; $R$
    says only velocity gets shocked; $Z$ picks out position from the
    state because that's all the GPS reports. Inside the tunnel we just
    drop the measurement update for $\eta_t$: the dynamics keep
    propagating, with uncertainty growing every step.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **What this framework buys you**

    1. **Principled decomposition.** Stitch trend + seasonality + exogenous
       drivers + noise; pull them apart on the posterior to see what's
       contributing what.
    2. **Free missing-data handling.** NaN in the observation? The filter
       just skips the measurement update at that step and propagates state
       forward. No imputation, no special code.
    3. **Forecasts with calibrated uncertainty.** Same engine, different
       tense.
    4. **Scenario analysis.** Swap in a hypothetical future covariate path
       and see what the model says.
    5. **Component decomposition.** Trend, seasonality, and event
       effects are separate states, so each can be read off (with
       uncertainty) from the fitted model.

    **What it costs**

    - **Linearity.** Everything has to be a linear system in the latent
      state. Nonlinear dynamics are workable but more effort.
    - **Gaussian errors.** Both shocks and measurement noise are normal.
      Heavier-tailed alternatives exist but take you off the beaten path.
    - **Evenly spaced time.** The math assumes fixed $\Delta t$. You can
      relax this but it complicates the matrices.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The smallest non-trivial SSM: position and velocity

    Imagine we're tracking something that moves smoothly: the download
    rate of a game, the wishlist count, monthly revenue, whatever. What
    we see is noisy, but the underlying thing has *momentum*: if it was
    going up yesterday, it's probably still going up today.

    The simplest model that captures "smooth, momentum-driven" is two
    latent states: **position** $x_t$ and **velocity** $v_t$.

    $$
    \begin{aligned}
    x_t &= x_{t-1} + v_{t-1} \\
    v_t &= v_{t-1} + \eta_t
    \end{aligned}
    $$

    Position moves by whatever the current velocity is. Velocity drifts
    randomly (we don't know what forces push it around, so we model the
    changes as noise). We observe only position, and with measurement
    noise.

    ### Simulate it
    """)
    return


@app.cell
def _():
    def _simulate(T=200, start="2024-01-01", obs_sigma=1.0):
        dates = pd.date_range(start, periods=T, freq="D")
        v = np.zeros(T)
        x = np.zeros(T)
        v[0] = 0.5
        for tt in range(1, T):
            v[tt] = v[tt - 1] + 0.05 * rng.standard_normal()
            x[tt] = x[tt - 1] + v[tt - 1]
        truth = pd.DataFrame({"position": x, "velocity": v}, index=dates)
        obs = pd.Series(
            x + obs_sigma * rng.standard_normal(T),
            index=dates,
            name="position_obs",
        )
        return truth, obs

    pv_truth, pv_obs = _simulate(obs_sigma=5)
    return pv_obs, pv_truth


@app.cell(hide_code=True)
def _(pv_obs, pv_truth):
    def _plot_pv(truth, obs):
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        panels = [
            {
                "col": "position",
                "color": "black",
                "ylabel": "position",
                "show_obs": True,
            },
            {
                "col": "velocity",
                "color": "C0",
                "ylabel": "velocity (latent)",
                "show_obs": False,
            },
        ]
        for ax, spec in zip(fig.axes, panels):
            ax.plot(
                truth.index,
                truth[spec["col"]],
                color=spec["color"],
                label=f"true {spec['col']}",
            )
            if spec["show_obs"]:
                ax.scatter(
                    obs.index,
                    obs,
                    s=6,
                    color="C3",
                    alpha=0.5,
                    label="noisy observation",
                )
            ax.set_ylabel(spec["ylabel"])
            ax.legend()
        axes[-1].set_xlabel("date")
        fig.suptitle("Position & velocity — the smallest SSM")
        fig.tight_layout()
        return fig

    _plot_pv(pv_truth, pv_obs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Velocity is the latent thing we never see directly. Position we see,
    noisily. The question is: *given the noisy observations, can we
    recover both position and velocity?*

    ### Fit it

    This exact model is already a structural component:
    `st.LevelTrend(order=2)`. `order=2` means "level plus its first
    derivative": which is exactly (position, velocity).
    `innovations_order=[0, 1]` says the level itself is deterministic
    given velocity (no shock to the level equation), but velocity drifts.
    """)
    return


@app.cell
def _():
    pv_mod = st.LevelTrend(order=2, innovations_order=[0, 1], name="pv")
    pv_mod += st.MeasurementError(name="obs")
    pv_ss = pv_mod.build(verbose=True)
    pv_ss
    return (pv_ss,)


@app.cell
def _(pv_obs, pv_ss):
    with pm.Model(coords=pv_ss.coords) as pv_model:
        P0_diag_pv = pm.Gamma("P0_diag", alpha=2.0, beta=1.0, dims=("state",))
        pm.Deterministic("P0", pt.diag(P0_diag_pv), dims=("state", "state_aux"))

        pm.Normal("initial_pv", mu=[0.0, 0.5], sigma=[5.0, 0.5], dims=("state_pv",))
        pm.HalfNormal("sigma_pv", sigma=0.1, dims=("shock_pv",))
        pm.HalfNormal("sigma_obs", sigma=1.0)

        pv_ss.build_statespace_graph(pv_obs)
    return (pv_model,)


@app.cell
def _(pv_model):
    with pv_model:
        pv_trace = pm.sample(tune=500, draws=250)
    return (pv_trace,)


@app.cell
def _(pv_trace):
    az.plot_trace_dist(
        pv_trace,
        var_names=["sigma_pv", "sigma_obs"],
        backend="matplotlib",
    )
    return


@app.cell
def _(pv_trace, pv_ss):
    cond_pv = pv_ss.sample_conditional_posterior(
        pv_trace,
        compile_kwargs={"mode": "NUMBA"},
    )
    comp_pv = pv_ss.extract_components_from_trace(cond_pv)
    comp_pv
    return (comp_pv,)


@app.cell(hide_code=True)
def _(comp_pv, pv_trace, pv_obs, pv_truth):
    def _plot_recovery(truth, obs, comp, trace):

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        panels = [
            {
                "state": "pv[level]",
                "truth_col": "position",
                "color": "C0",
                "show_obs": True,
            },
            {
                "state": "pv[trend]",
                "truth_col": "velocity",
                "color": "C1",
                "show_obs": False,
            },
        ]
        for ax, spec in zip(fig.axes, panels):
            est = comp.smoothed_posterior.sel(state=spec["state"])
            ax.plot(
                truth.index,
                truth[spec["truth_col"]],
                color="black",
                lw=1,
                label="true",
            )
            ax.plot(
                truth.index,
                est.mean(("chain", "draw")),
                color=spec["color"],
                label="posterior mean",
            )
            ax.fill_between(
                truth.index, *eti_band(est), alpha=0.25, color=spec["color"]
            )
            if spec["show_obs"]:
                ax.scatter(
                    obs.index,
                    obs,
                    s=6,
                    color="C3",
                    alpha=0.4,
                    label="observations",
                )
            ax.set_ylabel(spec["truth_col"])
            ax.legend(loc="upper left")
        axes[-1].set_xlabel("date")
        fig.suptitle("Recovered latent states from noisy observations of position only")
        fig.tight_layout()
        return fig

    _plot_recovery(pv_truth, pv_obs, comp_pv, pv_trace)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The velocity was never observed, and we recovered it. That's the whole
    point of filtering: the dynamics (position moves by velocity, velocity
    drifts) let us infer the hidden piece from the observable one. Every
    structural model in this workshop is some elaboration on this idea.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Structural components — the lego blocks

    `pymc_extras.statespace.structural` exposes a handful of components
    you'll reach for almost always. Each is a tiny SSM on its own; you
    combine them with `+` and call `.build()`. This section tours them.

    - **`LevelTrend`**: level and its derivatives (trend, curvature).
    - **`TimeSeasonality`**: repeating pattern in the time domain.
    - **`Cycle`**: long, damped oscillation with estimable period.
    - **`Autoregressive`**: serially correlated errors.
    - **`Regression`**: exogenous drivers (used heavily in the structural model below).
    - **`MeasurementError`**: observation noise (must be combined with
      at least one dynamic component).

    ### LevelTrend

    The general-purpose trend builder. Two knobs:

    - **`order`**: how many derivatives the latent state tracks.
      `order=1` is just a level; `order=2` adds a slope (think position +
      velocity, as in the earlier position+velocity example); `order=3` adds curvature.
    - **`innovations_order`**: which of those derivatives are *noisy*
      (i.e., receive a shock at every step). Pass a list of `0/1` flags,
      one per derivative: `[1, 0]` means "level is noisy, slope is a
      constant"; `[0, 1]` means "level is deterministic given slope,
      slope drifts". A bare integer `n` is shorthand for "the first `n`
      derivatives are noisy".

    A few common shapes:

    - `order=1, innovations_order=1`: single noisy level → random walk.
    - `order=2, innovations_order=1`: level noisy, slope constant →
      random walk with a fixed drift.
    - `order=2, innovations_order=[0, 1]`: level deterministic, slope
      wanders → a smooth trend. (The layoff case study instead uses
      `innovations_order=1`: noisy level, fixed drift.)
    """)
    return


@app.cell
def _():
    # Three flavors of LevelTrend:
    lt_grw = st.LevelTrend(order=1, innovations_order=1, name="grw")
    lt_rwd = st.LevelTrend(order=2, innovations_order=1, name="rwd")
    lt_llt = st.LevelTrend(order=2, innovations_order=[0, 1], name="llt")
    return lt_grw, lt_llt, lt_rwd


@app.cell(hide_code=True)
def _(lt_grw, lt_llt, lt_rwd):
    def _plot_lt_variants():
        variants = [
            (
                lt_grw,
                {"initial_grw": np.zeros(1), "sigma_grw": np.array([0.5])},
                "order=1, innov=1\n(random walk)",
            ),
            (
                lt_rwd,
                {
                    "initial_rwd": np.array([0.0, 0.1]),
                    "sigma_rwd": np.array([0.3]),
                },
                "order=2, innov=1\n(random walk + drift)",
            ),
            (
                lt_llt,
                {
                    "initial_llt": np.array([0.0, 0.0]),
                    "sigma_llt": np.array([0.05]),
                },
                "order=2, innov=[0,1]\n(smooth trend, drifting slope)",
            ),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        for ax, (component, params, title) in zip(fig.axes, variants):
            f = compile_component(component)
            _, ys = f(**params, steps=100, draws=10)
            ax.plot(ys.T)
            ax.set_title(title)
        fig.tight_layout()
        return fig

    _plot_lt_variants()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### TimeSeasonality

    A repeating pattern over `season_length`. The hidden states are the
    per-period levels (one per phase of the cycle, e.g. Mon..Sun for
    `season_length=7`); they're constrained to sum to zero so the
    seasonal component doesn't fight the trend for the overall level.

    - `innovations=False` (default), the pattern is rigid: whatever the
      seven phase values are at $t=0$, they stay the same forever.
    - `innovations=True`: a small shock perturbs the state at every
      step. Because the state rotates through one phase per step, what
      you actually *see* is each phase drifting cycle-to-cycle: this
      week's Monday-effect can morph slowly away from last year's.
      The shocks accumulate per step but the visible drift lands
      per cycle. Toggle the create cell below to `innovations=True` to
      watch the Mon/Tue/... peaks shift in height across weeks instead
      of repeating identically.
    """)
    return


@app.cell(hide_code=True)
def _():
    weekly_innovations = mo.ui.switch(value=False, label="innovations")
    weekly_innovations
    return (weekly_innovations,)


@app.cell
def _(weekly_innovations):
    weekly = st.TimeSeasonality(
        season_length=7, innovations=weekly_innovations.value, name="weekly"
    )
    return (weekly,)


@app.cell(hide_code=True)
def _(weekly):
    def _plot_weekly(component):
        L = component.season_length
        f = compile_component(component)
        kwargs = {"params_weekly": rng.normal(size=L - 1), "steps": 60}
        # Add the innovation sigma only when the component was created with
        # innovations=True (otherwise the compiled fn doesn't accept it).
        if getattr(component, "innovations", False):
            kwargs["sigma_weekly"] = 0.2
        _, ys = f(**kwargs)
        ys = np.asarray(ys).reshape(-1)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        n = len(ys)
        ax.plot(
            range(L),
            ys[:L],
            color="C0",
            lw=2,
            label=f"first period ({L} steps)",
        )
        # Overlap one point at index L-1 to keep the line continuous.
        ax.plot(
            range(L - 1, n),
            ys[L - 1 :],
            color="black",
            lw=1,
            label="repeating",
        )
        innov_note = ", innovations on" if "sigma_weekly" in kwargs else ""
        ax.set_title(f"TimeSeasonality(season_length={L}){innov_note}")
        ax.set_xlabel("day")
        ax.legend(loc="upper right")
        return fig

    _plot_weekly(weekly)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### FrequencySeasonality — the Fourier alternative

    `TimeSeasonality` carries one state per period (11 states for monthly data):
    interpretable, but bulky. `FrequencySeasonality` instead models the seasonal
    pattern in the frequency domain with `2n` states: `n` pairs of Fourier
    coefficients (a sine and cosine amplitude per harmonic). With `n=1` the
    pattern is a pure sine wave; at `n = season_length // 2` it can represent any
    periodic shape. Two things you buy with it: a compact state space for long
    seasonal periods, and support for non-integer season lengths (e.g. 365.25).
    The price is interpretability: the states are Fourier coefficients, so there
    is no isolable "January effect". The layoff case study below uses
    `FrequencySeasonality(season_length=12, n=1)`: a smooth annual cycle
    parameterized by just 2 coefficients.
    """)
    return


@app.cell
def _():
    freq_seasonal = st.FrequencySeasonality(season_length=12, n=1, name="annual")
    freq_seasonal.param_info
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Cycle

    Long, often-damped oscillation. Great for macro-like business
    cycles where you don't know the exact period: let the model
    estimate it.

    Three structural flags:

    - `estimate_cycle_length=True`: the period (steps per cycle) is a
      free parameter named `length_cyc`. Set `False` if you know the
      period a priori.
    - `dampen=True`: the oscillation amplitude decays each step by
      `dampening_factor_cyc` (so old shocks die out). Set `False` for a
      sustained oscillation.
    - `innovations=True`: adds a shock at every step; `False` makes
      the cycle deterministic given its initial state.

    `params_cyc` is the cycle's initial state: two numbers
    parameterizing the starting phase and amplitude of the oscillator.
    The plot below draws four cycles with random `length_cyc` between
    20 and 60 steps, all damped at `0.97` per step.
    """)
    return


@app.cell(hide_code=True)
def _():
    cyc_estimate_length = mo.ui.switch(value=True, label="estimate_cycle_length")
    cyc_dampen = mo.ui.switch(value=True, label="dampen")
    cyc_innovations = mo.ui.switch(value=False, label="innovations")
    mo.hstack([cyc_estimate_length, cyc_dampen, cyc_innovations])
    return cyc_dampen, cyc_estimate_length, cyc_innovations


@app.cell
def _(cyc_dampen, cyc_estimate_length, cyc_innovations):
    cyc_kwargs = dict(
        name="cyc",
        estimate_cycle_length=cyc_estimate_length.value,
        dampen=cyc_dampen.value,
        innovations=cyc_innovations.value,
    )
    # cycle_length is required only when estimate_cycle_length=False
    if not cyc_estimate_length.value:
        cyc_kwargs["cycle_length"] = 40
    cyc = st.Cycle(**cyc_kwargs)
    return (cyc,)


@app.cell(hide_code=True)
def _(cyc):
    def _plot_cycle(component):
        f = compile_component(component)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        # Multiple curves only make sense when something varies across them.
        n_draws = (
            4
            if getattr(component, "estimate_cycle_length", False)
            or getattr(component, "innovations", False)
            else 1
        )
        for _ in range(n_draws):
            kwargs = {"params_cyc": np.array([1.0, 0.0]), "steps": 200}
            if getattr(component, "estimate_cycle_length", False):
                kwargs["length_cyc"] = rng.uniform(20, 60)
            if getattr(component, "dampen", False):
                kwargs["dampening_factor_cyc"] = 0.97
            if getattr(component, "innovations", False):
                kwargs["sigma_cyc"] = 0.3
            _, ys = f(**kwargs)
            ax.plot(ys)
        bits = [
            "estimated length"
            if getattr(component, "estimate_cycle_length", False)
            else "fixed length=40",
            "dampened (0.97)" if getattr(component, "dampen", False) else "undampened",
        ]
        if getattr(component, "innovations", False):
            bits.append("innovations on")
        ax.set_title("Cycle — " + ", ".join(bits))
        return fig

    _plot_cycle(cyc)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Autoregressive

    Serially correlated errors: "yesterday's residual predicts
    today's". A useful catch-all for short-term persistence the other
    components don't capture.

    - `order`: number of lags. `AR(1)` depends on $y_{t-1}$ alone;
      `AR(2)` on $y_{t-1}$ and $y_{t-2}$; and so on.
    - `params_ar`: one coefficient per lag. The example below uses
      $\rho = 0.9$, meaning each step keeps 90% of the previous value
      plus a fresh innovation. Keep the coefficients in the stationary
      region (for `AR(1)`, $|\rho| < 1$) or the process blows up.
    - `sigma_ar`: standard deviation of the innovation term.
    """)
    return


@app.cell(hide_code=True)
def _():
    ar_order = mo.ui.slider(
        start=1, stop=4, value=1, step=1, label="order", show_value=True
    )
    ar_order
    return (ar_order,)


@app.cell
def _(ar_order):
    ar = st.Autoregressive(name="ar", order=ar_order.value)
    return (ar,)


@app.cell(hide_code=True)
def _(ar):
    def _plot_ar(component):
        order = len(component.order)
        # Stationary recipe that scales with order: heavy first lag, small
        # remainder. Sum stays well below 1 for orders up to ~4.
        print(order)
        coefs = np.zeros(order)
        coefs[0] = 0.7
        if order > 1:
            coefs[1:] = 0.1 / (order - 1)
        f = compile_component(component)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        _, ys = f(params_ar=coefs, sigma_ar=0.2, steps=200, draws=3)
        ax.plot(ys.T)
        coef_str = ", ".join(f"{c:.2f}" for c in coefs)
        ax.set_title(f"AR({order}) — params_ar=[{coef_str}], sigma_ar=0.2")
        return fig

    _plot_ar(ar)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Regression

    Attaches exogenous drivers: covariates the model doesn't predict
    but reads in at every step. The "states" of a Regression component
    are the *regression coefficients*; `state_names` labels them.

    - `state_names`: one label per covariate column. The compiled
      model will expect a `data_<name>` array of shape
      `(time, len(state_names))`.
    - `innovations`: when `True`, the regression coefficients drift
      over time (a time-varying-parameter regression). The default
      `False` keeps them constant.

    We use a Regression heavily in the structural-model section to attach the two layoff
    indicators (`layoff_active`, `layoff_decay`) as drivers of revenue.
    """)
    return


@app.cell
def _():
    reg_demo = st.Regression(
        state_names=["x1", "x2"], innovations=False, name="reg_demo"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### MeasurementError

    Observation noise. It can't stand alone: it needs at least one
    dynamic component to attach to, but you almost always want it. It
    buys numerical stability in the filter and reflects the honest fact
    that your data has measurement error.

    The only constructor knob is `name`; the noise standard deviation
    (`sigma_<name>`) is registered as a parameter for you to put a
    prior on inside `pm.Model`.
    """)
    return


@app.cell
def _():
    noise_demo = st.MeasurementError(name="noise_demo")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Composing components

    You combine with `+`. The smallest realistic model has a trend, an
    exogenous regressor, and observation noise: exactly the recipe the
    layoff case study uses:
    """)
    return


@app.cell
def _():
    composed_demo = (
        st.LevelTrend(order=2, innovations_order=[0, 1], name="lvl")
        + st.Regression(state_names=["x1"], name="reg")
        + st.MeasurementError(name="noise")
    )
    composed_ss_demo = composed_demo.build()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The resulting `ss_mod` object exposes `coords` (for your
    `pm.Model`) and `param_dims` (for sizing priors), and the methods
    `build_statespace_graph`, `sample_conditional_posterior`,
    `forecast`, and `extract_components_from_trace`. That's the whole
    API. Everything in the case study onward is one specific instance of this
    recipe with `Regression(state_names=["active", "decay"])`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Case study: monthly revenue around a layoff

    A case study in **interrupted time series (ITS)**: using
    `pymc_extras.statespace` to answer the question *"how much revenue
    did this layoff cost us, and for how long?"* given a short monthly
    revenue series with one (or several) discrete events.

    **The setup.** A client has ~3 years of monthly revenue for a single
    account. Partway through the series, the client announced a layoff.
    We want a posterior over:

    1. The immediate revenue hit from the event.
    2. The recovery profile over the months after.
    3. A **decomposition** of the series into trend, seasonal, and event components.

    **Why a state-space model.** We have very few observations, a short
    pre-event period (left-truncation), and honest measurement noise.
    The structural-time-series framework handles all three cleanly:

    - A drifting trend (`LevelTrend`) absorbs unmodeled macro movement.
    - A `Regression` component attaches the layoff signal (step + decay
      kernel) as exogenous drivers.
    - The Kalman filter's initial-state prior makes left-truncation an
      explicit modeling choice.
    - Reading off the latent level alone isolates the underlying trend
      with calibrated uncertainty.

    **A note on left truncation.** We observe 36 months, but the account
    existed before that; we just don't have the history. With only ~17
    pre-event observations, the filter's initial-state prior
    $(x_0, P_0)$ isn't washed out by data the way it would be with ten
    years of history; it shows up directly in the early level estimates
    and, downstream, in the component decomposition. The structural framework
    makes the choice explicit rather than implicit: `initial_level` is a
    prior on the latent state at $t=0$ (where we think the level and
    slope started), and $P_0$ is a prior on how *uncertain* we are about
    that starting point. A diffuse $P_0$ tells the filter "update
    aggressively from the first observations"; a tight $P_0$ anchors
    hard. We pick something in between (`Gamma(2, 2)` on the diagonal,
    mean 1, moderate spread) so the early posterior is mostly
    data-driven without being wildly volatile. This is the "right" way
    to handle a short pre-event window: not by pretending the data
    starts at the beginning of time, but by encoding the uncertainty
    about where it actually starts.


    ### Load the data
    """)
    return


@app.cell(hide_code=True)
def _():
    df = pd.read_csv(DATA_DIR / "layoff_single.csv", parse_dates=["date"]).set_index(
        "date"
    )
    df
    return (df,)


@app.cell(hide_code=True)
def _(df):
    def _plot_single(df):
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 7))
        axes[0].plot(df.index, df["log_revenue"], color="C0", marker="o", lw=1)
        axes[0].set_ylabel("log(monthly revenue)")
        axes[1].plot(df.index, df["log_headcount"], color="C2", marker="o", lw=1)
        axes[1].set_ylabel("log(headcount)")
        axes[2].step(df.index, df["layoff_active"], where="post", color="C3")
        axes[2].set_ylabel("layoff_active")
        axes[2].set_ylim(-0.2, 1.2)
        axes[3].plot(df.index, df["layoff_decay"], color="C3")
        axes[3].set_ylabel("layoff_decay")
        axes[-1].set_xlabel("month")
        fig.suptitle("Single-layoff scenario — 36 months")
        fig.tight_layout()
        return fig

    _plot_single(df)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Reading the plot:

    - **`log_revenue`**: small upward drift through the first
      ~17 months, a drop coinciding with (or just after) the event,
      and a gradual recovery back toward the pre-event level over the
      following year.
    - **`log_headcount`**: the parallel HR series. The layoff is
      much sharper here: a one-step drop, then slow rebuild via
      re-hiring. The univariate case study ignores this column; the BVAR
      extension promotes it to a co-modeled state in a restricted BVAR.
    - **`layoff_active`**: step indicator. 0 before the event, 1
      afterwards. A flexible basis function for any *persistent*
      component of the revenue effect.
    - **`layoff_decay`**: exp-decay kernel that turns on at the
      event and fades over months. A flexible basis for any
      *transient* component on top.

    `layoff_active` and `layoff_decay` are derived features, not part
    of the data-generating process: the actual mechanism is that
    layoffs hit headcount, and revenue follows lagged headcount. The
    univariate case study uses the two basis functions as a Regression-style
    approximation; the BVAR extension throws them out and models the mechanism directly.

    Both observation series have a couple of NaN months (revenue
    months 5–6, headcount months 21–23). They're deliberate: the
    framework's "free missing-data handling" promise from the motivation
    only means anything if the data actually has gaps. The univariate case
    study sails through the revenue gaps without special handling; the
    BVAR extension will do the same for the headcount gaps when it promotes
    that column to an observation in the joint model.

    The model's job in this section is to disentangle the three revenue
    signals (level, active step, decay) plus measurement noise: what
    shape does the layoff response take, and how does the underlying
    trend evolve once the event contributions are separated out?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build the structural model

    Four components:

    - `LevelTrend(order=2, innovations_order=1)`: a latent level with a
      fixed drift: the level
      accumulates noise like a random walk while the slope stays constant,
      the "random walk with drift" configuration from the tour.
    - `FrequencySeasonality(12, n=1)`: a smooth annual cycle (see the tour).
    - `Regression(state_names=["layoff_active", "layoff_decay"])`: the
      two layoff regressors as exogenous drivers. The regression
      coefficients are what we actually want to interpret.
    - `MeasurementError`: observation noise.

    Building the statespace returns a container that tells us which
    PyMC priors the model expects.
    """)
    return


@app.cell
def _():
    def build_model():
        level = st.LevelTrend(order=2, innovations_order=1, name="level")
        seasonal = st.FrequencySeasonality(
            season_length=12, n=1, innovations=False, name="seasonal"
        )
        reg = st.Regression(
            name="layoff",
            state_names=["active", "decay"],
            innovations=False,
        )
        noise = st.MeasurementError(name="noise")
        return (level + seasonal + reg + noise).build()

    ss = build_model()
    return (ss,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: Wire up priors and sample

    `ss` is built. Wrap it in a `pm.Model` with a prior on every
    parameter the SSM expects, then sample. Wrap the lot in a
    `fit(ss, df, ...)` function: the later forecasting exercise reuses
    it.

    Inspect `ss.param_dims` to see which priors are required and
    what dimensions each one needs. You'll also want
    `pm.Data("data_layoff", ...)` so `pm.set_data` can swap the
    regressors later. Then call `ss.build_statespace_graph(df["log_revenue"])`
    and `pm.sample()`.

    Center the `initial_level` prior on the scale of the observed data: the
    log-revenue series sits around 14.5, so `mu=[14.5, 0.0]` says "start near the
    data, with no prior slope."
    """)
    return


@app.cell
def _(df, ss):
    def exercise_fit():
        def fit(ss, df):
            with pm.Model(coords=ss.coords) as model:
                pm.Data(
                    "data_layoff",
                    df[["layoff_active", "layoff_decay"]].to_numpy(),
                    dims=["time", "state_layoff"],
                )
                # YOUR CODE HERE — priors for every parameter the SSM expects
                # (inspect ss.param_dims):
                #   initial_level   — initial [level, slope] of the latent trend
                #   sigma_level     — std of monthly level innovations
                #   beta_layoff     — regression coefficients on [active, decay]
                #   params_seasonal — 2 Fourier coefficients for the annual cycle
                #   sigma_noise     — observation noise std
                #   P0_diag (Gamma) + P0 = pt.diag(P0_diag) — initial state cov
                ...

                ss.build_statespace_graph(df["log_revenue"])
                trace = pm.sample(draws=250, tune=500)
            return model, trace

        _, my_trace = fit(ss, df)
        return my_trace

    return (exercise_fit,)


@app.cell(hide_code=True)
def _():
    run_statespace_fit = mo.ui.run_button(label="▶ Run exercise")
    run_statespace_fit
    return (run_statespace_fit,)


@app.cell(hide_code=True)
def _(exercise_fit, run_statespace_fit):
    mo.stop(
        not run_statespace_fit.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_fit()
    return


@app.cell(hide_code=True)
def _(df, ss):
    def solution_fit(ss, df):
        with pm.Model(coords=ss.coords) as model:
            pm.Data(
                "data_layoff",
                df[["layoff_active", "layoff_decay"]].to_numpy(),
                dims=["time", "state_layoff"],
            )

            pm.Normal(
                "initial_level",
                mu=[14.5, 0.0],
                sigma=[1.0, 0.02],
                dims=ss.param_dims["initial_level"],
            )
            pm.HalfNormal(
                "sigma_level",
                sigma=0.05,
                dims=ss.param_dims["sigma_level"],
            )

            pm.Normal(
                "beta_layoff",
                mu=0.0,
                sigma=0.3,
                dims=ss.param_dims["beta_layoff"],
            )
            pm.Normal(
                "params_seasonal",
                mu=0.0,
                sigma=0.05,
                dims=ss.param_dims["params_seasonal"],
            )
            pm.HalfNormal("sigma_noise", sigma=0.1)

            P0_diag = pm.Gamma(
                "P0_diag",
                alpha=2.0,
                beta=2.0,
                dims=ss.param_dims["P0"][:1],
            )
            pm.Deterministic("P0", pt.diag(P0_diag), dims=ss.param_dims["P0"])

            ss.build_statespace_graph(df["log_revenue"])
            trace = pm.sample(
                draws=250,
                tune=500,
            )
        return model, trace

    _, trace = solution_fit(ss, df)

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_fit)}\n```"),
                    mo.md(
                        "_The fitted result (`trace`) powers everything below, so the rest of the notebook works whether or not you complete the exercise._"
                    ),
                ]
            ),
        }
    )
    return (trace,)


@app.cell
def _(trace):
    az.plot_trace_dist(
        trace,
        var_names=[
            "beta_layoff",
            "params_seasonal",
            "sigma_level",
            "sigma_noise",
        ],
    )
    return


@app.cell
def _(trace):
    az.summary(
        trace,
        var_names=["beta_layoff", "params_seasonal", "sigma_level", "sigma_noise"],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Three flavors of "the latent state"

    The Kalman machinery gives us three different estimates of every
    hidden state at every time step:

    - **predicted**: $x_{t|t-1}$, the estimate using only past data.
      One-step-ahead forecasting done online.
    - **filtered**: $x_{t|t}$, the estimate using all data *through
      now*. The real-time best guess.
    - **smoothed**: $x_{t|T}$, the estimate using *all* data including
      future. The retrospective best guess.

    **The tunnel-and-GPS story.** Your car enters a tunnel:

    - **Filtered** view: at each moment inside the tunnel, you know
      where the car was when it entered, and the dynamics, but you
      don't know what happens next. Your uncertainty widens.
    - **Smoothed** view: the car came out the other side at a known
      time and place. Knowing that retrospectively tightens where it
      *must have been* during the tunnel: you can rule out
      trajectories that wouldn't match the exit point.
    - **Predicted** view: you only know up to where the car entered;
      every time step, you push the estimate forward one step.

    For historical decomposition (what we want next), use **smoothed**.
    For genuine forecasts, **predicted** is the honest comparison
    target: it's what the model would've said before seeing each next
    observation.

    `sample_conditional_posterior` produces all three;
    `extract_components_from_trace` then decomposes the smoothed
    posterior into per-component trajectories.
    """)
    return


@app.cell(hide_code=True)
def _():
    tunnel_flavor = mo.ui.radio(
        options=["predicted", "filtered", "smoothed"],
        value="filtered",
        label="tunnel example",
        inline=True,
    )
    tunnel_flavor
    return (tunnel_flavor,)


@app.cell(hide_code=True)
def _(tunnel_flavor):
    def _plot_tunnel_flavors(flavor):
        # Same DGP as §1, but here we compute predicted (forward, before
        # update), filtered (forward, after update), and smoothed
        # (backward RTS) so the radio above can flip between all three.
        rng_local = np.random.default_rng(7)
        n = 100
        tunnel = (40, 65)
        sigma_v, sigma_obs = 0.05, 0.8

        true_v = np.zeros(n)
        true_x = np.zeros(n)
        true_v[0] = 0.5
        for tt in range(1, n):
            true_v[tt] = true_v[tt - 1] + sigma_v * rng_local.standard_normal()
            true_x[tt] = true_x[tt - 1] + true_v[tt - 1]
        obs = true_x + sigma_obs * rng_local.standard_normal(n)
        observed = np.ones(n, dtype=bool)
        observed[tunnel[0] : tunnel[1]] = False

        T = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.array([[0.0, 0.0], [0.0, sigma_v**2]])
        Z = np.array([[1.0, 0.0]])
        H = np.array([[sigma_obs**2]])

        x_pred = np.zeros((n, 2))
        P_pred = np.zeros((n, 2, 2))
        x_filt = np.zeros((n, 2))
        P_filt = np.zeros((n, 2, 2))
        x_filt[0] = [0.0, 0.5]
        P_filt[0] = np.diag([1.0, 0.25])
        x_pred[0], P_pred[0] = x_filt[0], P_filt[0]
        for tt in range(1, n):
            x_pred[tt] = T @ x_filt[tt - 1]
            P_pred[tt] = T @ P_filt[tt - 1] @ T.T + Q
            if observed[tt]:
                innov = obs[tt] - (Z @ x_pred[tt])[0]
                S = (Z @ P_pred[tt] @ Z.T + H)[0, 0]
                K = (P_pred[tt] @ Z.T / S).flatten()
                x_filt[tt] = x_pred[tt] + K * innov
                P_filt[tt] = P_pred[tt] - np.outer(K, Z @ P_pred[tt])
            else:
                x_filt[tt], P_filt[tt] = x_pred[tt], P_pred[tt]

        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()
        for tt in range(n - 2, -1, -1):
            C = P_filt[tt] @ T.T @ np.linalg.inv(P_pred[tt + 1])
            x_smooth[tt] = x_filt[tt] + C @ (x_smooth[tt + 1] - x_pred[tt + 1])
            P_smooth[tt] = P_filt[tt] + C @ (P_smooth[tt + 1] - P_pred[tt + 1]) @ C.T

        flavors = {
            "predicted": (x_pred, P_pred, "C3"),
            "filtered": (x_filt, P_filt, "C0"),
            "smoothed": (x_smooth, P_smooth, "C2"),
        }
        x_arr, P_arr, color = flavors[flavor]
        std = np.sqrt(P_arr[:, 0, 0])
        ts = np.arange(n)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axvspan(
            tunnel[0], tunnel[1], color="gray", alpha=0.2, label="tunnel (no GPS)"
        )
        ax.plot(ts, true_x, color="black", lw=1, label="true position")
        ax.scatter(
            ts[observed], obs[observed], s=12, color="C3", alpha=0.6, label="noisy GPS"
        )
        ax.plot(ts, x_arr[:, 0], color=color, lw=1.2, label=f"{flavor} estimate")
        ax.fill_between(
            ts,
            x_arr[:, 0] - 2 * std,
            x_arr[:, 0] + 2 * std,
            alpha=0.25,
            color=color,
            label=f"{flavor} ±2σ",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("position")
        ax.set_title(f"Tunnel example — {flavor} estimate")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig

    _plot_tunnel_flavors(tunnel_flavor.value)
    return


@app.cell
def _(trace, ss):
    cond = ss.sample_conditional_posterior(trace, compile_kwargs={"mode": "NUMBA"})
    return (cond,)


@app.cell(hide_code=True)
def _():
    flavor_pick = mo.ui.radio(
        options=["predicted", "filtered", "smoothed"],
        value="smoothed",
        label="latent-state flavor",
        inline=True,
    )
    flavor_pick
    return (flavor_pick,)


@app.cell(hide_code=True)
def _(cond, df, flavor_pick, ss):
    def _plot_flavors(df, cond, ss, flavor):
        colors = {"predicted": "C3", "filtered": "C0", "smoothed": "C2"}
        obs_label = ss.observed_states[0]
        est = cond[f"{flavor}_posterior_observed"].sel(observed_state=obs_label)
        lo, hi = eti_band(est)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(
            df.index,
            df["log_revenue"],
            label="observed",
            color="black",
            lw=0.8,
            alpha=0.6,
        )
        ax.plot(
            df.index,
            est.mean(("chain", "draw")),
            label=f"{flavor} mean",
            color=colors[flavor],
            lw=1.2,
        )
        ax.fill_between(
            df.index, lo, hi, alpha=0.25, color=colors[flavor], label="89% ETI"
        )
        ax.set_ylabel("log(revenue)")
        ax.set_title(f"{flavor.capitalize()} latent state — mean and 89% ETI")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    _plot_flavors(df, cond, ss, flavor_pick.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Decomposition

    `extract_components_from_trace` splits the smoothed posterior into the
    per-component contributions (in observed-space for the `Regression`,
    latent-space for the `LevelTrend`). Summed back together they
    reconstruct the observed series; separately, they tell us where the
    signal lives.
    """)
    return


@app.cell
def _(cond, ss):
    comp = ss.extract_components_from_trace(cond)
    return (comp,)


@app.cell(hide_code=True)
def _(comp, df):
    def _plot_decomposition(df, comp):
        # smoothed_posterior holds the *latent* state for each component.
        # For LevelTrend and TimeSeasonality those are already in
        # observed units. For Regression it's the (constant, since
        # innovations=False) coefficient — multiply by the regressor
        # data to get the observed-space contribution.
        level = comp.smoothed_posterior.sel(state="level[level]")
        seasonal = comp.smoothed_posterior.sel(state="seasonal")
        beta_active = comp.smoothed_posterior.sel(state="layoff[active]")
        beta_decay = comp.smoothed_posterior.sel(state="layoff[decay]")
        active_contrib = beta_active * df["layoff_active"].to_numpy()
        decay_contrib = beta_decay * df["layoff_decay"].to_numpy()

        specs = [
            (level, "Latent level (underlying trend)", "C0"),
            (seasonal, "Annual seasonal pattern", "C2"),
            (active_contrib, "Layoff step contribution (β·active)", "C3"),
            (decay_contrib, "Layoff decay contribution (β·decay)", "C1"),
        ]
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 9))
        for ax, (series, title, color) in zip(fig.axes, specs):
            ax.plot(df.index, series.mean(("chain", "draw")), color=color, lw=1.2)
            ax.fill_between(df.index, *eti_band(series), alpha=0.25, color=color)
            ax.set_title(title)
        axes[-1].set_xlabel("month")
        fig.tight_layout()
        return fig

    _plot_decomposition(df, comp)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Forecast under a hypothetical second layoff

    **The setup.** Your CFO asks: *if another big client lays off in
    six months, what's the additional revenue exposure over the next
    two years?* The single-layoff model and its posterior
    (`ss`, `trace`) are what we have to answer with.

    **The framework's tool for this is `ss.forecast(...)`.** It
    extrapolates the latent state forward; for any model with
    exogenous regressors (like ours) you also pass a `scenario` dict
    that supplies the future values of those regressors. Different
    scenarios → different forecast paths from the same posterior.

    **Your task (30 min).** Compare two scenarios over a 24-month
    forecast horizon:

    - **Baseline**: no further layoffs. The original event's `decay`
      kernel keeps fading; `active` stays at 1.
    - **Second event in 6 months**: a new layoff hits at month 6 of
      the forecast window, adding a step to `active` and a fresh
      decay kernel on top of the existing one.

    The hard part (extending the regressor columns into the future) is provided as `make_scenario_regressors` below. You write the
    two `ss.forecast` calls: the pre-built plots below consume your
    forecasts.

    **What you'll produce.**

    1. A plot of historical `log_revenue` + both forecast scenarios
       with 89% ETI ribbons.
    2. A scenario-comparison plot: per-month and cumulative dollar
       difference between the two scenarios.

    **Tips.**

    - The forecast API: `ss.forecast(trace, start=..., periods=...,
      scenario={"data_layoff": ndarray}, random_seed=SEED)`. Use the
      *same* `random_seed` for both scenarios so the per-month
      difference is a clean draw-by-draw delta (not noise).
    - The object returned by `ss.forecast` is an InferenceData with
      its own groups: print it and dig around to find which one
      holds the predictive samples for the observed series.
    """)
    return


@app.cell(hide_code=True)
def _():
    def make_scenario_regressors(
        n_forecast,
        n_historical,
        event_month=17,
        second_event_at=None,
        tau=6,
    ):
        """Future (active, decay) regressors continuing from the historical data.

        Args:
            n_forecast: number of months to forecast.
            n_historical: length of the historical series (e.g. len(df)).
            event_month: month index of the original layoff event.
            second_event_at: months into the forecast horizon for a hypothetical
                new layoff (None → baseline scenario, no new event).
            tau: decay half-life in months (matches the historical kernel).

        Returns:
            ndarray of shape (n_forecast, 2): columns are [active, decay].
        """
        last_decay_t = n_historical - event_month
        months_since_first = np.arange(last_decay_t, last_decay_t + n_forecast)
        decay = np.exp(-months_since_first / tau)
        active = np.ones(n_forecast)
        if second_event_at is not None:
            post = np.arange(n_forecast) - second_event_at
            active = active + (post >= 0).astype(float)
            decay = decay + np.where(post >= 0, np.exp(-post / tau), 0.0)
        return np.column_stack([active, decay])

    return (make_scenario_regressors,)


@app.cell
def _(df, make_scenario_regressors):
    # Sanity-check the helper: baseline regressors over a 24-month horizon.
    baseline_regs = make_scenario_regressors(24, len(df))
    print(baseline_regs.shape)
    return (baseline_regs,)


@app.cell(hide_code=True)
def _(baseline_regs):
    _x = np.arange(19, 19 + len(baseline_regs))
    plt.plot(_x, baseline_regs[:, 1])
    plt.xlabel("months since the original layoff (event was at month 17)")
    plt.ylabel("decay regressor")
    plt.title("Baseline forecast — residual tail of the historical event")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The decay starts at $\exp(-19/6) \approx 0.04$ because the
    forecast picks up 19 months past the original event: most of the
    transient hit has already faded. The `active` column (not plotted)
    sits at a constant `1`: that's the *permanent* part of the
    layoff. In the second-event scenario, the decay curve gets a
    fresh `exp(-(t-6)/6)` bump added on top starting at forecast
    month 6: that bump is what drives the dollar gap between
    scenarios.
    """)
    return


@app.cell
def _(baseline_regs, df, trace, ss):
    def exercise_forecast():
        # YOUR CODE HERE — regressors for the second-layoff scenario:
        # make_scenario_regressors(24, len(df), second_event_at=6)
        event_regs = ...

        my_fc_baseline = ss.forecast(
            trace,
            start=df.index[-1],
            periods=24,
            scenario={"data_layoff": baseline_regs},
            random_seed=SEED,
        )
        # YOUR CODE HERE — the same forecast under event_regs
        my_fc_event = ...
        return my_fc_event

    return (exercise_forecast,)


@app.cell(hide_code=True)
def _():
    run_forecast = mo.ui.run_button(label="▶ Run exercise")
    run_forecast
    return (run_forecast,)


@app.cell(hide_code=True)
def _(exercise_forecast, run_forecast):
    mo.stop(
        not run_forecast.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_forecast()
    return


@app.cell(hide_code=True)
def _(baseline_regs, df, trace, make_scenario_regressors, ss):
    def solution_forecast():
        n_forecast = 24
        second_event_at = 6

        event_regs = make_scenario_regressors(
            n_forecast, len(df), second_event_at=second_event_at
        )

        fc_baseline = ss.forecast(
            trace,
            start=df.index[-1],
            periods=n_forecast,
            scenario={"data_layoff": baseline_regs},
            random_seed=SEED,
        )
        fc_event = ss.forecast(
            trace,
            start=df.index[-1],
            periods=n_forecast,
            scenario={"data_layoff": event_regs},
            random_seed=SEED,
        )
        return n_forecast, second_event_at, fc_baseline, fc_event

    n_forecast, second_event_at, fc_baseline, fc_event = solution_forecast()

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_forecast)}\n```"),
                    mo.md(
                        "_The forecasts (`fc_baseline`, `fc_event`) power the plots below, so they work whether or not you complete the exercise._"
                    ),
                ]
            ),
        }
    )
    return fc_baseline, fc_event, n_forecast, second_event_at


@app.cell(hide_code=True)
def _(df, fc_baseline, fc_event, n_forecast, second_event_at, ss):
    def _plot_forecast_scenarios():
        obs_label = ss.observed_states[0]
        future_dates = pd.date_range(
            df.index[-1] + pd.DateOffset(months=1),
            periods=n_forecast,
            freq="MS",
        )

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(
            df.index,
            df["log_revenue"],
            color="black",
            lw=1,
            label="historical",
        )
        for fc, label, color in [
            (fc_baseline, "no further layoffs", "C0"),
            (fc_event, f"new event in {second_event_at} months", "C3"),
        ]:
            y = fc.forecast_observed.sel(observed_state=obs_label)
            ax.plot(
                future_dates,
                y.mean(("chain", "draw")),
                color=color,
                lw=1.2,
                label=label,
            )
            ax.fill_between(future_dates, *eti_band(y), alpha=0.2, color=color)
        ax.axvline(df.index[-1], color="gray", ls="--", lw=0.7, label="forecast start")
        ax.set_ylabel("log(revenue)")
        ax.set_title("Forecast under two scenarios — same posterior, different futures")
        ax.legend(loc="lower left")
        fig.tight_layout()
        return fig

    _plot_forecast_scenarios()
    return


@app.cell(hide_code=True)
def _(df, fc_baseline, fc_event, n_forecast, ss):
    def _plot_scenario_difference():
        obs_label = ss.observed_states[0]
        # Same random_seed across the two forecasts → draws line up,
        # so we can take a per-draw difference for a clean per-month
        # causal effect.
        y_baseline = np.exp(fc_baseline.forecast_observed.sel(observed_state=obs_label))
        y_event = np.exp(fc_event.forecast_observed.sel(observed_state=obs_label))
        monthly_loss = y_baseline - y_event
        cumulative_loss = monthly_loss.cumsum("time")

        future_dates = pd.date_range(
            df.index[-1] + pd.DateOffset(months=1),
            periods=n_forecast,
            freq="MS",
        )

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
        for ax, series, title in zip(
            fig.axes,
            [monthly_loss, cumulative_loss],
            [
                "Per-month difference vs. baseline ($)",
                "Cumulative difference ($)",
            ],
        ):
            ax.plot(future_dates, series.mean(("chain", "draw")), color="C3")
            ax.fill_between(future_dates, *eti_band(series), alpha=0.25, color="C3")
            ax.axhline(0, color="black", lw=0.6, linestyle="--")
            ax.set_title(title)
            ax.set_ylabel("$")
        axes[-1].set_xlabel("month")
        fig.tight_layout()
        return fig

    _plot_scenario_difference()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The cumulative-loss panel is the headline number: the posterior
    over the dollar exposure of *one additional layoff event in six
    months*, conditional on the model. Everything in this plot is
    a model-based scenario comparison: same posterior parameters, two
    different exogenous-regressor futures, draw-by-draw differenced.
    The 89% ETI gives the CFO a calibrated range, not just a point
    estimate.

    **Caveat.** The hypothetical event uses the same decay half-life
    and the same `beta_layoff` posterior as the historical one, i.e.,
    we're assuming a "similar kind of layoff." Different-magnitude
    events would need a per-event `beta` (see "Where to next" below).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extension — modeling headcount jointly (restricted BVAR)

    So far the layoff has been an exogenous regressor: we
    hand-engineered `layoff_active` and `layoff_decay` as the *shape*
    of its impact on revenue, then asked the model for the
    coefficients. That works, but it's a thin causal story: we're
    asserting "revenue dropped at this time and recovered like
    *this*", not "revenue dropped *because headcount dropped*".

    The dataset includes a parallel `log_headcount` series. Promoting
    it from a hand-coded basis function to a *modeled* series turns
    the description into a mechanism: layoffs cut headcount, headcount
    drives revenue. The natural joint structure is a **restricted
    VAR(1)**:

    $$
    \begin{aligned}
    h_t &= \phi_h\, h_{t-1} + \varepsilon_t^h \\
    r_t &= \phi_r\, r_{t-1} + \gamma\, h_{t-1} + \varepsilon_t^r
    \end{aligned}
    $$

    Headcount has its own AR dynamics. Revenue depends on its own lag
    *and* on lagged headcount. The restriction is that revenue does
    *not* feed back into headcount: the layoff calendar is set by
    the client's executives, not by what last quarter's billing did.
    In matrix form:

    $$
    T = \begin{bmatrix} \phi_h & 0 \\ \gamma & \phi_r \end{bmatrix}
    $$

    The zero in the upper-right is the restriction. We use the
    off-the-shelf `pmss.BayesianVARMAX` and impose the restriction by
    constructing the AR-coefficient tensor as a `pm.Deterministic`:
    the zero entry stays a literal zero, the others get scalar priors.

    **What this buys.**

    - **A structural counterfactual.** Shock headcount by 0.3 log
      points (~26%) at $t=0$ and trace what revenue does over the
      next 24 months. That's an impulse-response, with a posterior, and unlike the earlier univariate counterfactual, the recovery shape is *learned*
      from the data, not stipulated by the analyst.
    - **The layoff doesn't appear in the model.** No `layoff_active`,
      no `layoff_decay`. The layoff is a fact about the data (a big
      innovation to $h_t$ at one specific month), not a feature we
      engineered. The mechanism (not the calendar) is what gets
      modeled.
    - **Two side benefits worth knowing about** (we'll see both
      below): joint forecasting becomes one call instead of needing
      `make_scenario_regressors` for the predictor (the forecasting
      exercise's pain point), and headcount's NaN gaps now get the same free
      missing-data treatment as revenue's, because headcount is now
      an observation rather than a `pm.Data` input.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Detrend each series

    The VAR(1) above is stationary around zero: no intercept, no
    trend. The data have visible drift (headcount slowly grows
    pre-layoff; revenue drifts up over the whole window), so we
    pre-detrend with simple OLS lines and fit on the residuals. The
    detrend ignores NaN entries when fitting the line and propagates
    NaN through to the residuals so the filter still sees the gaps.

    This is a shortcut. The clean version adds a deterministic trend
    to the state, doubling the state dimension. The OLS line is
    biased a bit by the layoff drop itself; for short series the bias
    is tolerable.
    """)
    return


@app.cell(hide_code=True)
def _(df):
    def _detrend(s):
        x = np.arange(len(s))
        vals = s.to_numpy()
        mask = ~np.isnan(vals)
        a, b = np.polyfit(x[mask], vals[mask], 1)
        trend = a * x + b
        dev = pd.Series(vals - trend, index=s.index, name=s.name)
        return dev, pd.Series(trend, index=s.index, name=s.name)

    h_dev, h_trend = _detrend(df["log_headcount"])
    r_dev, r_trend = _detrend(df["log_revenue"])
    single_dev = pd.concat([h_dev, r_dev], axis=1)
    single_trend = pd.concat([h_trend, r_trend], axis=1)
    return single_dev, single_trend


@app.cell(hide_code=True)
def _(single_dev):
    def _plot_dev(df):
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 4.5))
        axes[0].plot(df.index, df["log_headcount"], color="C2", marker="o", lw=1)
        axes[0].axhline(0, color="black", lw=0.5)
        axes[0].set_ylabel("log_headcount (detrended)")
        axes[1].plot(df.index, df["log_revenue"], color="C0", marker="o", lw=1)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_ylabel("log_revenue (detrended)")
        axes[-1].set_xlabel("month")
        fig.suptitle("Both series, detrended — input to the VAR (gaps preserved)")
        fig.tight_layout()
        return fig

    _plot_dev(single_dev)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Set up the BVAR via `BayesianVARMAX`

    `pmss.BayesianVARMAX` is the off-the-shelf vector ARMA component.
    For VAR(1) we set `order=(1, 0)`. With two endogenous series, the
    AR-coefficient tensor `ar_params` has shape `(2, 1, 2)`: `[i, 0, j]`
    is the coefficient on $x_{j, t-1}$ in the equation for $x_{i, t}$.

    For the restricted VAR we never declare `ar_params` as a free
    random variable. Instead we build it from scalar priors and a
    literal zero, and register the result as a `pm.Deterministic`:
    the framework picks it up by name. (The class docstring says it
    explicitly: *"For restricted models, set zeros directly on the
    priors."*)
    """)
    return


@app.cell
def _():
    bvar = pmss.BayesianVARMAX(
        order=(1, 0),
        endog_names=["log_headcount", "log_revenue"],
        stationary_initialization=False,
        measurement_error=False,
    )
    return (bvar,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Wire up priors and sample

    Five scalar dynamics parameters, one zero, plus initial-state
    mean/covariance. Priors:

    - `phi_h ~ TruncatedNormal(0.7, 0.2)`: moderate AR persistence
      for headcount, kept inside the unit circle.
    - `phi_r ~ TruncatedNormal(0.3, 0.3)`: weaker prior on revenue's
      own persistence; the data should drive it.
    - `gamma ~ Normal(0, 0.5)`: agnostic about the cross-coupling
      sign and magnitude. The mechanism (more headcount → more
      revenue) implies positive, but we let the data decide.
    - `sigma_h, sigma_r ~ HalfNormal`: innovation scales.

    `ar_params[0, 0, 1]` is hard-coded to zero. That's the restriction.
    """)
    return


@app.cell
def _(bvar, single_dev):
    with pm.Model(coords=bvar.coords) as bvar_model:
        phi_h = pm.TruncatedNormal("phi_h", mu=0.7, sigma=0.2, lower=-0.99, upper=0.99)
        phi_r = pm.TruncatedNormal("phi_r", mu=0.3, sigma=0.3, lower=-0.99, upper=0.99)
        gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)
        sigma_h = pm.HalfNormal("sigma_h", sigma=0.1)
        sigma_r = pm.HalfNormal("sigma_r", sigma=0.05)

        # Restricted AR(1) coefficient tensor. Index [i, 0, j] = coef
        # on variable j's lag in equation i. Endog order is
        # (log_headcount, log_revenue), so:
        #   [0, 0, 0] = phi_h     headcount autoregression
        #   [0, 0, 1] = 0         ← THE RESTRICTION (no r → h feedback)
        #   [1, 0, 0] = gamma     revenue depends on lagged headcount
        #   [1, 0, 1] = phi_r     revenue autoregression
        ar_tensor = pt.zeros((2, 1, 2))
        ar_tensor = ar_tensor[0, 0, 0].set(phi_h)
        ar_tensor = ar_tensor[1, 0, 0].set(gamma)
        ar_tensor = ar_tensor[1, 0, 1].set(phi_r)
        pm.Deterministic("ar_params", ar_tensor, dims=bvar.param_dims["ar_params"])

        # Diagonal innovation covariance — independent shocks on h and r.
        pm.Deterministic(
            "state_cov",
            pt.diag(pt.stack([sigma_h**2, sigma_r**2])),
            dims=bvar.param_dims["state_cov"],
        )

        pm.Normal("x0", mu=0.0, sigma=0.5, dims=bvar.param_dims["x0"])
        P0_diag = pm.Gamma("P0_diag", alpha=2.0, beta=10.0, dims=bvar.param_dims["x0"])
        pm.Deterministic("P0", pt.diag(P0_diag), dims=bvar.param_dims["P0"])

        bvar.build_statespace_graph(single_dev)
        bvar_trace = pm.sample(tune=500, draws=250)
    return (bvar_trace,)


@app.cell
def _(bvar_trace):
    az.plot_trace_dist(
        bvar_trace,
        var_names=["phi_h", "phi_r", "gamma", "sigma_h", "sigma_r"],
        backend="matplotlib",
    )
    return


@app.cell
def _(bvar_trace):
    az.summary(
        bvar_trace,
        var_names=["phi_h", "phi_r", "gamma", "sigma_h", "sigma_r"],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Smoothed posterior — and the headcount gap fills itself

    The data has a 3-month gap in `log_headcount` at months 21–23.
    Because headcount is now an *observation* in the joint model, the
    Kalman smoother handles those NaNs the same way it handles missing
    GPS in the earlier tunnel example: skip the measurement update,
    propagate state via the dynamics, narrow back up when the next
    observation arrives. The widening of the ETI ribbon during the
    gap is the posterior telling you exactly how much it lost from
    those missing observations.

    This is the payoff of promoting a covariate to a state. In the
    univariate case study, a NaN in `log_headcount` (had we been using it
    as a regressor) would have crashed the model. Here it just costs us
    some uncertainty during the gap.
    """)
    return


@app.cell
def _(bvar, bvar_trace):
    cond_bvar = bvar.sample_conditional_posterior(
        bvar_trace,
        compile_kwargs={"mode": "NUMBA"},
    )
    return (cond_bvar,)


@app.cell(hide_code=True)
def _(cond_bvar, df, single_trend):
    def _plot_smoothed_with_gap():
        # Smoothed posterior is in detrended deviation-space; add the
        # OLS trend back to plot in original log-headcount units.
        smoothed_dev = cond_bvar.smoothed_posterior_observed.sel(
            observed_state="log_headcount"
        )
        trend_h = single_trend["log_headcount"].to_numpy()
        smoothed = smoothed_dev + trend_h

        observed = df["log_headcount"]
        gap_mask = observed.isna()
        gap_dates = observed.index[gap_mask]

        fig, ax = plt.subplots(figsize=(11, 4))
        if len(gap_dates):
            ax.axvspan(
                gap_dates[0],
                gap_dates[-1] + pd.DateOffset(months=1),
                color="gray",
                alpha=0.15,
                label="missing observations",
            )
        ax.plot(
            observed.index,
            observed,
            color="black",
            marker="o",
            lw=1,
            label="observed",
        )
        ax.plot(
            df.index,
            smoothed.mean(("chain", "draw")),
            color="C2",
            label="smoothed posterior mean",
        )
        ax.fill_between(
            df.index,
            *eti_band(smoothed),
            alpha=0.25,
            color="C2",
            label="89% ETI",
        )
        ax.set_ylabel("log(headcount)")
        ax.set_title(
            "Headcount: observations + smoothed posterior — "
            "the joint filter fills the gap"
        )
        ax.legend(loc="lower left")
        fig.tight_layout()
        return fig

    _plot_smoothed_with_gap()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Impulse response — a representative layoff shock

    The IRF asks the counterfactual: *if headcount drops by 0.3 log
    points (~26%, the size of one layoff in this dataset) at $t=0$
    and nothing else, how does revenue respond over the next 24
    months?* The dynamics are linear, so the answer scales: a
    half-size layoff gives half this response.
    """)
    return


@app.cell
def _(bvar, bvar_trace):
    n_irf = 24
    shock_size = -0.30
    shock_trajectory = np.zeros((n_irf, 2))
    shock_trajectory[0, 0] = shock_size

    irf_bvar = bvar.impulse_response_function(
        bvar_trace,
        shock_trajectory=shock_trajectory,
    )
    return irf_bvar, n_irf


@app.cell(hide_code=True)
def _(irf_bvar, n_irf):
    def _plot_irf(irf):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ts = np.arange(n_irf)
        specs = [
            ("log_headcount", "Headcount response", "C2"),
            ("log_revenue", "Revenue response", "C0"),
        ]
        for ax, (state_name, title, color) in zip(fig.axes, specs):
            arr = irf.irf.sel(state=state_name)
            ax.plot(ts, arr.mean(("chain", "draw")), color=color)
            ax.fill_between(ts, *eti_band(arr), alpha=0.25, color=color)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel("months since shock")
            ax.set_ylabel(r"$\Delta$ log")
            ax.set_title(title)
        fig.suptitle("IRF: a 0.3 log-point negative shock to headcount at t=0")
        fig.tight_layout()
        return fig

    _plot_irf(irf_bvar)
    return


@app.cell(hide_code=True)
def _(df, irf_bvar, n_irf):
    def _plot_cumulative_dollars():
        # Approximate dollar loss at month k:
        #   baseline_revenue_k * (1 - exp(Δ log_revenue_k))
        # ≈ -baseline_revenue_k * Δ log_revenue_k for small Δ.
        # Use the latest historical revenue as the baseline scale.
        baseline_dollar = float(np.exp(df["log_revenue"].iloc[-1]))
        delta_log_r = irf_bvar.irf.sel(state="log_revenue")
        monthly_loss = -baseline_dollar * delta_log_r
        cumulative_loss = monthly_loss.cumsum("time")

        ts = np.arange(n_irf)
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
        for ax, series, title in zip(
            fig.axes,
            [monthly_loss, cumulative_loss],
            ["Monthly $ revenue lost", "Cumulative $ revenue lost"],
        ):
            ax.plot(ts, series.mean(("chain", "draw")), color="C3")
            ax.fill_between(ts, *eti_band(series), alpha=0.25, color="C3")
            ax.axhline(0, color="black", lw=0.6, linestyle="--")
            ax.set_ylabel("$")
            ax.set_title(title)
        axes[-1].set_xlabel("months since shock")
        fig.suptitle(
            "Revenue impulse response to a representative layoff shock, in dollars (BVAR)"
        )
        fig.tight_layout()
        return fig

    _plot_cumulative_dollars()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### BVAR vs Regression — what changed

    The univariate case study and this BVAR answer the same business question
    (*what does the layoff cost?*) two different ways:

    | | Univariate Regression | BVAR |
    |---|---|---|
    | What's modeled | revenue alone | headcount + revenue |
    | Layoff appears as | hand-coded `(active, decay)` regressors | nothing: it's a big innovation in the data |
    | Recovery shape | hard-coded $\exp(-(t-t_e)/\tau)$ | learned via $\phi_h$ |
    | Counterfactual | subtract regression contribution | impulse-response |
    | Forecasting | needs `make_scenario_regressors` for *every* covariate | one `forecast` call extrapolates both series |
    | Missing data in covariate | crashes (`pm.Data` can't have NaN) | filter handles transparently: see the smoothed-posterior section |
    | Identification cost | low (3 + extras parameters) | higher (5 + extras), and the restriction |
    | Causal story | "revenue dropped because event happened" | "revenue dropped because headcount dropped" |

    With short data (36 monthly observations, one event), the
    Regression model is statistically more efficient: fewer
    parameters, sharper posteriors. The BVAR is more honest about
    the mechanism, doesn't require a fixed decay shape, gives an IRF
    for free, *and* gracefully degrades when the predictor itself has
    gaps. The restriction (`T[0, 1] = 0`) is a strong assumption: if
    revenue *does* feed back into headcount decisions on long horizons (shrinking accounts triggering further cuts), the IRF
    underestimates the total dynamic effect.

    A hierarchical extension lifts both: pool
    $\phi_h, \gamma$ across many client accounts and per-account
    identification stops being the bottleneck.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where to next
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Statespace modeling is an extremely rich area with tons of resources and active research. If you're interested in learning more, here are some resources:

    - [Forecasting: Principals and Practice (3rd Edition)](https://otexts.com/fpp3/) is the go-to practitioner's reference text for time-series modeling. It's not specifically statespace, but basically everything here has an analogue in there, and the theory is the same
    - [Time Series Analysis by Statespace Methods](https://academic.oup.com/book/16563?login=false) is the go-to academic reference for the specific class of models we saw in this workshop. This is a theory-first book, so expect a lot of math. Useful if you want to deeply understand what is going on under the hood, but less useful for a practitioner.
    - [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf) is also a reference that gets recommended a lot, but I haven't read it
    - [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) is, for my money, the best reference out there for Kalman filtering. It's extremely approachable and the code is written in a hackable way to help build intuition about how these methods work. There are nice sections at the end on advanced topics related to non-linear and non-Guassian systems as well.

    For more PyMC specific resources:
    - The [pymc-extras repo has several example notebooks](https://github.com/pymc-devs/pymc-extras/tree/main/notebooks) about statespace models that build on the content we saw here
    - This [presentation](https://www.youtube.com/watch?v=G9VWXZdbtKQ) of the PyMC Statspace functionality, and the associated [github resources ](https://github.com/jessegrabowski/statespace-presentation)
    - [This talk at PyData Berlin 2025 ](https://www.youtube.com/watch?v=lXU5dr6Lmgo), and the [associated notebooks](https://github.com/AlexAndorra/pydata-berlin-statespace-models)
    """)
    return


if __name__ == "__main__":
    app.run()
