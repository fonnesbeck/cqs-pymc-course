import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import inspect
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import polars as pl
    import pymc as pm
    import arviz_plots as azp

    RANDOM_SEED = 42
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


@app.cell(hide_code=True)
def header():
    mo.md(r"""
    # Session 2.2: Building Models with PyMC

    In this session, we'll continue exploring PyMC by building probabilistic models, specifying priors, and working with deterministic and observed variables.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Building Models in PyMC

    Now that we understand the basic building blocks of PyMC models, let's see how to combine them to build a complete model. Let's see if we can implement the rat toxicity model from the previous section!

    Recall our dataset:
    """)
    return


@app.cell
def _():
    dose_1 = np.array([-0.86, -0.3, -0.05, 0.73])
    n_1 = 5
    deaths_1 = np.array([0, 1, 3, 5])
    return deaths_1, dose_1, n_1


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(r"""
    In this dataset `log_dose` includes 4 levels of dosage, on the log scale, each administered to `n=5` rats during the experiment. The response variable is `deaths`, the number of positive responses to the dosage.

    The number of deaths can be modeled as a binomial response, with the probability of death being a linear function of dose:

    $$
    \begin{aligned}
    \text{logit}(p_i) &= \beta_0 + \beta_1 x_i \\
    y_i &\sim \text{Bin}(n_i, p_i) \\
    \end{aligned}
    $$

    Our research interest is in estimating the **LD50**, the dose at which 50% of the rats die.
    """),
            mo.callout(
                mo.md(r"""
    **Exercise:** How do we calculate the LD50 from the model?
    """),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prior Specification

    The first step in specifying a PyMC model is defining the prior distributions for each unknown parameter.

    - what are the unknown parameters?
    - which distributions should we use to characterize our beliefs about their values?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We need to create priors for the regression coefficients. One approach is to specify scalar priors for each coefficient:
    """)
    return


@app.cell
def _():
    def build_dose_scalar_model():
        with pm.Model() as model:
            pm.Normal("beta0", mu=0, sigma=10)
            pm.Normal("beta1", mu=0, sigma=10)
        return model

    dose_model = build_dose_scalar_model()
    dose_model
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Alternately, we can create a single, vector-valued parameter that contains both the intercept and slope parameters:
    """)
    return


@app.cell
def _():
    with pm.Model() as dose_model_1:
        beta_1 = pm.Normal("beta", mu=0, sigma=10, shape=2)
    dose_model_1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Better yet, we can used **named dimensions** to specify the shape of the random variable.
    """)
    return


@app.cell
def _():
    with pm.Model(coords=dict(coeffs=["intercept", "slope"])) as dose_model_2:
        beta_2 = pm.Normal("beta", mu=0, sigma=10, dims="coeffs")
    dose_model_2
    return beta_2, dose_model_2


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Deterministic Variables

    A deterministic variable is one whose values are **completely determined** by the values of their parents.

    In our model, the probability of death is a deterministic function of the dose and the parameters of the model.
    """)
    return


@app.cell
def _(beta_2, dose_1, dose_model_2):
    with dose_model_2:
        p = pm.math.invlogit(beta_2[0] + beta_2[1] * dose_1)
    return (p,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    There are two types of deterministic variables in PyMC:

    #### Anonymous deterministic variables

    The easiest way to create a deterministic variable is to operate on or transform one or more variables in a model directly, as we have done above for `p`.

    These are called *anonymous* variables because they are not named variables, as we did for `beta` above. We simply specified the variable as a Python (or, Pytensor) expression. This is therefore the simplest way to construct a determinstic variable. The only caveat is that the values generated by anonymous determinstics are not recorded to the model output during model fitting. So, this approach is only appropriate for intermediate values in your model that you do not wish to obtain posterior estimates for, alongside the other variables in the model.

    #### Named deterministic variables

    To ensure that deterministic variables' values are accumulated during sampling, they should be instantiated using the **named deterministic** interface; this uses the `Deterministic` function to create the variable. Two things happen when a variable is created this way:

    1. The variable is given a name (passed as the first argument)
    2. The variable is appended to the model's list of random variables, which ensures that its values are tallied.

    Where does that expression come from? At the LD50, half the subjects die:
    $p = 0.5$, so $\text{logit}(p) = \log(0.5/0.5) = 0$. Setting the linear
    predictor to zero, $\beta_0 + \beta_1 x = 0$, and solving gives
    $x = -\beta_0/\beta_1$. In the model below the two coefficients live in one vector, so this becomes `-beta_2[0] / beta_2[1]`.

    Since we are interested in estimating LD50, let's create a named deterministic variable for it:
    """)
    return


@app.cell
def _(beta_2, dose_model_2):
    with dose_model_2:
        pm.Deterministic("ld50", -beta_2[0] / beta_2[1])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Observed Random Variables

    Stochastic random variables whose values are observed are represented by a different class than unobserved random variables. An `ObservedRV` object is instantiated any time a stochastic variable is specified with data passed as the `observed` argument.

    In our model, the observed rat deaths are represented by an `ObservedRV` using a binomial random variable.
    """)
    return


@app.cell
def _(deaths_1, dose_model_2, n_1, p):
    with dose_model_2:
        y_2 = pm.Binomial("y", n=n_1, p=p, observed=deaths_1)
    dose_model_2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Factor Potentials

    For some applications, we want to be able to modify the joint density by incorporating terms that don't correspond to probabilities of variables conditional on parents. For example, suppose in our model we want to constrain the slope to be positive, so that the joint density becomes:

    $$p(y,\beta) \propto p(y|\beta) p(\beta) I(\beta \gt 0)$$

    We call such log-probability terms **factor potentials** (Jordan 2004). Bayesian
    hierarchical notation doesn't accomodate these potentials.

    ### Creation of Potentials

    A potential can be created via the `Potential` function, in a way very similar to `Deterministic`'s named interface:
    """)
    return


@app.cell
def _(beta_2, dose_model_2):
    with dose_model_2:
        pm.Potential("slope_constraint", pm.math.switch(beta_2[1] < 0, -np.inf, 0))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The function takes just a `name` as its first argument and an expression returning the appropriate log-probability as the second argument.

    A common use of a factor potential is to represent an observed likelihood, where the **observations are partly a function of model variables**. In the example below, we are representing the error in a linear regression model as a zero-mean normal random variable. Thus, the "data" in this scenario is the residual, which is a function both of the data and the regression parameters.

    $$\begin{align*}
    \epsilon_i &= y_i - \hat{y}_i \\
    &= y_i - (\mu + \beta x_i)
    \end{align*}$$

    $$\epsilon_i \sim \text{Normal}(0, \sigma)$$


    If we represent this as a standard likelihood function (a `Distribution` with an `observed` keyword argument), we run into problems. This parameterization would not be compatible with an observed stochastic, because the `err` term would become fixed in the likelihood and not be allowed to change during sampling.
    """)
    return


@app.cell
def _():
    y_vals = np.array([15, 10, 16, 11, 9, 11, 10, 18, 11])
    x_vals = np.array([1, 2, 4, 5, 6, 8, 19, 18, 12])
    return x_vals, y_vals


@app.cell
def _(x_vals, y_vals):
    def demo_observed_error():
        err_msg = None
        try:
            with pm.Model():
                sigma = pm.HalfCauchy("sigma", 5)
                beta = pm.Normal("beta", 0, sigma=2)
                mu = pm.Normal("mu", 0, sigma=10)
                err = y_vals - (mu + beta * x_vals)
                pm.Normal("like", mu=0, sigma=sigma, observed=err)
        except TypeError as e:
            err_msg = str(e)
        return err_msg

    err_msg = demo_observed_error()
    _output = (
        mo.callout(mo.md(f"**Expected error:** `{err_msg}`"), kind="warn")
        if err_msg
        else None
    )
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Instead, we can re-express the likelihood as a factor potential, which is a function of the data and the model parameters.
    """)
    return


@app.cell
def _(x_vals, y_vals):
    def build_potential_model():
        with pm.Model() as model:
            sigma = pm.HalfCauchy("sigma", 5)
            beta = pm.Normal("beta", 0, sigma=2)
            mu = pm.Normal("mu", 0, sigma=10)
            err = y_vals - (mu + beta * x_vals)
            pm.Potential("like", pm.logp(pm.Normal.dist(0, sigma=sigma), err))
        return model

    _potential_model = build_potential_model()
    _potential_model
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Our model specification is complete! Let's take a look at the model structure:
    """)
    return


@app.cell
def _(dose_model_2):
    dose_model_2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Alternative Model Specification

    The use of a `Potential` to constrain the slope to be positive is not a great solution. A better approach is to use a prior distribution that is constrained to be positive.

    Try re-specifying the model using this strategy.
    """)
    return


@app.cell
def _(deaths_1, dose_1, n_1):
    def exercise_positive_slope_model():
        with pm.Model() as model:
            # YOUR CODE HERE — Normal prior for the intercept, and a
            # positive-constrained prior for the slope (no Potential needed)
            beta0 = ...
            beta1 = ...
            p = pm.math.invlogit(beta0 + beta1 * dose_1)
            pm.Deterministic("ld50", -beta0 / beta1)
            pm.Binomial("y", n=n_1, p=p, observed=deaths_1)
        return model

    return (exercise_positive_slope_model,)


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Hint": mo.md(
                "Positive-valued distributions include `HalfNormal`, `LogNormal`, and `Gamma`."
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    run_positive_slope = mo.ui.run_button(label="▶ Run exercise")
    run_positive_slope
    return (run_positive_slope,)


@app.cell(hide_code=True)
def _(exercise_positive_slope_model, run_positive_slope):
    mo.stop(
        not run_positive_slope.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_positive_slope_model()
    return


@app.cell(hide_code=True)
def _(deaths_1, dose_1, n_1):
    def solution_positive_slope_model():
        with pm.Model() as model:
            beta0 = pm.Normal("beta0", 0, sigma=1)
            beta1 = pm.LogNormal("beta1", 0, sigma=1)
            p = pm.math.invlogit(beta0 + beta1 * dose_1)
            pm.Deterministic("ld50", -beta0 / beta1)
            pm.Binomial("y", n=n_1, p=p, observed=deaths_1)
        return model

    dose_model_4 = solution_positive_slope_model()

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_positive_slope_model)}\n```"
                    ),
                    dose_model_4,
                    mo.md(
                        "_This model (`dose_model_4`) is used by the cells that follow, so the rest of the notebook works whether or not you complete the exercise._"
                    ),
                ]
            ),
        }
    )
    return (dose_model_4,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Parameter Transformation

    To support efficient sampling by PyMC's MCMC algorithms, any continuous variables that are **constrained** to a sub-interval of the real line are **automatically transformed** so that their support is unconstrained. This frees sampling algorithms from having to deal with boundary constraints.

    For example, if we look at the variables we have created in the model so far:
    """)
    return


@app.cell
def _(dose_model_4):
    _output = mo.md(f"""
    `{dose_model_4.value_vars}`
    """)
    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The model's `value_vars` attribute stores the values of each random variable actually used by the model's log-likelihood.

    As the name suggests, the variable `beta1` has been log-transformed, and this is the space over which posterior sampling takes place. When a sample is drawn, the value of the transformed variable is simply back-transformed to recover the original variable.

    By default, auto-transformed variables are ignored when summarizing and plotting model output, since they are not generally of interest to the user.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prior Predictive Checks

    Before we go ahead and estimate the model paramters from the data, it's a good idea to perform a prior predictive check. This involves sampling from the model before data are incorporated, and gives you an idea of the range of observations that would be considered reasonable within the scope of the modeling assumptions (including choice of priors). If the simnulations generate too many extreme observations relative to our expectations based on domain knowledge, then it can be an indication of problems with model formulation.
    """)
    return


@app.cell
def _(dose_model_4):
    with dose_model_4:
        prior_sample = pm.sample_prior_predictive(1000)
    prior_sample
    return (prior_sample,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's see what our simulated datasets look like:
    """)
    return


@app.cell
def _(prior_sample):
    azp.plot_dist(prior_sample, group="prior_predictive", var_names=["y"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    How do we interpret this?

    The prior predictive shows deaths relatively uniformly distributed from 0 to 5. This is reasonable since we haven't yet incorporated information about the dose-response relationship. The lack of extreme concentration at any particular value suggests our priors aren't overly informative.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If we are happy with our model specification, we can go ahead and estimate the model.
    """)
    return


@app.cell
def _(dose_model_4):
    with dose_model_4:
        trace = pm.sample(nuts_sampler="nutpie", random_seed=RANDOM_SEED)
    trace
    return (trace,)


@app.cell
def _(trace):
    azp.plot_dist(trace, var_names=["beta0", "beta1", "ld50"])
    return


@app.cell
def _(deaths_1, dose_1, n_1, trace):
    def plot_dose_response_posterior():
        empirical_p = deaths_1 / n_1
        dose_range = np.linspace(dose_1.min(), dose_1.max(), 100)
        beta0_samples = trace.posterior["beta0"].values.flatten()
        beta1_samples = trace.posterior["beta1"].values.flatten()

        def logistic(x):
            return 1 / (1 + np.exp(-x))

        fig = go.Figure()
        for i in np.arange(-250, 0):
            prob = logistic(beta0_samples[i] + beta1_samples[i] * dose_range)
            fig.add_trace(
                go.Scatter(
                    x=dose_range,
                    y=prob,
                    mode="lines",
                    line=dict(color="blue", width=1),
                    opacity=0.1,
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=dose_1,
                y=empirical_p,
                mode="markers",
                marker=dict(color="black", size=8),
                showlegend=False,
            )
        ).update_layout(
            xaxis_title="log(dose)",
            yaxis_title="Probability of death",
            template="simple_white",
            showlegend=False,
        )
        return fig

    plot_dose_response_posterior()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prediction and forecasting

    We might also be interested in predicting on unseen or data. For example, how many deaths would we expect if we administered the LD50 dose?

    In PyMC, we can use `pm.Data` objects for our data. It allows you to define data as a symbolic node in the model that you can later switch out for other data.
    """)
    return


@app.cell
def _(deaths_1, dose_1):
    with pm.Model(
        coords=dict(
            coeffs=["intercept", "slope"], dosis=["first", "second", "third", "fourth"]
        )
    ) as dose_model_5:
        dose_data = pm.Data("dose_data", dose_1, dims="dosis")
        deaths_data = pm.Data("deaths_data", deaths_1, dims="dosis")
    return deaths_data, dose_data, dose_model_5


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice the `coords` argument in the `pm.Data` constructor. This allows us to specify named dimensions for the data variable, which is useful when working with multi-dimensional data.

    In the previous model, we specified separate variables for the slope and intercept. We can instead use a single vector-valued variable with named dimensions to represent both coefficients.
    """)
    return


@app.cell
def _(dose_model_5):
    with dose_model_5:
        beta_6 = pm.Normal("beta", 0, sigma=2, dims="coeffs")
    return (beta_6,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The rest of the model remains the same:
    """)
    return


@app.cell
def _(beta_6, deaths_data, dose_data, dose_model_5, n_1):
    with dose_model_5:
        p_4 = pm.Deterministic(
            "p", pm.math.invlogit(beta_6[0] + beta_6[1] * dose_data), dims="dosis"
        )
        pm.Deterministic("ld50", -beta_6[0] / beta_6[1])
        pm.Binomial("y", n=n_1, p=p_4, observed=deaths_data, dims="dosis")
    return


@app.cell
def _(dose_model_5):
    dose_model_5
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now, we estimate the model:
    """)
    return


@app.cell
def _(dose_model_5):
    with dose_model_5:
        trace_1 = pm.sample(nuts_sampler="nutpie", random_seed=RANDOM_SEED)
    trace_1
    return (trace_1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For our prediction, we want to pass the LD50 dose to the model. We will just take the mean of the posterior samples for the LD50.
    """)
    return


@app.cell
def _(trace_1):
    trace_1.posterior["ld50"].values.mean()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Swapping data is done with `pm.set_data`. Then, we sample from the **posterior predictive distribution** by calling `pm.sample_posterior_predictive`. This samples data from the model using the fitted parameters and the new data.

    The zero passed to `deaths_data` is just a dummy value, since we are not conditioning on any observed deaths.
    """)
    return


@app.cell
def _(dose_model_5, trace_1):
    with dose_model_5:
        pm.set_data(
            {
                "dose_data": [trace_1.posterior["ld50"].values.mean()],
                "deaths_data": [0],
            },
            coords={"dosis": ["ld50"]},
        )
        predictive_samples = pm.sample_posterior_predictive(trace_1)
    predictive_samples
    return (predictive_samples,)


@app.cell
def _(n_1, predictive_samples):
    def plot_ld50_predictive():
        predicted_deaths = predictive_samples.posterior_predictive["y"].values.flatten()
        counts = np.bincount(predicted_deaths.astype(int), minlength=n_1 + 1)
        probs = counts / counts.sum()
        fig = go.Figure(go.Bar(x=list(range(n_1 + 1)), y=probs))
        fig.update_layout(
            xaxis_title="Number of Deaths (out of 5)",
            yaxis_title="Probability",
            title="Posterior Predictive: Deaths at LD50 Dose",
            xaxis=dict(dtick=1),
        )
        return fig

    plot_ld50_predictive()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Or, we can generate predictions across a range of doses to see how the probability of death changes with dose.
    """)
    return


@app.cell
def _(dose_model_5, trace_1):
    with dose_model_5:
        pm.set_data(
            {"dose_data": np.linspace(-1, 1, 200), "deaths_data": [0] * 200},
            coords={"dosis": [f"d{i}" for i in range(1, 201)]},
        )
        predictive_samples_1 = pm.sample_posterior_predictive(
            trace_1, var_names=["y", "p"]
        )
    predictive_samples_1
    return (predictive_samples_1,)


@app.cell
def _(deaths_1, dose_1, n_1, predictive_samples_1):
    def plot_predictive_dose_response():
        p_samples = predictive_samples_1.posterior_predictive["p"].values
        dose_range = np.linspace(-1, 1, 200)
        flat_p = p_samples.reshape(-1, p_samples.shape[-1])

        fig = go.Figure()
        n_lines = 250
        idx = np.random.default_rng(42).choice(flat_p.shape[0], n_lines, replace=False)
        for i in idx:
            fig.add_trace(
                go.Scatter(
                    x=dose_range,
                    y=flat_p[i],
                    mode="lines",
                    line=dict(color="rgba(21, 74, 114, 0.5)", width=0.5),
                    showlegend=False,
                )
            )
        mean_prob = flat_p.mean(axis=0)
        fig.add_trace(
            go.Scatter(
                x=dose_range,
                y=mean_prob,
                mode="lines",
                line=dict(color="rgb(21, 74, 114)", width=3),
                name="Mean",
            )
        )
        empirical_p = deaths_1 / n_1
        fig.add_trace(
            go.Scatter(
                x=dose_1,
                y=empirical_p,
                mode="markers",
                marker=dict(color="red", size=8),
                name="Observed Data",
            )
        )
        fig.update_layout(xaxis_title="Log Dose", yaxis_title="Probability of Death")
        return fig

    plot_predictive_dose_response()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise: Smart Drug Clinical Trial

    Now that we have a flavor for how PyMC models are specified, let's try building a more complex model.

    We will use a fictitious example from Kruschke (2013) concerning the evaluation of a clinical trial for drug efficacy. The trial aims to evaluate whether a "smart drug" increases intelligence by comparing IQ scores of individuals in a treatment arm (those receiving the drug) to those in a control arm (those receiving a placebo). There are 47 individuals and 42 individuals in the treatment and control arms, respectively.

    Build a model to evaluate the evidence for the drug's effectiveness.

    The data are available in two variables prepared below: `iq`, an array of all
    89 IQ scores, and `group_id`, an integer indicator for each score
    (0 = drug, 1 = placebo).
    """)
    return


@app.cell
def _():
    # fmt: off
    drug_iq_scores = (
        101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105,
        103, 100, 95, 102, 106, 109, 102, 82, 102, 100, 102, 102, 101,
        102, 102, 103, 103, 97, 97, 103, 101, 97, 104, 96, 103, 124, 101,
        101, 100, 101, 101, 104, 100, 101
    )
    placebo_iq_scores = (
        99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100,
        105, 88, 101, 100, 104, 100, 100, 100, 101, 102, 103, 97, 101,
        101, 100, 101, 99, 101, 100, 100, 101, 100, 99, 101, 100, 102,
        99, 100, 99
    )
    # fmt: on

    drug = pl.DataFrame(dict(iq=drug_iq_scores, group="drug"))
    placebo = pl.DataFrame(dict(iq=placebo_iq_scores, group="placebo"))
    n1 = len(drug)
    n0 = len(placebo)
    trial_data = pl.concat([drug, placebo])

    def plot_iq_histogram():
        fig = px.histogram(
            trial_data,
            x="iq",
            color="group",
            barmode="overlay",
            histnorm="percent",
            labels={"iq": "IQ Score", "group": "Group"},
            title="Distribution of IQ Scores by Group",
            color_discrete_map={"drug": "#636EFA", "placebo": "#EF553B"},
        )
        fig.update_layout(
            legend_title_text="",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            xaxis_title="IQ Score",
            yaxis_title="Percentage",
        )
        return fig

    iq = trial_data.select("iq").to_numpy().squeeze()
    group_id = (
        trial_data["group"].cast(pl.Categorical).to_physical().to_numpy().squeeze()
    )
    plot_iq_histogram()
    return group_id, iq


@app.cell
def _(group_id, iq):
    def exercise_drug_model():
        drug_iq = iq[group_id == 0]
        placebo_iq = iq[group_id == 1]

        with pm.Model() as model:
            # YOUR CODE HERE — priors for each group's mean and a shared sigma
            # YOUR CODE HERE — likelihoods for drug_iq and placebo_iq
            # YOUR CODE HERE — an effect_size Deterministic (difference of means)
            ...
        return model

    return (exercise_drug_model,)


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Hint": mo.md(
                "We are interested in the expected difference in IQ between the treatment and control groups."
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    run_drug_model = mo.ui.run_button(label="▶ Run exercise")
    run_drug_model
    return (run_drug_model,)


@app.cell(hide_code=True)
def _(exercise_drug_model, run_drug_model):
    mo.stop(
        not run_drug_model.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_drug_model()
    return


@app.cell(hide_code=True)
def _(group_id, iq):
    def solution_trial_model():
        with pm.Model() as model:
            mu_drug = pm.Normal("mu_drug", 100, sigma=10)
            mu_placebo = pm.Normal("mu_placebo", 100, sigma=10)
            sigma = pm.HalfNormal("sigma", 10)
            pm.Normal("iq_drug", mu=mu_drug, sigma=sigma, observed=iq[group_id == 0])
            pm.Normal(
                "iq_placebo", mu=mu_placebo, sigma=sigma, observed=iq[group_id == 1]
            )
            pm.Deterministic("effect_size", mu_drug - mu_placebo)
        return model

    def extension_best_model():
        with pm.Model(coords=dict(group=["drug", "placebo"])) as model:
            mu = pm.Normal("mu", 100, sigma=10, dims="group")
            sigma = pm.Uniform("sigma", lower=0, upper=20, dims="group")
            nu = pm.Exponential("nu_minus_one", 1 / 30) + 1
            pm.StudentT(
                "like", nu=nu, mu=mu[group_id], sigma=sigma[group_id], observed=iq
            )
            pm.Deterministic("effect_size", mu[0] - mu[1])
        return model

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_trial_model)}\n```"),
                    mo.lazy(
                        lambda: solution_trial_model(),
                        show_loading_indicator=True,
                    ),
                ]
            ),
            "Extension: a robust BEST model": mo.vstack(
                [
                    mo.md(
                        "Kruschke's *Bayesian Estimation Supersedes the t-test* "
                        "(BEST) model makes two upgrades you will meet properly in "
                        "later sessions. First, instead of separate `mu_drug` / "
                        "`mu_placebo` variables it declares one vector `mu` with "
                        '`dims="group"` and picks each observation\'s mean by '
                        "**indexing with the group array**: `mu[group_id]`, the "
                        "core trick behind the hierarchical models of Session 4.2. "
                        "Second, it swaps the Normal likelihood for a **StudentT** "
                        "whose degrees of freedom `nu` are learned from the data "
                        "(`pm.Exponential(1/30) + 1` shifts the prior so `nu > 1`), "
                        "making the comparison robust to outliers like the two "
                        "IQ scores above 120."
                    ),
                    mo.md(f"```python\n{inspect.getsource(extension_best_model)}\n```"),
                    mo.lazy(
                        lambda: extension_best_model(),
                        show_loading_indicator=True,
                    ),
                ]
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
