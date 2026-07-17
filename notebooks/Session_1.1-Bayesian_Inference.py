import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


with app.setup:
    import marimo as mo
    import inspect
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import polars as pl
    import pymc as pm
    from scipy import stats
    import arviz as az
    import matplotlib.pyplot as plt
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
    plt.rcParams["savefig.dpi"] = 120
    plt.rcParams["figure.dpi"] = 120
    RANDOM_SEED = 42
    data_path = Path(__file__).parent / "data"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Session 1.1: Bayesian Inference

    This session introduces the core ideas of Bayesian inference through hands-on examples.

    **Topics:**
    Bayesian updating, Beta-Binomial model, A/B testing, continuous estimation, prior sensitivity, and shrinkage

    ---
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Bayes' Theorem

    At the heart of Bayesian inference is a simple formula for updating beliefs in light of evidence:

    $$
    P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}
    $$

    | Term | Name | Meaning |
    |------|------|---------|
    | $P(\theta)$ | **Prior** | What we believe about $\theta$ before seeing data |
    | $P(D \mid \theta)$ | **Likelihood** | How probable the data is, given a particular $\theta$ |
    | $P(D)$ | **Evidence** (marginal likelihood) | Total probability of the data across all $\theta$ |
    | $P(\theta \mid D)$ | **Posterior** | Our updated belief after seeing the data |

    Since $P(D)$ is just a normalising constant, we often write:

    $$
    P(\theta \mid D) \propto P(D \mid \theta) \, P(\theta)
    $$

    **Posterior is proportional to Likelihood times Prior.**

    The posterior is always a *compromise* between the prior and the data. With more data, the likelihood dominates; with less data, the prior has more influence.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Testing for a Rare Disease

    Suppose a disease affects **0.1%** of the population. A screening test has:

    - **Sensitivity** = 95% — it correctly detects 95% of people who have the disease
    - **Specificity** = 95% — it correctly clears 95% of people who are healthy (5% false positive rate)

    You take the test and it comes back **positive**. What is the probability you actually have the disease?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We have two hypotheses and one observation:

    - $H_1$: **you have the disease**, with prior $P(H_1) = 0.001$
    - $H_2$: **you are healthy**, with prior $P(H_2) = 0.999$
    - **Data**: one positive test

    The likelihoods come from the test's accuracy:

    - $P(+ \mid H_1) = 0.95$ (sensitivity)
    - $P(+ \mid H_2) = 0.05$ (false positive rate)

    Applying Bayes' theorem:

    $$
    P(H_1 \mid +) = \frac{P(+ \mid H_1)\,P(H_1)}{P(+ \mid H_1)\,P(H_1) + P(+ \mid H_2)\,P(H_2)}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    def plot_disease_update():
        prior_sick = 0.001
        prior_healthy = 1 - prior_sick
        sens = 0.95
        fpr = 0.05

        evidence = sens * prior_sick + fpr * prior_healthy
        post_sick = sens * prior_sick / evidence
        post_healthy = fpr * prior_healthy / evidence

        hypotheses = ["Disease", "Healthy"]
        priors = [prior_sick, prior_healthy]
        posteriors = [post_sick, post_healthy]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=hypotheses,
                y=priors,
                name="Prior",
                marker_color="#154A72",
                opacity=0.45,
                text=[f"{p:.3%}" for p in priors],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                x=hypotheses,
                y=posteriors,
                name="Posterior (after +)",
                marker_color="#81C240",
                opacity=0.85,
                text=[f"{p:.1%}" for p in posteriors],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Bayesian Update: Prior vs Posterior after One Positive Test",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1.1], tickformat=".0%"),
            barmode="group",
            width=650,
            height=420,
        )
        return fig, post_sick

    disease_fig, post_sick = plot_disease_update()

    mo.vstack(
        [
            disease_fig,
            mo.callout(
                mo.md(f"**P(disease | positive test) = {post_sick:.1%}**"),
                kind="warn",
            ),
            mo.md(r"""
            The positive test shifts the probability of disease from 0.1% to roughly 1.9%. The posterior is still small because the prior is so low that one positive test is not enough to outweigh it.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Why Does This Feel Wrong?

    Most people — including physicians — intuitively guess that the probability is somewhere around 95%, matching the test's accuracy. The actual answer of ~2% feels absurd. What's going on?

    The trap is treating the test's accuracy as if it directly answers the question we care about. But $P(+ \mid \text{disease})$ and $P(\text{disease} \mid +)$ are **not the same thing**, and they can differ by orders of magnitude. Reversing the direction of a conditional probability without accounting for the base rate is sometimes called the **base rate fallacy**, and it's one of the most robust findings in cognitive psychology.

    The key insight is that a rare disease means there are *far more* healthy people than sick people to begin with. Even with a low false positive rate, applying the test to a large healthy population generates many false alarms — enough to swamp the smaller number of true positives. The test *is* informative: it raises the probability of disease roughly twentyfold. But a twentyfold increase from a tiny base is still a small number.

    This is why Bayesian reasoning matters in practice. Whenever you're tempted to go straight from "the evidence looks strong" to "the hypothesis must be true," Bayes' theorem forces you to ask: *how likely was the hypothesis in the first place?*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## The Beta-Binomial Model

    In digital marketing, A/B testing is used to evaluate the effect of different versions of an ad, web page, etc. More generally, it has the same structure as tests of medical treatments, public policy interventions, product design, and more.

    As an example, suppose we test an email campaign that includes a promotional offer. We send version A to 100 people, and 8 of them accept the offer. So the observed conversion rate in this sample is 8%.
    """)
    return


@app.cell
def _():
    campaign_A = dict(n=100, k=8)
    return (campaign_A,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Some questions we'd like to answer:

    - What is the probability someone responds to this email?
    - How confident are we about that estimate?
    - If we send the same email to 100 more people, how many will respond?

    And if we have more than one version of the email, we want to compare them.

    To answer these questions, we'll use the beta-binomial model:

    - A beta distribution to represent what we believe about the conversion rate.
    - A binomial to represent the distribution of outcomes (number of responses).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The Beta Distribution

    The first step is to define a **prior distribution** that represents what we believe about conversion rates of ads like this one, in general, before we consider the data.

    We'll use a beta distribution because it's defined from 0 to 1, which is the right domain for a probability, and it has parameters we can choose to specify the location and width of the prior.

    ### Pseudocounts Interpretation

    The Beta distribution parameters have an intuitive interpretation as **pseudocounts** — imaginary prior observations:

    - $\alpha$ = number of prior "successes" (responses)
    - $\beta$ = number of prior "failures" (non-responses)
    - Prior mean = $\alpha / (\alpha + \beta)$
    - Prior "sample size" = $\alpha + \beta$

    After observing $k$ successes in $n$ trials, the posterior is:

    $$\text{Beta}(\alpha + k, \, \beta + n - k)$$

    This is called a **conjugate update** — the posterior has the same family as the prior. For Beta(2, 5): we act as if we've already seen 2 responses and 5 non-responses, so our prior "sample size" is 7.
    """)
    return


@app.cell(hide_code=True)
def _():
    def plot_beta_prior():
        prior_dist = pz.Beta(2, 5)
        fig, ax = plt.subplots(figsize=(7, 2.5))
        prior_dist.plot_pdf(ax=ax)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_xlabel("Conversion rate")
        ax.set_ylabel("Density")
        ax.set_title("Beta(2, 5) Prior Distribution")
        fig.tight_layout()
        return fig, prior_dist

    beta_prior_fig, beta_prior_dist = plot_beta_prior()

    mo.vstack(
        [
            mo.md(
                f"**Mean:** {beta_prior_dist.mean():.3f}, **Std:** {beta_prior_dist.std():.3f}"
            ),
            mo.md(
                "With these parameters, the mean is about 0.29 — we think a typical response rate is about 29%. But the width of the distribution indicates we are quite unsure."
            ),
            beta_prior_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The Binomial Distribution

    The prior tells us what we believe about the conversion rate $\theta$ before seeing data. We also need a **likelihood** — a probability model that links $\theta$ to the data we actually observe.

    Each email recipient either converts or doesn't, so we can think of every send as an independent Bernoulli trial with success probability $\theta$. The total number of conversions from $n$ independent trials is then distributed as:

    $$
    k \sim \text{Binomial}(n, \theta)
    $$

    This gives us $P(k \mid \theta)$ — the probability of observing $k$ conversions for any given value of $\theta$. Bayes' theorem will combine this with the prior to produce the posterior.

    To build intuition, let's pick an arbitrary single value from the prior — say $\theta = 0.2$ — and ask what the Binomial says about the number of conversions we'd expect from 50 emails.
    """)
    return


@app.cell(hide_code=True)
def _():
    def plot_binomial_demo():
        binom_dist = pz.Binomial(50, 0.2)
        fig, ax = plt.subplots(figsize=(7, 2.5))
        binom_dist.plot_pdf(color="C1", ax=ax)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_xlabel("Number of conversions")
        ax.set_ylabel("PMF")
        ax.set_title("Binomial(50, 0.2)")
        fig.tight_layout()
        return fig, binom_dist

    binom_fig, binom_demo_dist = plot_binomial_demo()

    mo.vstack(
        [
            mo.md(
                f"**Mean:** {binom_demo_dist.mean():.1f} — but values from 2 to 20 have non-negligible probability."
            ),
            binom_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## How Posteriors Evolve with Data

    Before we build a PyMC model, let's build intuition for the most fundamental property of Bayesian inference: **more data means less uncertainty**.

    Suppose we want to estimate the proportion of female births in a population. We start with a uniform prior (any value from 0 to 1 is equally likely) and update it as we observe data.

    The unnormalized posterior for $\theta$ given $y$ successes in $n$ trials is:

    $$P(\theta \mid y, n) \propto \theta^y (1 - \theta)^{n - y}$$
    """)
    return


@app.cell(hide_code=True)
def _():
    def plot_posterior_evolution():
        from plotly.subplots import make_subplots

        theta = np.linspace(0, 1, 1001)

        datasets = [
            (5, 3, "n = 5, y = 3"),
            (20, 9, "n = 20, y = 9"),
            (750, 365, "n = 750, y = 365"),
        ]

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[d[2] for d in datasets],
            shared_yaxes=False,
        )

        for col, (n, y, label) in enumerate(datasets, 1):
            posterior = theta**y * (1 - theta) ** (n - y)
            posterior = posterior / (posterior.sum() * (theta[1] - theta[0]))

            fig.add_trace(
                go.Scatter(
                    x=theta,
                    y=np.ones_like(theta),
                    mode="lines",
                    name="Prior (uniform)",
                    line=dict(color="#154A72", width=1.5, dash="dash"),
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=theta,
                    y=posterior,
                    mode="lines",
                    name="Posterior",
                    line=dict(color="#81C240", width=2.5),
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
            fig.update_xaxes(title_text="θ", row=1, col=col)

        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_layout(
            title="Posterior Evolution: More Data → Less Uncertainty",
            width=900,
            height=350,
        )
        return fig

    posterior_evolution_fig = plot_posterior_evolution()

    mo.vstack(
        [
            posterior_evolution_fig,
            mo.md("""
        With only 5 observations the posterior is wide — many values of θ are plausible.
        With 750 observations it's tightly concentrated near the true value (~0.487).
        **The data overwhelm the prior.**
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In general we don't know the response rate — we have to estimate it from data.

    ---

    ## The Bayesian Update

    The fundamental idea in Bayesian statistics is that we can use data to update our beliefs.

    - We start with a **prior distribution** that represents what we believe before we see the data
    - We compute a **posterior distribution** that represents what we believe after we see the data

    There are several ways to do this computation:

    - Mathematically (conjugate updates), or
    - Computationally, using grid approximation, variational inference, or **MCMC**

    MCMC is [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) sampling, which takes a prior distribution and data, and generates a random sample from an approximate posterior distribution.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Building the Model in PyMC

    PyMC provides several MCMC algorithms along with distributions we can use to assemble a model of the data-generating process.

    In this example, we model the data-generating process as:

    1. Choose a conversion rate from the **prior**, which is a Beta distribution
    2. Choose a number of conversions from the **likelihood**, which is a Binomial distribution
    """)
    return


@app.cell
def _(campaign_A):
    campaign_A
    return


@app.cell
def _(campaign_A):
    with pm.Model() as beta_binom_model:
        conversion_rate = pm.Beta("conversion_rate", alpha=2, beta=5)
        pm.Binomial(
            "obs", p=conversion_rate, n=campaign_A["n"], observed=campaign_A["k"]
        )

    pm.model_to_graphviz(beta_binom_model)
    return (beta_binom_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Before we look at any data, let's draw samples from the **prior** distribution. This serves two purposes:

    1. **Sanity check**: it confirms that PyMC has encoded the model we intended — the sampled values should trace out the Beta(2, 5) curve we plotted earlier.
    2. **Prior predictive check**: it lets us see what the model believes *before* seeing data. If the prior generates implausible values, that's a signal the prior is misspecified and should be revised before we ever touch the observed data.
    """)
    return


@app.cell
def _(beta_binom_model):
    with beta_binom_model:
        prior_samples_ab = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    prior_samples_ab
    return (prior_samples_ab,)


@app.cell(hide_code=True)
def _(prior_samples_ab):
    prior_pred_counts = prior_samples_ab.prior_predictive["obs"].values.flatten()

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(prior_pred_counts, bins=range(0, 102, 2), alpha=0.7)
    ax.axvline(8, color="C1", linestyle="--", label="Observed (k = 8)")
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("Number of conversions (out of 100)")
    ax.set_ylabel("Frequency")
    ax.set_title("Prior Predictive Check")
    ax.legend()
    fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                "The histogram shows the distribution of conversion counts implied by the Beta(2, 5) prior, applied to a campaign of n = 100 recipients: for each draw of `conversion_rate` from the prior, simulate one count from `Binomial(100, conversion_rate)`. The dashed line marks the value we actually observed (k = 8). Because the prior is broad, the predictive distribution covers a wide range of plausible counts — but it doesn't put weight on absurd values, so the prior isn't ruling out anything reasonable before we fit."
            ),
            fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now let's sample from the **posterior** — our beliefs after combining the prior with the observed data.

    For simple conjugate problems like this one, the posterior has a closed form. In general, though, the posterior is a complicated function of the prior, the likelihood, and the data, and we can't compute it exactly. PyMC instead draws a large random sample from the posterior, and we use that sample to represent the distribution: to compute means, intervals, probabilities, and predictions.
    """)
    return


@app.cell
def _(beta_binom_model):
    with beta_binom_model:
        idata_ab = pm.sample(random_seed=RANDOM_SEED)
    idata_ab
    return (idata_ab,)


@app.cell(hide_code=True)
def _(idata_ab):
    def plot_posterior_conversion_rate():
        fig, ax = plt.subplots(figsize=(7, 2.5))
        samples = idata_ab.posterior["conversion_rate"].values.flatten()
        _grid, _pdf, _ = az.kde(samples)
        ax.plot(_grid, _pdf)
        ax.set_xlabel("Conversion rate")
        ax.set_ylabel("Density")
        ax.set_title("Posterior: Conversion Rate")
        fig.tight_layout()
        return fig

    posterior_cr_fig = plot_posterior_conversion_rate()

    mo.vstack(
        [
            mo.md("""
            The posterior distribution represents what we believe **after seeing the data**. It's centered near the observed conversion rate of 8%, but it's pulled slightly toward the prior mean of 0.29 — a small amount of shrinkage, because with only 100 trials the prior still has a little influence.

            The 94% HDI (highest density interval) gives us a direct probability statement: there is a 94% chance that the true conversion rate lies in this range, *subject to the assumptions of the model* — for example, that the conversion rate doesn't change over time, and that recipients respond independently. This is a stronger claim than a frequentist 94% confidence interval, which is a statement about the long-run coverage of an interval procedure rather than about the parameter itself.
            """),
            posterior_cr_fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## A/B Testing: Comparing Two Versions

    So far we've estimated the conversion rate for a single campaign. But most real decisions involve a comparison: is version B better than version A? A frequentist approach to this question typically reaches for a hypothesis test, which ultimately delivers a binary verdict — "significant" or "not significant."

    But rather than merely distinguishing between *"yes, there's a difference"* and *"no, there isn't"*, it is more useful to compute the full distribution of possible differences. We want to know **how different** the conversion rates are, not just **whether** they differ — and with what probability one is larger than the other, and by how much.

    Now suppose we introduce a new version B of the email. We think it might be more effective, so we run an A/B test.
    """)
    return


@app.cell
def _():
    ab_data_A = dict(n=180, k=16)
    ab_data_B = dict(n=20, k=5)
    return ab_data_A, ab_data_B


@app.cell
def _(ab_data_A, ab_data_B):
    with pm.Model() as ab_model:
        cr_A = pm.Beta("conversion_rate_A", alpha=2, beta=5)
        cr_B = pm.Beta("conversion_rate_B", alpha=2, beta=5)
        pm.Binomial("obs_A", p=cr_A, n=ab_data_A["n"], observed=ab_data_A["k"])
        pm.Binomial("obs_B", p=cr_B, n=ab_data_B["n"], observed=ab_data_B["k"])

        idata_ab_test = pm.sample(random_seed=RANDOM_SEED)
    pm.model_to_graphviz(ab_model)
    return (idata_ab_test,)


@app.cell(hide_code=True)
def _(idata_ab_test):
    def plot_ab_comparison():
        posterior = az.extract(idata_ab_test)
        samples_A = posterior["conversion_rate_A"].values
        samples_B = posterior["conversion_rate_B"].values

        fig, axes = plt.subplots(1, 2, figsize=(12, 3))

        # Plot overlapping posteriors
        _g_a, _p_a, _ = az.kde(samples_A)
        axes[0].plot(_g_a, _p_a, label="A")
        _g_b, _p_b, _ = az.kde(samples_B)
        axes[0].plot(_g_b, _p_b, color="C1", label="B")
        axes[0].set_xlabel("Conversion Rate")
        axes[0].set_title("Posterior Distributions")
        axes[0].legend()

        # Plot the difference
        diff = samples_B - samples_A
        _g_d, _p_d, _ = az.kde(diff)
        axes[1].plot(_g_d, _p_d)
        axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("B - A")
        axes[1].set_title("Posterior Difference (B - A)")

        plt.tight_layout()

        p_B_better = (samples_B > samples_A).mean()
        return fig, p_B_better

    ab_compare_fig, ab_p_B_better = plot_ab_comparison()

    mo.vstack(
        [
            ab_compare_fig,
            mo.md(f"""
            **Probability that B is better than A:** {ab_p_B_better:.1%}

            Rather than a binary hypothesis test, we get the full **distribution** of possible differences. We know *how different* the conversion rates are, not just *whether* they are different.

            At any point in time, these posterior distributions represent what we believe about A and B. A natural next question is: *how do we put those beliefs into action?* We'll come back to that when we look at Bayesian bandits.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Bayesian Estimation: Continuous Data

    So far we've modelled **proportions** (binary outcomes). But Bayesian inference works just as naturally with **continuous measurements**.

    **Radon** is a naturally occurring radioactive gas that is the second leading cause of lung cancer. The US Environmental Protection Agency (EPA) action level is **4 pCi/L** (on the log scale, $\log(4) \approx 1.386$).

    We have measurements of indoor radon levels across Minnesota. Let's estimate the typical (log) radon level in **Hennepin County** (Minneapolis) and ask: *does the average home exceed the safe level?*
    """)
    return


@app.cell(hide_code=True)
def _():
    hennepin_radon = (
        pl.read_csv(data_path / "radon.csv")
        .filter(pl.col("county") == "HENNEPIN")
        .select("log_radon")
    )

    radon_values = hennepin_radon["log_radon"].to_numpy()

    radon_hist = go.Figure()
    radon_hist.add_trace(
        go.Histogram(
            x=radon_values[~np.isnan(radon_values)],
            nbinsx=30,
            histnorm="probability density",
            marker_color=PYMC_BLUE,
            opacity=0.7,
        )
    )
    radon_hist.add_vline(
        x=np.log(4),
        line_dash="dot",
        line_color="red",
        annotation_text="EPA action level (log 4)",
    )
    radon_hist.update_layout(
        title=f"Log-Radon Levels in Hennepin County (n = {len(radon_values)})",
        xaxis_title="log(radon) pCi/L",
        yaxis_title="Density",
        width=700,
        height=400,
    )

    mo.vstack(
        [
            radon_hist,
            mo.md(
                f"**{len(radon_values)} measurements.** The red line marks the EPA action level."
            ),
        ]
    )
    return (hennepin_radon,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model Specification

    We model the log-radon measurements as normally distributed:

    $$
    \begin{align}
    \mu &\sim \text{Normal}(0, 10) \\
    \sigma &\sim \text{HalfNormal}(5) \\
    y_i &\sim \text{Normal}(\mu, \sigma)
    \end{align}
    $$

    - **$\mu$**: the average log-radon level (wide prior centered at 0)
    - **$\sigma$**: the measurement spread (constrained to be positive)
    - **$y_i$**: each observed measurement
    """)
    return


@app.cell
def _(hennepin_radon):
    with pm.Model() as radon_model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal(
            "y", mu=mu, sigma=sigma, observed=hennepin_radon["log_radon"].to_numpy()
        )

    pm.model_to_graphviz(radon_model)
    return (radon_model,)


@app.cell
def _(radon_model):
    with radon_model:
        radon_prior = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    radon_prior
    return (radon_prior,)


@app.cell(hide_code=True)
def _(radon_prior):
    prior_pred_y = radon_prior.prior_predictive["y"].values.flatten()

    radon_prior_fig = go.Figure()
    radon_prior_fig.add_trace(
        go.Histogram(
            x=np.clip(prior_pred_y, -50, 50),
            nbinsx=80,
            histnorm="probability density",
            marker_color="#154A72",
            opacity=0.6,
            name="Prior predictive",
        )
    )
    radon_prior_fig.update_layout(
        title="Prior Predictive Check: Simulated Radon Measurements",
        xaxis_title="log(radon)",
        yaxis_title="Density",
        width=700,
        height=350,
    )

    mo.vstack(
        [
            radon_prior_fig,
            mo.md("""
        The prior predictive shows what data we'd expect **before** seeing real measurements.
        With wide priors, the simulated range is enormous — but it includes the plausible range, so the priors aren't ruling out reasonable values.
        """),
        ]
    )
    return


@app.cell
def _(radon_model):
    with radon_model:
        radon_idata = pm.sample(random_seed=RANDOM_SEED)
    radon_idata
    return (radon_idata,)


@app.cell(hide_code=True)
def _(radon_idata):
    def plot_radon_posterior():
        fig, ax = plt.subplots(figsize=(8, 2.5))
        _mu = radon_idata.posterior["mu"].values.flatten()
        _grid, _pdf, _ = az.kde(_mu)
        ax.plot(_grid, _pdf)
        ax.axvline(np.log(4), color="C1", linestyle="--", label="log(4 pCi/L)")
        ax.legend()
        ax.set_title("Posterior: Mean Log-Radon (μ)")
        fig.tight_layout()
        return fig

    radon_posterior_fig = plot_radon_posterior()

    mu_samples = radon_idata.posterior["mu"].values.flatten()
    prob_above = float((mu_samples > np.log(4)).mean())

    mo.vstack(
        [
            radon_posterior_fig,
            mo.callout(
                mo.md(f"""
            **P(μ > log(4)) = {prob_above:.1%}** — the probability that the *average* home in Hennepin County exceeds the EPA action level.

            Note: individual homes can still exceed the threshold even if the average is below it.
            """),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(radon_idata):
    def plot_posterior_predictive_check():
        posterior = az.extract(radon_idata)
        mus = posterior["mu"].values
        sigmas = posterior["sigma"].values
        y_sim = stats.norm(loc=mus, scale=sigmas).rvs()

        observed = radon_idata.observed_data["y"].values

        x_grid = np.linspace(-3, 5, 300)
        kde_obs = stats.gaussian_kde(observed)(x_grid)
        kde_sim = stats.gaussian_kde(y_sim)(x_grid)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=kde_obs,
                mode="lines",
                name="Observed",
                line=dict(color="#154A72", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=kde_sim,
                mode="lines",
                name="Posterior predictive",
                line=dict(color="#81C240", width=2.5, dash="dash"),
            )
        )
        fig.update_layout(
            title="Posterior Predictive Check",
            xaxis_title="log(radon)",
            yaxis_title="Density",
            width=700,
            height=350,
        )
        return fig

    ppc_fig = plot_posterior_predictive_check()

    mo.vstack(
        [
            ppc_fig,
            mo.md(r"""
            The **posterior predictive distribution** is a third distribution type, alongside the prior and the posterior. It represents the distribution of new observations we'd expect if we ran the same experiment again, given our current beliefs about the parameters. In code: for each draw of $(\mu, \sigma)$ from the posterior, simulate one new observation from $\text{Normal}(\mu, \sigma)$.

            It is our main tool for **model checking**: if the simulated data doesn't resemble the observed data, the model is misspecified. Here the two densities overlap closely, so the Normal likelihood is a reasonable fit for these measurements.

            The posterior predictive is wider than the posterior on $\mu$ because it combines **two distinct sources of uncertainty**:

            - **Epistemic** uncertainty — we don't know the exact values of $\mu$ and $\sigma$. This uncertainty shrinks as we collect more data; with enough measurements, the posterior on $\mu$ becomes nearly a point.
            - **Aleatoric** uncertainty — even if we knew $\mu$ and $\sigma$ exactly, individual homes would still vary around the mean. This uncertainty does *not* shrink with more data — it's baked into the data-generating process.

            This distinction matters whenever you report predictions: a narrow posterior on the mean tells you you're confident about the *average* home, but it says nothing about how much individual homes scatter around that average.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(r"""
        ---

        ## Prior Sensitivity and Shrinkage

        The previous examples use a **weakly informative prior** — it "lets the data speak for itself." That's a reasonable default when you have a lot of data, because the likelihood will dominate the prior in the posterior anyway. But when you have only a small dataset, the prior makes a big difference, and "letting the data speak for itself" can produce wildly unrealistic estimates.

        ### The Rookie Problem

        *Adapted from Chapter 4 of Allen Downey's* Think Bayes.
        """),
            mo.callout(
                mo.md("""In Major League Baseball, most players have a batting average between .170 and .310.
            Suppose a player appearing in their first game gets 3 hits out of 3 attempts.
            What is the posterior distribution for their probability of getting a hit?"""),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    def plot_weak_prior_batting():
        # Weak prior: Beta(2, 5)
        with pm.Model():
            p_weak = pm.Beta("p", 2, 5)
            pm.Binomial("obs", p=p_weak, n=3, observed=3)
            idata_weak = pm.sample(random_seed=RANDOM_SEED)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        samples = idata_weak.posterior["p"].values.flatten()
        _grid, _pdf, _ = az.kde(samples)
        ax.plot(_grid, _pdf)
        ax.set_xlabel("Batting average")
        ax.set_ylabel("Density")
        ax.set_title("Posterior with Weak Prior Beta(2, 5)")
        fig.tight_layout()
        return fig

    weak_prior_fig = plot_weak_prior_batting()

    mo.vstack(
        [
            weak_prior_fig,
            mo.md(
                "With a weak prior, the posterior is pulled toward 1.0 by the data (3/3). This is clearly unrealistic — no one bats 1.000 over a full season."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We have a lot of information about batting averages in the major leagues. The typical average is around .250 with a standard deviation of about .030. Let's use that to build an **informative prior**.
    """)
    return


@app.function
def compute_beta_params(mean, std):
    """Convert mean and standard deviation to Beta distribution parameters."""
    alpha = mean * ((mean * (1 - mean)) / std**2 - 1)
    beta = (1 - mean) * ((mean * (1 - mean)) / std**2 - 1)
    return alpha, beta


@app.cell(hide_code=True)
def _():
    def compute_batting_beta_params():
        a, b = compute_beta_params(mean=0.250, std=0.030)
        return a, b

    beta_param_a, beta_param_b = compute_batting_beta_params()
    mo.md(
        f"Beta parameters for mean=0.250, std=0.030: **alpha={beta_param_a:.1f}, beta={beta_param_b:.1f}**"
    )
    return


@app.cell(hide_code=True)
def _():
    def plot_informative_prior_batting():
        a, b = compute_beta_params(mean=0.250, std=0.030)

        with pm.Model():
            p_info = pm.Beta("p", a, b)
            pm.Binomial("obs", p=p_info, n=3, observed=3)
            idata_info = pm.sample(random_seed=RANDOM_SEED)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        samples = idata_info.posterior["p"].values.flatten()
        _grid, _pdf, _ = az.kde(samples)
        ax.plot(_grid, _pdf)
        ax.set_xlabel("Batting average")
        ax.set_ylabel("Density")
        ax.set_title("Posterior with Informative Prior")
        fig.tight_layout()
        return fig

    info_prior_fig = plot_informative_prior_batting()

    mo.vstack(
        [
            info_prior_fig,
            mo.md(r"""
            With a strongly informative prior, the outcome of three at-bats barely moves the needle. This is the Bayesian version of [**shrinkage**](https://en.wikipedia.org/wiki/Shrinkage_(statistics)) — so called because an estimate based on small data "shrinks" toward the prior mean.

            In frequentist statistics, shrinkage is a regularization technique that has to be deliberately bolted on (think ridge regression, James–Stein estimators). In Bayesian statistics, it emerges *automatically* from the prior-posterior compromise: whenever the likelihood is weak relative to the prior, the posterior is pulled toward the prior, and extreme point estimates are tamed.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Baseball Batting Averages: Hierarchical Shrinkage

    With only one player the hierarchical model isn't necessary. But with hundreds of players, a hierarchical model learns about players *in general* from the entire dataset, and uses that to improve estimates for individual players — especially those with few at-bats.
    """)
    return


@app.cell(hide_code=True)
def _():
    batting_df = (
        pl.read_csv(data_path / "batting_2023.csv", encoding="utf8")
        .drop_nulls(subset=["BA"])
        .filter(pl.col("AB") > 0)
    )
    mo.vstack(
        [
            mo.md(f"**{batting_df.height} players** in the 2023 MLB batting dataset"),
            batting_df.select("Name", "Tm", "AB", "H", "BA").head(10),
        ]
    )
    return (batting_df,)


@app.cell(hide_code=True)
def _(batting_df):
    def plot_batting_average_histogram():
        ba_vals = batting_df["BA"].to_numpy()

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=ba_vals[~np.isnan(ba_vals)],
                nbinsx=50,
                histnorm="probability density",
                marker_color=PYMC_BLUE,
                opacity=0.7,
            )
        )
        fig.update_layout(
            title="Distribution of Batting Averages (2023 MLB)",
            xaxis_title="Batting Average",
            yaxis_title="Density",
            width=700,
            height=400,
        )
        return fig

    ba_hist_fig = plot_batting_average_histogram()

    mo.vstack(
        [
            ba_hist_fig,
            mo.md(
                "Notice that the most extreme batting averages (very low and very high) tend to belong to players with few at-bats."
            ),
        ]
    )
    return


@app.cell
def _(batting_df):
    def build_baseball_model():
        at_bats = batting_df["AB"].to_numpy()
        hits = batting_df["H"].to_numpy()
        n_players = len(batting_df)

        with pm.Model() as model:
            alpha_hyper = pm.Gamma("alpha", 16, 8)
            beta_hyper = pm.Gamma("beta", 40, 8)
            player_ba = pm.Beta(
                "player_ba", alpha=alpha_hyper, beta=beta_hyper, shape=n_players
            )
            pm.Binomial("hits", p=player_ba, n=at_bats, observed=hits)
        return model

    baseball_model = build_baseball_model()
    pm.model_to_graphviz(baseball_model)
    return (baseball_model,)


@app.cell
def _(baseball_model):
    with baseball_model:
        idata_baseball = pm.sample(random_seed=RANDOM_SEED)
    idata_baseball
    return (idata_baseball,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Before looking at shrinkage, here is the informative prior implied by the
    fitted hierarchical model. First, the posterior densities of the
    hyperparameters `alpha` and `beta`, which land close to 7.5 and 23; then
    raw draws from that `Beta(7.5, 23)` distribution itself — the
    population-level batting-average curve every player's estimate gets
    shrunk toward below.
    """)
    return


@app.cell
def _(idata_baseball):
    az.plot_dist(idata_baseball, var_names=["alpha", "beta"])
    return


@app.cell
def _():
    plt.hist(pm.draw(pm.Beta.dist(7.5, 23), 1000), bins=50)
    return


@app.cell(hide_code=True)
def _(batting_df, idata_baseball):
    def plot_shrinkage():
        observed_ba = batting_df["BA"].to_numpy()
        at_bats = batting_df["AB"].to_numpy()
        posterior_ba = idata_baseball.posterior["player_ba"].values
        posterior_mean_ba = np.mean(posterior_ba, axis=(0, 1))

        # Bounds: clip extreme observed outliers via percentiles so a 1/1
        # perfect batter doesn't blow out the axis. Use the same range for
        # both axes so the 1:1 diagonal is visually meaningful.
        obs_lo = float(np.nanpercentile(observed_ba, 1))
        obs_hi = float(np.nanpercentile(observed_ba, 99))
        post_lo = float(np.nanmin(posterior_mean_ba))
        post_hi = float(np.nanmax(posterior_mean_ba))
        lo = min(obs_lo, post_lo) - 0.02
        hi = max(obs_hi, post_hi) + 0.02

        # Sqrt scaling of marker size so small-AB players stay visible
        marker_size = np.sqrt(at_bats) * 0.7 + 6

        fig = go.Figure()
        # 1:1 reference line drawn first so points sit on top
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=observed_ba,
                y=posterior_mean_ba,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=at_bats,
                    colorscale=[[0, PYMC_GREEN], [1, PYMC_BLUE]],
                    colorbar=dict(title="At Bats", thickness=15, len=0.85, x=1.02),
                    opacity=0.75,
                    line=dict(width=0.5, color="white"),
                ),
                text=[
                    f"{n}: AB={ab}"
                    for n, ab in zip(batting_df["Name"].to_list(), at_bats)
                ],
                hoverinfo="text+x+y",
                showlegend=False,
            )
        )
        fig.update_layout(
            title="Shrinkage: Observed vs Posterior Mean Batting Average",
            xaxis=dict(title="Observed BA", range=[lo, hi], constrain="domain"),
            yaxis=dict(title="Posterior Mean BA", range=[lo, hi], constrain="domain"),
            width=560,
            height=560,
            showlegend=False,
            margin=dict(l=60, r=80, t=60, b=50),
        )
        return fig

    shrinkage_fig = plot_shrinkage()

    mo.vstack(
        [
            shrinkage_fig,
            mo.md("""
            Players with extreme observed batting averages (from few at-bats) are **shrunk** toward the league average. Players with many at-bats stay closer to their observed values.

            This is the hierarchical model at work — it learns the overall distribution and uses it to regularize individual estimates.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Exercise: Prior Sensitivity in A/B Testing

    Build a PyMC model for an A/B test where version A got 12 conversions out of 150 emails, and version B got 8 conversions out of 50 emails.

    1. Use a `Beta(1, 1)` (uniform) prior for both conversion rates
    2. Sample the posterior and compute `P(B > A)`
    3. Then change the prior to `Beta(2, 20)` (informed: low conversion expected) and compare
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md("""
        **Your task:** Build two models — one with `Beta(1, 1)` priors and one with `Beta(2, 20)` priors. Compare `P(B > A)` under the two priors.

        Pass `random_seed=RANDOM_SEED` to `pm.sample` for reproducibility. Write your models inside the `_exercise_prior_sensitivity` function in the scaffold cell below — it must set both `p_b_better_uniform` and `p_b_better_informed` before the markdown summary renders.
        """),
        kind="info",
    )
    return


@app.cell
def _():
    def _exercise_prior_sensitivity():
        # YOUR CODE HERE — build both models (Beta(1, 1) and Beta(2, 20) priors),
        # sample each, and compute P(B > A) under both.
        p_b_better_uniform = ...
        p_b_better_informed = ...
        if p_b_better_uniform is ... or p_b_better_informed is ...:
            return mo.callout(
                mo.md("Replace the `...` placeholders above, then re-run this cell."),
                kind="info",
            )
        return mo.md(
            f"P(B > A) — uniform prior: **{p_b_better_uniform:.1%}**, "
            f"informed prior: **{p_b_better_informed:.1%}**"
        )

    _exercise_prior_sensitivity()
    return


@app.cell(hide_code=True)
def _():
    def solution_prior_sensitivity():
        results = {}
        for label, a, b in [
            ("Uniform Beta(1,1)", 1, 1),
            ("Informed Beta(2,20)", 2, 20),
        ]:
            with pm.Model():
                cr_a = pm.Beta("cr_A", a, b)
                cr_b = pm.Beta("cr_B", a, b)
                pm.Binomial("obs_A", p=cr_a, n=150, observed=12)
                pm.Binomial("obs_B", p=cr_b, n=50, observed=8)
                idata = pm.sample(random_seed=RANDOM_SEED)
            post = az.extract(idata)
            p_b_better = float((post["cr_B"].values > post["cr_A"].values).mean())
            results[label] = (idata, p_b_better)

        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        for idx, (label, (idata, p_win)) in enumerate(results.items()):
            post = az.extract(idata)
            grid_a, pdf_a, _ = az.kde(post["cr_A"].values)
            axes[idx].plot(grid_a, pdf_a, label="A")
            grid_b, pdf_b, _ = az.kde(post["cr_B"].values)
            axes[idx].plot(grid_b, pdf_b, color="C1", label="B")
            axes[idx].set_title(f"{label}\nP(B>A) = {p_win:.1%}")
            axes[idx].set_xlabel("Conversion Rate")
            axes[idx].legend()
        fig.tight_layout()
        return fig

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        f"```python\n{inspect.getsource(solution_prior_sensitivity)}\n```"
                    ),
                    mo.lazy(solution_prior_sensitivity, show_loading_indicator=True),
                    mo.md(
                        "The informative `Beta(2, 20)` prior pulls both estimates toward "
                        "a lower conversion rate and can change the relative comparison "
                        "between A and B, especially with small samples."
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

    ## Bayesian Bandits: Adaptive Experimentation

    Earlier we built a posterior for the conversion rate of email campaigns A and B. But how do we put those beliefs into action? A traditional A/B test says: pick a sample size up front, send half of the next batch of emails to A and half to B, then decide. This is simple and statistically rigorous, but it has a real cost — while the test is running, half of every batch goes to the *worse* campaign.

    Bayesian bandits take a different approach: send each email to A or B in **proportion to its probability of being the better campaign**. As long as we're unsure, A and B get roughly equal shares of the mailing list. As we become confident that one converts better, it claims more of the next batch. This naturally balances **exploration** (occasionally sending to the worse-looking campaign to learn more) with **exploitation** (sending to the campaign we currently believe is best).

    **Thompson Sampling** is a remarkably simple algorithm that implements this idea:

    1. Maintain a Beta posterior over the conversion rate for each campaign
    2. For each new email, **sample** one conversion rate from each campaign's posterior
    3. **Send** that email using whichever campaign's sample was highest
    4. **Update** that campaign's posterior with the observed outcome (converted or not)

    There's no explicit exploration/exploitation tuning parameter here. The randomness in step 2 does all the work: when the two posteriors heavily overlap, both campaigns are chosen often; when one pulls away, its samples win more often.

    Before turning the whole loop into an algorithm, let's walk through one cycle by hand. We'll start fresh, simulate a small uneven test, fit a model, use the posterior to choose the next allocation, then re-fit and watch our beliefs update.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First, a small helper that simulates running a campaign: it draws the number of conversions from a Binomial with the campaign's true rate, then adds the new emails sent and conversions to a running total. Each campaign is just a dictionary with its (unknown to the algorithm) true rate `p`, the cumulative number of emails sent `n`, and the cumulative number of conversions `k`.

    Note that — unlike the original notebook example — `bandit_run_campaign` returns a *new* dictionary rather than mutating the one we pass in. This keeps the function safe to call from marimo's reactive cells, where re-running a cell that mutated state would silently double-count.
    """)
    return


@app.cell
def _():
    def bandit_run_campaign(state, n, seed):
        """Simulate sending ``n`` emails for ``state`` and return updated totals."""
        rng = np.random.default_rng(seed)
        k = int(rng.binomial(n, state["p"]))
        return {"p": state["p"], "n": state["n"] + n, "k": state["k"] + k}

    return (bandit_run_campaign,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Suppose campaign A has a true conversion rate of 10% and campaign B is actually a bit better at 15% — though of course the algorithm doesn't know that.
    """)
    return


@app.cell
def _():
    bandit_state_A0 = {"p": 0.10, "n": 0, "k": 0}
    bandit_state_B0 = {"p": 0.15, "n": 0, "k": 0}
    bandit_state_A0, bandit_state_B0
    return bandit_state_A0, bandit_state_B0


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We don't yet know that B is better, so we run an initial uneven A/B test: 80 emails for A, 20 for B. (We tilt toward A because it's the incumbent.)
    """)
    return


@app.cell
def _(bandit_run_campaign, bandit_state_A0, bandit_state_B0):
    bandit_state_A1 = bandit_run_campaign(bandit_state_A0, 80, seed=11)
    bandit_state_B1 = bandit_run_campaign(bandit_state_B0, 20, seed=12)
    bandit_state_A1, bandit_state_B1
    return bandit_state_A1, bandit_state_B1


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we update our beliefs about each campaign's conversion rate by fitting the same Beta–Binomial model from the A/B section, but using the *current* totals as observed data.
    """)
    return


@app.cell
def _(bandit_state_A1, bandit_state_B1):
    with pm.Model() as bandit_model_v1:
        cr_A_v1 = pm.Beta("conversion_rate_A", alpha=2, beta=5)
        cr_B_v1 = pm.Beta("conversion_rate_B", alpha=2, beta=5)
        pm.Binomial(
            "obs_A", p=cr_A_v1, n=bandit_state_A1["n"], observed=bandit_state_A1["k"]
        )
        pm.Binomial(
            "obs_B", p=cr_B_v1, n=bandit_state_B1["n"], observed=bandit_state_B1["k"]
        )
        bandit_idata_v1 = pm.sample(random_seed=RANDOM_SEED)
    return (bandit_idata_v1,)


@app.cell(hide_code=True)
def _(bandit_idata_v1):
    def plot_bandit_posteriors(idata, title):
        samples_A = idata.posterior["conversion_rate_A"].values.flatten()
        samples_B = idata.posterior["conversion_rate_B"].values.flatten()
        p_sup_B = float((samples_B > samples_A).mean())

        # Smooth densities via Gaussian KDE
        grid = np.linspace(0.0, 0.5, 400)
        kde_A = stats.gaussian_kde(samples_A)(grid)
        kde_B = stats.gaussian_kde(samples_B)(grid)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=kde_A,
                name="A posterior",
                line=dict(color=PYMC_BLUE, width=2),
                fill="tozeroy",
                fillcolor="rgba(21, 74, 114, 0.18)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=kde_B,
                name="B posterior",
                line=dict(color=PYMC_GREEN, width=2),
                fill="tozeroy",
                fillcolor="rgba(129, 194, 64, 0.18)",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="conversion rate θ",
            yaxis_title="density",
            height=320,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig, p_sup_B

    bandit_post_fig_v1, bandit_p_sup_B_v1 = plot_bandit_posteriors(
        bandit_idata_v1, "Posteriors after first batch (100 emails)"
    )
    mo.vstack(
        [
            bandit_post_fig_v1,
            mo.md(
                f"On average B looks better, but the posteriors overlap a lot. The posterior probability that **B converts better than A** is **{bandit_p_sup_B_v1:.1%}** — promising, but far from certain."
            ),
        ]
    )
    return bandit_p_sup_B_v1, plot_bandit_posteriors


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here is the Thompson-style allocation rule, applied at the level of a *batch* rather than a single email: split the next 100 emails between A and B in proportion to each campaign's posterior probability of being the better one.
    """)
    return


@app.cell
def _(bandit_p_sup_B_v1):
    bandit_n_B_next = int(round(100 * bandit_p_sup_B_v1))
    bandit_n_A_next = 100 - bandit_n_B_next
    bandit_n_A_next, bandit_n_B_next
    return bandit_n_A_next, bandit_n_B_next


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we send the next batch of 100 emails using that split, accumulating the new conversions into each campaign's running totals.
    """)
    return


@app.cell
def _(
    bandit_n_A_next,
    bandit_n_B_next,
    bandit_run_campaign,
    bandit_state_A1,
    bandit_state_B1,
):
    bandit_state_A2 = bandit_run_campaign(bandit_state_A1, bandit_n_A_next, seed=21)
    bandit_state_B2 = bandit_run_campaign(bandit_state_B1, bandit_n_B_next, seed=22)
    bandit_state_A2, bandit_state_B2
    return bandit_state_A2, bandit_state_B2


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And we re-fit the model on the accumulated data — this automates what you could do by manually re-running the cells above with updated totals: refit, recompute, and compare in one step.
    """)
    return


@app.cell
def _(bandit_state_A2, bandit_state_B2):
    with pm.Model() as bandit_model_v2:
        cr_A_v2 = pm.Beta("conversion_rate_A", alpha=2, beta=5)
        cr_B_v2 = pm.Beta("conversion_rate_B", alpha=2, beta=5)
        pm.Binomial(
            "obs_A", p=cr_A_v2, n=bandit_state_A2["n"], observed=bandit_state_A2["k"]
        )
        pm.Binomial(
            "obs_B", p=cr_B_v2, n=bandit_state_B2["n"], observed=bandit_state_B2["k"]
        )
        bandit_idata_v2 = pm.sample(random_seed=RANDOM_SEED)
    return (bandit_idata_v2,)


@app.cell(hide_code=True)
def _(bandit_idata_v2, plot_bandit_posteriors):
    bandit_post_fig_v2, bandit_p_sup_B_v2 = plot_bandit_posteriors(
        bandit_idata_v2, "Posteriors after second batch (200 emails total)"
    )
    mo.vstack(
        [
            bandit_post_fig_v2,
            mo.md(
                f"After the second batch, the green curve has tightened around its true value and the posterior probability that **B converts better than A** has moved to **{bandit_p_sup_B_v2:.1%}**. If we kept looping — fit, re-allocate, send another batch, re-fit — the algorithm would send a larger and larger share of each new batch through B until it was effectively certain."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### From batches to a continuous algorithm

    Doing this loop by hand, one batch at a time, gets old fast. Two simplifications turn it into a clean online algorithm:

    1. **Don't wait for a batch.** Run Thompson sampling at the level of a *single* email — for each new send, draw one sample from each campaign's current posterior and pick the larger one.
    2. **Don't re-run PyMC every step.** Because the Beta prior is conjugate to the Binomial likelihood, the posterior after observing $k$ conversions in $n$ emails is just $\text{Beta}(\alpha_0 + k,\ \beta_0 + n - k)$. Updating reduces to incrementing two counters.

    The simulation below does both. You control the true rates, the size of the mailing list, and the seed; everything updates reactively, and a scrubber lets you replay the run email by email.
    """)
    return


@app.cell(hide_code=True)
def _():
    bandit_rate_a = mo.ui.slider(
        0.01,
        0.5,
        step=0.01,
        value=0.10,
        label="True conversion rate — Campaign A",
        show_value=True,
    )
    bandit_rate_b = mo.ui.slider(
        0.01,
        0.5,
        step=0.01,
        value=0.15,
        label="True conversion rate — Campaign B",
        show_value=True,
    )
    bandit_n_rounds_ui = mo.ui.slider(
        50, 500, step=10, value=200, label="Emails to send", show_value=True
    )
    bandit_seed_ui = mo.ui.number(start=0, stop=9999, step=1, value=42, label="Seed")
    mo.vstack(
        [
            mo.md(
                "**Try changing the true conversion rates, the size of the mailing list, or the seed — every plot below updates automatically.**"
            ),
            mo.hstack(
                [bandit_rate_a, bandit_rate_b, bandit_n_rounds_ui, bandit_seed_ui],
                justify="start",
                gap=2,
            ),
        ]
    )
    return bandit_n_rounds_ui, bandit_rate_a, bandit_rate_b, bandit_seed_ui


@app.cell(hide_code=True)
def _(bandit_n_rounds_ui, bandit_rate_a, bandit_rate_b, bandit_seed_ui):
    def simulate_thompson(true_rates, n_rounds, seed):
        rng = np.random.default_rng(seed)
        n_variants = len(true_rates)
        alphas = np.ones(n_variants)
        betas = np.ones(n_variants)
        # Snapshots of (alpha, beta) after each round; row 0 is the prior.
        alpha_hist = np.zeros((n_rounds + 1, n_variants))
        beta_hist = np.zeros((n_rounds + 1, n_variants))
        alpha_hist[0] = alphas
        beta_hist[0] = betas
        chosen = np.zeros(n_rounds, dtype=int)
        outcomes = np.zeros(n_rounds, dtype=int)
        for t in range(n_rounds):
            samples = rng.beta(alphas, betas)
            pick = int(np.argmax(samples))
            success = int(rng.random() < true_rates[pick])
            alphas[pick] += success
            betas[pick] += 1 - success
            alpha_hist[t + 1] = alphas
            beta_hist[t + 1] = betas
            chosen[t] = pick
            outcomes[t] = success
        return alpha_hist, beta_hist, chosen, outcomes

    bandit_true_rates = (bandit_rate_a.value, bandit_rate_b.value)
    bandit_n = bandit_n_rounds_ui.value
    (
        bandit_alpha_hist,
        bandit_beta_hist,
        bandit_chosen,
        bandit_outcomes,
    ) = simulate_thompson(bandit_true_rates, bandit_n, bandit_seed_ui.value)

    # Posterior P(B > A) at every round, via vectorised Monte Carlo from each
    # snapshot of the Beta posteriors. Cheap enough for n <= 500.
    _rng_q = np.random.default_rng(0)
    _n_samp = 2000
    _a_a = bandit_alpha_hist[1:, 0, None]
    _b_a = bandit_beta_hist[1:, 0, None]
    _a_b = bandit_alpha_hist[1:, 1, None]
    _b_b = bandit_beta_hist[1:, 1, None]
    _samp_a = _rng_q.beta(_a_a, _b_a, size=(bandit_n, _n_samp))
    _samp_b = _rng_q.beta(_a_b, _b_b, size=(bandit_n, _n_samp))
    bandit_p_b_best_hist = (_samp_b > _samp_a).mean(axis=1)
    return (
        bandit_alpha_hist,
        bandit_beta_hist,
        bandit_chosen,
        bandit_n,
        bandit_outcomes,
        bandit_p_b_best_hist,
        bandit_true_rates,
    )


@app.cell(hide_code=True)
def _(bandit_n):
    bandit_step = mo.ui.slider(
        1,
        bandit_n,
        value=bandit_n,
        step=1,
        label="Scrub through emails sent",
        show_value=True,
        full_width=True,
    )
    bandit_step
    return (bandit_step,)


@app.cell(hide_code=True)
def _(
    bandit_alpha_hist,
    bandit_beta_hist,
    bandit_chosen,
    bandit_n,
    bandit_outcomes,
    bandit_p_b_best_hist,
    bandit_step,
    bandit_true_rates,
):
    _step = bandit_step.value
    _a_a, _a_b = bandit_alpha_hist[_step]
    _b_a, _b_b = bandit_beta_hist[_step]

    # --- Beta posteriors at the current round ---
    _theta = np.linspace(0.0, 0.5, 400)
    _pdf_a = stats.beta.pdf(_theta, _a_a, _b_a)
    _pdf_b = stats.beta.pdf(_theta, _a_b, _b_b)
    post_fig = go.Figure()
    post_fig.add_trace(
        go.Scatter(
            x=_theta,
            y=_pdf_a,
            name=f"A: Beta({_a_a:.0f}, {_b_a:.0f})",
            line=dict(color=PYMC_BLUE, width=2),
            fill="tozeroy",
            fillcolor="rgba(21, 74, 114, 0.18)",
        )
    )
    post_fig.add_trace(
        go.Scatter(
            x=_theta,
            y=_pdf_b,
            name=f"B: Beta({_a_b:.0f}, {_b_b:.0f})",
            line=dict(color=PYMC_GREEN, width=2),
            fill="tozeroy",
            fillcolor="rgba(129, 194, 64, 0.18)",
        )
    )
    post_fig.add_vline(
        x=bandit_true_rates[0],
        line_dash="dot",
        line_color=PYMC_BLUE,
        annotation_text="true A",
        annotation_position="top",
    )
    post_fig.add_vline(
        x=bandit_true_rates[1],
        line_dash="dot",
        line_color=PYMC_GREEN,
        annotation_text="true B",
        annotation_position="top",
    )
    post_fig.update_layout(
        title=f"Posterior beliefs after {_step} emails sent",
        xaxis_title="conversion rate θ",
        yaxis_title="density",
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )

    # --- Allocation + probability of superiority over time ---
    _rounds = np.arange(1, bandit_n + 1)
    _frac_b = np.cumsum(bandit_chosen == 1) / _rounds
    alloc_fig = go.Figure()
    alloc_fig.add_trace(
        go.Scatter(
            x=_rounds,
            y=_frac_b,
            name="fraction of emails sent via B",
            line=dict(color=PYMC_GREEN, width=2),
        )
    )
    alloc_fig.add_trace(
        go.Scatter(
            x=_rounds,
            y=bandit_p_b_best_hist,
            name="P(B converts better than A)",
            line=dict(color=PYMC_BLUE, width=2, dash="dash"),
        )
    )
    alloc_fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    alloc_fig.add_vline(x=_step, line_dash="dash", line_color="#888")
    alloc_fig.update_layout(
        title="Allocation tracks belief",
        xaxis_title="emails sent",
        yaxis_title="probability",
        yaxis_range=[0, 1.05],
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(yanchor="top", y=0.4, xanchor="right", x=0.98),
    )

    # --- Stats panel ---
    _picks = bandit_chosen[:_step]
    _outs = bandit_outcomes[:_step]
    _n_a = int((_picks == 0).sum())
    _n_b = int((_picks == 1).sum())
    _k_a = int(_outs[_picks == 0].sum())
    _k_b = int(_outs[_picks == 1].sum())
    _obs_a = f"{_k_a / _n_a:.1%}" if _n_a > 0 else "—"
    _obs_b = f"{_k_b / _n_b:.1%}" if _n_b > 0 else "—"
    _p_now = bandit_p_b_best_hist[_step - 1]
    # Expected regret = conversions lost vs. always sending the better campaign
    _best_rate = max(bandit_true_rates)
    _expected_optimal = _best_rate * _step
    _expected_actual = bandit_true_rates[0] * _n_a + bandit_true_rates[1] * _n_b
    _regret = _expected_optimal - _expected_actual

    summary = mo.callout(
        mo.md(f"""
    **After {_step} of {bandit_n} emails sent**

    | campaign | true rate | emails sent | conversions | observed rate |
    |---|---:|---:|---:|---:|
    | **A** | {bandit_true_rates[0]:.0%} | {_n_a} | {_k_a} | {_obs_a} |
    | **B** | {bandit_true_rates[1]:.0%} | {_n_b} | {_k_b} | {_obs_b} |

    Posterior **P(B converts better than A) ≈ {_p_now:.1%}** &nbsp;&nbsp;·&nbsp;&nbsp; expected regret ≈ **{_regret:.1f} conversions** lost vs. always using the better campaign.
        """),
        kind="info",
    )

    mo.vstack(
        [
            mo.hstack([post_fig, alloc_fig], widths="equal", gap=1),
            summary,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Things to try:**

    - Drag the **scrubber** from the first email to the last. Watch the Beta posteriors start as flat priors (Beta(1, 1)) and sharpen around the true conversion rates. Notice how the green curve (Campaign B) almost always pulls away faster — because the algorithm sends more of the mailing list through B once it suspects B is the better campaign.
    - Set both true conversion rates **equal** (e.g. 0.10 and 0.10). The green allocation line wanders around 0.5 indefinitely and P(B better than A) hovers near 50%. Thompson sampling correctly *refuses to commit* when neither campaign is actually better.
    - Make the gap **tiny** (0.10 vs 0.11). It now takes hundreds of emails — and many seeds will favour A for a long stretch before flipping. Small effects need large mailings, even for adaptive methods.
    - Make the gap **large** (0.05 vs 0.30). The algorithm locks in on B within ~30 emails.
    - Change the **seed** to see how much early luck matters before the data overwhelms the prior.

    This strategy — also called the **Bayesian bandit** — is optimal in the sense that it maximises expected conversions given current beliefs. It resolves the exploration/exploitation tradeoff automatically, with no tuning parameter, and converges to always sending the better campaign in the long run.

    | | Traditional A/B Test | Bayesian Bandits |
    |---|---|---|
    | **Sample size** | Fixed in advance | Adaptive |
    | **Assignment** | 50/50 split of the mailing list | Shifts toward the better campaign |
    | **Regret** | High (many emails sent via the worse campaign) | Low (few wasted on the worse campaign) |
    | **When to use** | Strict statistical protocols | Optimising an ongoing email programme |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    <div style="text-align: center; color: #888; font-size: 0.85rem; padding-top: 1rem;">
    Introduction to PyMC and Bayesian Modeling
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
