import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pymc as pm
    import arviz as az
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import preliz as pz
    import matplotlib.pyplot as plt
    from plotly.subplots import make_subplots
    import warnings
    from pathlib import Path
    import base64
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

    az.style.use("arviz-variat")

    data_path = Path(__file__).parent / "data"

    RANDOM_SEED = 42

    warnings.filterwarnings("ignore", module="mkl_fft")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    return (
        PYMC_BLUE,
        PYMC_DARK_GREEN,
        PYMC_GREEN,
        PYMC_LIGHT_BLUE,
        Path,
        RANDOM_SEED,
        az,
        base64,
        data_path,
        np,
        pl,
        plt,
        pm,
        px,
    )


@app.cell(hide_code=True)
def header(Path, base64, mo):
    logo_path = Path(__file__).parent / "images" / "pymc-labs-logo.png"
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="300" style="margin-bottom: 0.5rem;">'
    else:
        logo_html = ""

    mo.md(f"""
    {logo_html}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generalized Linear Models

    > ⚠ DRAFT — CONTENT TO BE AUTHORED (scaffold only).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Why Linear Regression Isn't Enough

    Our fish weight model worked well because (after log-transformation) the response was approximately continuous and symmetric. But many real-world outcomes don't fit this mold:

    - **Counts** (number of events, page views, defects) — non-negative integers
    - **Proportions** (approval rates, conversion rates) — bounded between 0 and 1
    - **Skewed continuous data** (insurance claims, reaction times) — non-negative with long tails

    A Normal likelihood with an identity link can predict negative values, fractional counts, or probabilities outside [0, 1]. **Generalized Linear Models (GLMs)** solve this by combining three components:

    1. **A distribution family** appropriate for the data type (Poisson for counts, Binomial for proportions, etc.)
    2. **A link function** that maps the linear predictor to the distribution's natural parameter space
    3. **A linear predictor** $\eta = \alpha + \beta_1 x_1 + \cdots + \beta_p x_p$ — same as before

    $$g(\mu) = \eta = X\beta$$

    where $g$ is the link function and $\mu = E[y]$.

    | Data type | Distribution | Link function | $g(\mu)$ |
    |-----------|-------------|---------------|----------|
    | Continuous, symmetric | Normal | Identity | $\mu$ |
    | Counts | Poisson | Log | $\log(\mu)$ |
    | Overdispersed counts | Negative Binomial | Log | $\log(\mu)$ |
    | Binary / proportions | Binomial | Logit | $\log\frac{\mu}{1-\mu}$ |
    | Proportions with overdispersion | Beta-Binomial | Logit | $\log\frac{\mu}{1-\mu}$ |

    We will explore these types of models in the next session!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Poisson regression for count data

    > ⚠ TO BE AUTHORED.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handling overdispersion with Negative Binomial regression

    > ⚠ TO BE AUTHORED.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logistic regression for binary outcomes

    > ⚠ TO BE AUTHORED.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reuse pointers

    - Code patterns: `skill://pymc-modeling` § Common Patterns → GLMs (`pm.math.sigmoid`, `pm.math.exp`); `references/specialized_likelihoods.md` for ZIP/NegBin.
    - Datasets already in `notebooks/data/`: `poisson_sneeze.csv`, `day.csv` (bike counts), `salmon.csv`, `fish-market.csv`, `penguins.csv` (binary species), `macron_popularity.csv`.
    - Snippets to adapt from London: overdispersion demo (`Session_1.2:1020-1096`), coal-mining Poisson change-point (`Session_2:2509-2581`), golf logistic/Binomial (`Session_3.2:148-228`), latent-GP Poisson + NegativeBinomial (`Session_6B:183-193,294-309,418-431`).
    """)
    return


if __name__ == "__main__":
    app.run()
