import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import base64
    import inspect
    from pathlib import Path
    import numpy as np
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    import pytensor.xtensor as ptx
    import arviz_plots as azp
    import io

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

    data_path = Path(__file__).parent / "data"


@app.cell(hide_code=True)
def header():
    def render_pytensor_logo():
        img_path = Path(__file__).parent / "images" / "PyTensor_RGB.png"
        img_html = ""
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            img_html = f'<img src="data:image/png;base64,{b64}" width="400">'
        return img_html

    pytensor_html = render_pytensor_logo()

    mo.md(f"""
    # Session 2.1: PyTensor and the PyMC API

    In this session we look under PyMC's hood. PyMC is built on **PyTensor**, a library for defining and compiling computational graphs. We'll learn just enough PyTensor to read the graphs PyMC builds, then use that lens to understand what a PyMC model actually *is*, and tour the parts of the PyMC API you'll use in every model: distributions, shapes and `dims`, `logp` and `draw`, gradients, `pm.math`, and the model-debugging toolkit.

    ## PyTensor Basics

    {pytensor_html}

    PyTensor is the computational backend of PyMC: every model you build with PyMC is, underneath, a PyTensor **computational graph**. You will rarely write PyTensor directly; the goal of this section is to learn just enough to *read* what PyMC builds for you. That literacy pays off throughout the course: it explains what model variables really are, what "compiling" a model means, and how to decode PyTensor's error messages when something goes wrong.

    In PyTensor, you define a computational graph explicitly. You start with input variables that are essentially placeholders, and from these build intermediate variables by applying operators. While PyTensor is designed to feel similar to NumPy, there is a key difference: PyTensor operations build a graph of computations to be executed **lazily**, rather than immediately returning values.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(r"""
    ### Tensors and Basic Operations

    To begin, let's define some PyTensor tensors and show how to perform some basic operations.
    """),
            mo.callout(
                mo.md(r"""
    **What is a Tensor?**

    A tensor is a multi-dimensional array that serves as the fundamental data structure.

    Think of it as a generalization of more familiar concepts:

    * A 0-D tensor is a single number (a scalar).
    * A 1-D tensor is a list of numbers (a vector).
    * A 2-D tensor is a grid of numbers (a matrix).
    * A 3-D tensor is a cube of numbers, and so on for any number of dimensions.
    """),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A tensor can be a **scalar** or a **vector** with any number of dimensions.

    Concretely:
    """)
    return


@app.cell(hide_code=True)
def _():
    x = pt.tensor(shape=(), dtype="float64")
    y = pt.tensor(shape=(2,), dtype="float64")

    mo.md(f"""
    `x` type: `{x.type}`, shape = `{x.type.shape}`
    `y` type: `{y.type}`, shape = `{y.type.shape}`
    """)
    return x, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we have defined the `x` and `y` tensors, we can create a new one by adding them together.
    """)
    return


@app.cell
def _(x, y):
    z = x + y
    z.name = "x + y"
    z
    return (z,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To make the computation a bit more complex let's take the logarithm of the resulting tensor.
    """)
    return


@app.cell
def _(z):
    w = pt.log(z)
    w, type(w)
    return (w,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We did not give `w` a name, so it prints something more descriptive: its the index-0 output of calling the `log` function. Its type is `TensorVariable`, which is the base class for all PyTensor variables.

    So PyTensor works something like NumPy, but it builds a **graph of operations** rather than executing, more like a symbolic computation library.

    We can use the `pytensor.dprint` function to print the computational graph of any given tensor.
    """)
    return


@app.cell(hide_code=True)
def _(w):
    w.dprint()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This output shows the structure of the computation that PyTensor has built for the variable `w`.

    Think of it as a "recipe" (in reverse):

    - **`Log [id A] 'log(x + y)'`**: This is the final result, `w`. It's calculated by taking the logarithm (`Log`) of an intermediate value named `'log(x + y)'`. PyTensor assigns it an internal identifier `A`.
    - **`Add [id B] 'x + y'`**: This is the input to the `Log` operation. It's an intermediate value named `'x + y'` (which we called `z` in the code), calculated by an addition (`Add`). Its internal ID is `B`.
    - **`ExpandDims{axis=0} [id C]`**: This is the first input to the `Add` operation. `ExpandDims` is an operation that changes the shape of a tensor. Here, it's likely making the scalar `x` compatible for addition with the vector `y`. Its ID is `C`.
      - **`<Scalar(float64, shape=())> [id D]`**: This is the input to `ExpandDims`. It's our original scalar tensor `x` (ID `D`), which holds a single 64-bit floating-point number.
    - **`<Vector(float64, shape=(2,))> [id E]`**: This is the second input to the `Add` operation. It's our original vector tensor `y` (ID `E`), which holds two 64-bit floating-point numbers.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Functions

    Note that this graph does not do any computation (yet!). It is simply defining the sequence of steps to be done. We can use `pytensor.function` to define a callable object so that we can push values through the graph.

    PyTensor functions are **compiled** from symbolic expressions into efficient callable functions. The `pytensor.function` constructor takes several key arguments that define how the function will behave:

    - The `inputs` argument specifies which PyTensor variables will be provided when calling the function. These become the function's parameters.

    - The `outputs` argument defines which symbolic expressions should be evaluated and returned when the function is called.
    """)
    return


@app.cell
def _(w, x, y):
    f = pytensor.function(inputs=[x, y], outputs=w)
    return (f,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that the graph is compiled, we can push some concrete values:
    """)
    return


@app.cell
def _(f):
    f(0, [1, np.e])
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(r"""
    **TIP:** Sometimes we just want to debug, we can use `pytensor.graph.basic.Variable.eval` for that:

    `eval` compiles behind the scenes the first time you call it, so it is perfect for spot checks but wasteful in a loop. When you need repeated evaluation, compile once with `pytensor.function` and reuse the result.
    """),
        kind="info",
    )
    return


@app.cell
def _(w, x, y):
    w.eval({x: 0, y: [1, np.e]})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can set intermediate values as well
    """)
    return


@app.cell
def _(w, z):
    w.eval({z: [1, np.e]})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Graph Optimization

    One of the most important features of `pytensor` is that it can automatically **optimize** the mathematical operations inside a graph. Let's consider a simple example:
    """)
    return


@app.cell
def _():
    a = pt.tensor(shape=(), name="a")
    b = pt.tensor(shape=(), name="b")

    c = a / b
    c.name = "a / b"

    c.dprint()
    return a, b, c


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now let's multiply `b` times `c`. This should result in simply `a`.
    """)
    return


@app.cell
def _(b, c):
    d = b * c
    d.name = "b * c"

    d.dprint()
    return (d,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The graph shows the full computation, including the superfluous operations, but once we compile it the operation becomes the identity on `a` as expected.
    """)
    return


@app.cell
def _(a, b, d):
    g = pytensor.function(inputs=[a, b], outputs=d)
    g.dprint()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    These automatic simplifications are called **graph rewrites**. Because your code builds a graph rather than executing immediately, PyTensor is free to replace pieces of that graph with cheaper or safer equivalents before anything runs. Beyond removing redundant arithmetic, PyTensor can recognize and replace many common expressions with more numerically stable equivalents, for example, `log(1 + x)` is rewritten to use `log1p`, which stays accurate when `x` is very close to zero. (This is not a blanket guarantee: custom expressions can still be numerically unstable.)

    When PyMC compiles your model's log-probability, all of these rewrites are applied automatically; you get the optimized, stabilized graph without asking for it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can watch one of these stability rewrites happen. Build the naive expression `log(1 + p)`; the raw graph faithfully records an `Add` followed by a `Log`:
    """)
    return


@app.cell
def _():
    p = pt.tensor(shape=(), name="p")
    naive_log = pt.log(1 + p)
    naive_log.dprint()
    return naive_log, p


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compile it, and the rewrite has fired: the addition and logarithm are gone, replaced by a single `Log1p` node.
    """)
    return


@app.cell
def _(naive_log, p):
    log1p_fn = pytensor.function(inputs=[p], outputs=naive_log)
    log1p_fn.dprint()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Automatic Differentiation

    Here is the real payoff of building graphs instead of computing eagerly: PyTensor can **differentiate** them. `pt.grad` takes the graph of a scalar expression and returns a *new graph* that computes its derivative, exactly (no finite differences), by applying the chain rule node by node.

    This one feature is what makes modern Bayesian inference practical. The NUTS sampler you will meet in Session 3.1 needs the gradient of the model's joint log-probability at every step of every trajectory; PyMC derives those gradients automatically from the graph your model builds.
    """)
    return


@app.cell
def _():
    s = pt.tensor(shape=(), name="s")
    loss = s**2 + pt.sin(s)
    dloss = pt.grad(loss, wrt=s)

    dloss_fn = pytensor.function(inputs=[s], outputs=dloss)
    dloss_fn.dprint()
    return (dloss_fn,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The compiled derivative graph is exactly what you would have written by hand, $2s + \cos(s)$, fused into a single `Composite` node. It evaluates like any other compiled function:
    """)
    return


@app.cell
def _(dloss_fn):
    dloss_fn(0.0), dloss_fn(np.pi)
    return


@app.cell(hide_code=True)
def _():
    def render_apply_diagram():
        img_path = Path(__file__).parent / "images" / "apply.png"
        img_html = ""
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            img_html = f'<img src="data:image/png;base64,{b64}" width="400">'
        return img_html

    apply_html = render_apply_diagram()

    mo.md(f"""
    ### What is in a PyTensor Graph?

    The following diagram shows the basic structure of a PyTensor graph.

    {apply_html}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can make these concepts more tangible by explicitly accessing them from our earlier example. Let's compute the graph components for the tensor `z`.
    """)
    return


@app.cell(hide_code=True)
def _(z):
    mo.md(f"""
    - `z.type` = `{z.type}`
    - `z.name` = `{z.name}`
    - `z.owner` = `{z.owner}`
    - `z.owner.inputs` = `{z.owner.inputs}`
    - `z.owner.op` = `{z.owner.op}`
    - `z.owner.outputs` = `{z.owner.outputs}`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## From PyTensor to PyMC: A Model Is a Graph

    Defining variables, building expressions, compiling functions: everything we just did by hand is exactly what PyMC automates. Bayesian inference begins with a probability model that relates unknown parameters to observed data; PyMC provides high-level building blocks for constructing these models, and **every one of those building blocks is a PyTensor object underneath**. In this section we'll build the smallest possible model and inspect it with the graph-reading tools from the previous section.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model Contexts and Random Variables

    The canonical way to specify a PyMC model is with the `Model` **context manager**, the `with pm.Model() as model:` idiom you have already seen. Think of `Model` as a tape machine: while the context is open, it records every random variable and other component you create, along with the bookkeeping needed to move values between your Python code and the underlying graph.

    Most importantly for our purposes, a `Model` can compile PyTensor functions from the variables registered with it, which is exactly the machinery we are about to inspect.
    """)
    return


@app.cell
def _():
    with pm.Model() as model:
        z_1 = pm.Normal("z", mu=0.0, sigma=5.0)
    model
    return model, z_1


@app.cell
def _(model):
    model.named_vars
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    What did `pm.Normal("z", ...)` actually create? Not a special PyMC object: a plain PyTensor `TensorVariable`, whose `owner.op` is a **`RandomVariable`** operation. This is the same graph anatomy we inspected earlier:
    """)
    return


@app.cell
def _(z_1):
    type(z_1), z_1.owner.op
    return


@app.cell
def _(z_1):
    z_1.dprint()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Asking for the log-probability of a value builds a **new** graph (the density expression) from the random variable's graph:
    """)
    return


@app.cell
def _(z_1):
    z_logp = pm.logp(z_1, 2.5)
    z_logp.dprint()
    return (z_logp,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A graph is not a number; to evaluate it we compile, exactly as before. `model.compile_logp()` applies the same compile-then-call pattern to the model's **joint** log-probability graph (the sum over all of the model's variables). Our model has a single variable, so the joint log-probability matches the single-variable graph we just printed:
    """)
    return


@app.cell
def _(model, z_logp):
    z_logp.eval(), model.compile_logp()({"z": 2.5})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And because the joint log-probability is itself a graph, `pt.grad` applies to it too. `model.compile_dlogp()` compiles the gradient of the joint log-density with respect to every free variable. This is precisely the function that gradient-based samplers like NUTS call over and over during sampling; for our $\text{Normal}(0, 5)$ the analytic answer is $-z/\sigma^2 = -2.5/25$:
    """)
    return


@app.cell
def _(model):
    model.compile_dlogp()({"z": 2.5})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The Model's Bookkeeping

    The `Model` context records more than names. Three attributes expose the machinery that inference algorithms rely on:

    - `model.free_RVs`: the unobserved random variables (what a sampler must explore)
    - `model.value_vars`: the **value variables**, the inputs that compiled `logp`/`dlogp` functions actually accept
    - `model.rvs_to_values`: the mapping between the two

    For an unbounded variable the value variable is a plain stand-in. But watch what happens to a bounded one:
    """)
    return


@app.cell
def _():
    with pm.Model() as bookkeeping_model:
        bk_mu = pm.Normal("mu", 0, 1)
        bk_sigma = pm.HalfNormal("sigma", 1)

    bookkeeping_model.free_RVs, bookkeeping_model.value_vars
    return (bookkeeping_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `HalfNormal` variable `sigma` gets a value variable named `sigma_log__`: PyMC samples bounded parameters on an unconstrained (here, logarithmic) scale and transforms back automatically, so the sampler never has to worry about boundaries. You will meet these transformations properly in Session 2.2, but this is why sampler output and error messages sometimes mention variables with `_log__` or `_interval__` suffixes that you never defined.
    """)
    return


@app.cell
def _(bookkeeping_model):
    bookkeeping_model.rvs_to_values
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Core Components

    - **Stochastic Random Variables**: Variables whose values are not completely determined by their parents. These represent uncertainty in the model parameters or data generating process.
        - **Prior distributions** for model parameters
        - **Likelihood distributions** for observed data

        $$x \sim \text{Normal}(\mu, \sigma)$$

    - **Deterministic Variables**: Variables whose values are completely determined by their parents through a mathematical operation; no additional randomness is added by the operation. These represent transformations or combinations of other variables.

        $$y = \mu + \beta x$$

    - **Factor Potentials**: Additional terms that modify the joint log-probability without being variables themselves. This is useful for implementing constraints or complex likelihood terms.

        $$| x - y | < 1$$

    You will build Deterministic variables and Potentials hands-on in Session 2.2; for now it is enough to know they exist.

    These building blocks connect in a directed acyclic graph (DAG) that completely specifies a joint probability distribution. PyMC leverages this graph structure to efficiently sample from the posterior distribution using its available inference algorithms.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Distributions and Random Variables

    Statistical distributions are provided in PyMC as subclasses of `Distribution`: `pm.Normal`, `pm.Binomial`, and so on. Calling one does **not** return a distribution object: as we saw above, it creates a PyTensor `TensorVariable` whose `owner.op` is a `RandomVariable` (such as `NormalRV`), and registers that variable with the enclosing `Model`. This registration is why the named form is only usable inside a model context.

    `Distribution` subclasses accept several arguments when constructed. Some of the most important are:

    `name`
    : Name for the new model variable. This argument is **required**, and is used as a label and index value for the variable.
    """)
    return


@app.cell
def _():
    with pm.Model():
        x_2 = pm.Normal(name="x")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `shape`
    : The variable's shape.
    """)
    return


@app.cell
def _():
    with pm.Model():
        x_matrix = pm.Normal("x_matrix", shape=(3, 3))

    pm.draw(x_matrix, random_seed=1)  # better to use a numpy generator as seed
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `dims`
    : A tuple of dimension names known to the model.
    """)
    return


@app.cell
def _():
    city_names = ["Vancouver", "Calgary", "Toronto", "Montreal", "Halifax"]
    with pm.Model(coords={"city": city_names}) as model_1:
        x_city = pm.Normal("x_city", dims="city")
    model_1
    return model_1, x_city


@app.cell
def _(model_1):
    with model_1:
        samples = pm.sample_prior_predictive(1000)
    azp.plot_forest(samples, group="prior")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### How Shapes Are Determined

    `shape` and `dims` are two of **three** ways a variable can get its shape, and the interplay between them is the most common source of beginner model bugs. Let's work through all three with a real dataset: the Palmer penguins, which record body mass for three species.
    """)
    return


@app.cell
def _():
    penguins = pl.read_csv(data_path / "penguins.csv", null_values="NA")
    species_mass = (
        penguins.drop_nulls(subset=["body_mass_g"])
        .group_by("species")
        .agg(pl.col("body_mass_g").mean())
        .sort("species")
    )
    species_mass
    return (species_mass,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **1. Implied shape.** Pass vector-valued parameters and the variable silently inherits their shape. No `shape` argument in sight, yet this is a length-3 random variable, one component per species:
    """)
    return


@app.cell
def _(species_mass):
    species_mu = species_mass["body_mass_g"].to_numpy()

    with pm.Model():
        mass_implied = pm.Normal("mass", mu=species_mu, sigma=100)

    mass_implied.shape.eval()
    return mass_implied, species_mu


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **2. Broadcasting.** Notice that `sigma=100` is a scalar: it was broadcast against the length-3 `mu` exactly as NumPy would broadcast arrays. The result is three independent normals, each with its own mean, sharing one spread. `pm.draw` shapes follow along, with the draws dimension prepended:
    """)
    return


@app.cell
def _(mass_implied):
    pm.draw(mass_implied, draws=1000, random_seed=RANDOM_SEED).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **3. Explicit `shape=` on top of broadcasting.** An explicit `shape` can *extend* the parameter shapes: ask for `(2, 3)` and the length-3 `mu` broadcasts across the new leading axis. Think of it as a sex × species grid of variables, every row sharing the same species means:
    """)
    return


@app.cell
def _(species_mu):
    with pm.Model():
        mass_grid = pm.Normal("mass_grid", mu=species_mu, sigma=100, shape=(2, 3))

    pm.draw(mass_grid, random_seed=RANDOM_SEED).round(0)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    But integer axes are anonymous: nothing in `mass_grid` records that row 0 means female, or that column 2 means Gentoo. That bookkeeping burden is exactly what `dims` and `coords` remove. Give the model labeled coordinates and declare the variable's dimensions by name; downstream, every trace, summary table, and plot inherits the labels:
    """)
    return


@app.cell
def _(species_mass, species_mu):
    with pm.Model(
        coords={
            "sex": ["female", "male"],
            "species": species_mass["species"].to_list(),
        }
    ) as penguin_model:
        pm.Normal("mass", mu=species_mu, sigma=100, dims=("sex", "species"))

    with penguin_model:
        penguin_prior = pm.sample_prior_predictive(500, random_seed=RANDOM_SEED)

    azp.plot_forest(penguin_prior, group="prior")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Every interval in the forest plot is labeled with its coordinates; no mental bookkeeping about which index is which. This is why the course uses `dims` for any variable with meaningful structure.

    Finally, learn to recognize what happens when shapes *cannot* reconcile. A length-3 `mu` cannot fill a length-4 variable, and the error says so in broadcasting terms:
    """)
    return


@app.cell
def _(species_mu):
    try:
        pm.draw(pm.Normal.dist(mu=species_mu, shape=(4,)))
    except ValueError as err:
        print(f"ValueError: {err}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Dims-Aware Math: `pytensor.xtensor`

    Dims do more than label axes in plots. PyTensor has a dims-aware tensor type, `pytensor.xtensor`, where operations align on dimension **names** rather than axis positions. Matrix multiplication is the flagship example: `dot` contracts whichever dimension the two operands share, so you never specify an axis. Build a coefficient matrix and a feature vector that share the `feature` dimension:
    """)
    return


@app.cell
def _():
    coef = ptx.xtensor("coef", dims=("species", "feature"), shape=(3, 2))
    feats = ptx.xtensor("feats", dims=("feature",), shape=(2,))

    species_score = ptx.dot(feats, coef)
    species_score.type
    return coef, feats, species_score


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    No axis argument anywhere: the shared `feature` dimension was contracted *because the names match*, leaving one value per `species`. And since names, not positions, drive the contraction, transposing an operand and reversing the argument order changes nothing:
    """)
    return


@app.cell
def _(coef, feats, species_score):
    species_score_reversed = ptx.dot(feats, coef.transpose("feature", "species"))

    coef_vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    feat_vals = np.array([10.0, 1.0])
    (
        species_score.eval({coef: coef_vals, feats: feat_vals}),
        species_score_reversed.eval({coef: coef_vals, feats: feat_vals}),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With two matrices the same rule applies: the shared dimension is contracted and every un-shared dimension survives, labels intact. Compare this with `np.dot`, where you would be juggling transposes and remembering what axis 0 means:
    """)
    return


@app.cell
def _(coef):
    island_effects = ptx.xtensor("island_effects", dims=("feature", "island"), shape=(2, 4))

    ptx.dot(coef, island_effects).type
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This is the machinery that model `dims` ride on. When a later session multiplies an observations × features design matrix by a feature-length coefficient vector, dimension names keep that bookkeeping straight for you.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `initval`
    : Numeric or symbolic untransformed initial value of matching shape, or one of the following initial value strategies: "moment", "prior". Depending on the sampler's settings, a random jitter may be added to numeric, symbolic or moment-based initial values in the transformed space.
    """)
    return


@app.cell
def _():
    with pm.Model() as model_2:
        x_3 = pm.Normal("x", initval=-2)
    return model_2, x_3


@app.cell
def _(model_2):
    model_2.rvs_to_initial_values
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PyMC includes most of the **probability density functions** (for continuous variables) and **probability mass functions** (for discrete variables) used in statistical modeling. These distributions are divided into five distinct categories:

    * Univariate continuous
    * Univariate discrete
    * Multivariate
    * Mixture
    * Timeseries

    Probability distributions are all subclasses of `Distribution`, which in turn has two major subclasses: `Discrete` and `Continuous`. In terms of data types, a `Continuous` random variable is given whichever floating point type is defined by `pytensor.config.floatX`, while `Discrete` variables are given the `int64` type.
    """)
    return


@app.cell
def _(x_3):
    x_3.dtype
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Multivariate and Timeseries random variables are vector-valued, rather than scalar (though `Continuous` and `Discrete` variables may have non-scalar values).
    """)
    return


@app.cell
def _(x_3):
    x_3.shape.eval()
    return


@app.cell
def _(x_city):
    x_city.shape.eval()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Every distribution knows how to do two things: **generate random draws** and **compute the log-probability of a value**. Random generation is implemented by the underlying `RandomVariable` operation (the `NormalRV` we saw in the graph) and log-probabilities are provided through PyMC's `logp` dispatch system.

    You don't call either mechanism directly. The user-facing functions are `pm.logp()` for log-probability and `pm.draw()` for simulation, and PyMC's inference algorithms use the same machinery internally when fitting models.
    """)
    return


@app.cell
def _(x_3):
    pm.logp(x_3, value=0).eval()
    return


@app.cell
def _(x_city):
    pm.logp(x_city, value=np.random.randn(5)).eval()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ...and `pm.draw()` simulates values from a variable, whether or not it belongs to a model:
    """)
    return


@app.cell
def _(x_3):
    pm.draw(x_3, draws=10)
    return


@app.cell
def _(x_city):
    pm.draw(x_city, draws=5)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Many distributions also expose their **cumulative distribution function** and its inverse, through `pm.logcdf` and `pm.icdf`. Both follow the same pattern as `pm.logp`: they build a graph that you then evaluate. The inverse CDF of a standard normal at 0.975 returns a familiar number:
    """)
    return


@app.cell
def _():
    std_normal = pm.Normal.dist(0, 1)
    np.exp(pm.logcdf(std_normal, 0).eval()), pm.icdf(std_normal, 0.975).eval()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Sometimes we wish to use a distribution without adding a variable to a model, for example, just to generate random numbers. For this purpose, `Distribution` classes have a `.dist()` class method that returns an **unregistered** random-variable tensor: the same kind of `TensorVariable` as before, but not attached to any model, so it can be created and used anywhere.
    """)
    return


@app.cell
def _():
    def plot_exponential_samples():
        x_dist = pm.Exponential.dist(1)
        samples = pm.draw(x_dist, draws=1000)
        return px.histogram(
            samples, title="Exponential Distribution Samples"
        ).update_layout(xaxis_title="Value", yaxis_title="Count", showlegend=False)

    plot_exponential_samples()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The `pm.math` Namespace

    When you need mathematical operations on model variables (link functions, conditional logic, matrix products) reach for the **`pm.math`** namespace. These are the same symbolic PyTensor operations from the start of this session, collected under a PyMC-friendly name. Commonly used members include `pm.math.log`, `exp`, `sqrt`, `dot`, `sum`, `switch` (element-wise if/else), and the link functions `invlogit`/`logit`.

    Because they are symbolic, they add nodes to the graph rather than computing immediately, so they can be applied to model variables just like `+` or `*`. Later sessions use `pm.math.switch` and `pm.math.invlogit` without further introduction; this is where they come from.
    """)
    return


@app.cell
def _():
    grid = np.linspace(-3, 3, 7)
    pm.math.invlogit(grid).eval(), pm.math.switch(grid > 0, 1.0, 0.0).eval()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Debugging Models

    A model that *builds* without error can still be broken *numerically*. The most common failure: the model's log-probability is `-inf` at the sampler's starting point, so sampling fails immediately. PyMC ships a small toolkit for catching this before you sample:

    - `model.initial_point()`: the starting values for every variable
    - `model.point_logps()`: each variable's log-probability at that point
    - `model.debug()`: an automated diagnosis that pinpoints the offending variable and value

    Here is a model with a planted bug: a Poisson count variable whose initial value is negative, outside the distribution's support. It constructs without complaint:
    """)
    return


@app.cell
def _():
    with pm.Model() as broken_model:
        count = pm.Poisson("count", mu=2.0, initval=-1)

    broken_model.initial_point(), broken_model.point_logps()
    return (broken_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `point_logps()` shows *which* variable has a non-finite log-probability. `model.debug()` goes further, evaluating each variable's parameters and reporting exactly which value is to blame:
    """)
    return


@app.cell
def _(broken_model):
    broken_model.debug()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: Fix the Broken Model

    The model below builds without error, but sampling would fail immediately: its initial log-probability is `-inf`.

    1. Use `model.point_logps()` (and `model.debug()` if you like) to find which variable is broken, and why.
    2. Fix the model definition so that every value in `point_logps()` is finite.

    The function should still define a variable named `"y"` and return `model.point_logps()`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Hint": mo.md(
                "A `Binomial` with `n=5` can only take the values 0 through 5. "
                "Where does the model's starting value for `y` come from? "
                "Check `model.initial_point()`."
            ),
        }
    )
    return


@app.function
def exercise_debug_model():
    with pm.Model() as model:
        y = pm.Binomial("y", n=5, p=0.5, initval=6)

    # YOUR CODE HERE — inspect model.point_logps() (and model.debug())
    # to find the problem, then fix the model definition above so the
    # log-probability is finite.
    return model.point_logps()


@app.cell(hide_code=True)
def _():
    run_debug = mo.ui.run_button(label="▶ Run exercise")
    run_debug
    return (run_debug,)


@app.cell(hide_code=True)
def _(run_debug):
    mo.stop(
        not run_debug.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    _logps = exercise_debug_model()
    _passed = set(_logps) == {"y"} and all(np.isfinite(_v) for _v in _logps.values())
    mo.md(
        f"`point_logps()` = `{_logps}`: "
        + (
            "**passed** ✅"
            if _passed
            else "**not yet** ❌: the model must keep a variable named `y`, "
            "and every log-probability must be finite."
        )
    )
    return


@app.cell(hide_code=True)
def _():
    def solution_debug_model():
        with pm.Model() as model:
            y = pm.Binomial("y", n=5, p=0.5, initval=2)
        return model.point_logps()

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(
                        "The bug is the `initval`: 6 is outside the support of "
                        "`Binomial(n=5, p=0.5)`, which only admits values 0–5, so the "
                        "starting log-probability is `-inf`. The fix is a valid initial "
                        "value (or simply deleting `initval`, PyMC then chooses a "
                        "sensible default)."
                    ),
                    mo.md(f"```python\n{inspect.getsource(solution_debug_model)}\n```"),
                    mo.lazy(
                        lambda: mo.md(f"Result: `{solution_debug_model()}`"),
                        show_loading_indicator=True,
                    ),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Custom Distributions

    If you have a well-behaved density function, we can use it in a model to build a model log-likelihood function. Almost any Pytensor function can be turned into a
    distribution using the `CustomDist` function. For example, a **uniformly-distributed** stochastic variable could be created manually from a function that computes its log-probability as follows:
    """)
    return


@app.cell
def _():
    def uniform_logp(value, lower, upper):
        return pm.math.switch(
            (value > upper) | (value < lower), -np.inf, -pm.math.log(upper - lower)
        )

    with pm.Model():
        u = pm.CustomDist("u", 0, 10, logp=uniform_logp, dtype="float32")
    return (u,)


@app.cell
def _(u):
    pm.logp(u, 1).eval()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Passing values outside the support of the distribution to `logp()` will return `-inf`, since the value has no probability.
    """)
    return


@app.cell
def _(u):
    pm.logp(u, -4).eval()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To emphasize, the Python function passed to `CustomDist` should compute the *log*-density or *log*-probability of the variable. That is why the return value in the example above is `-log(upper - lower)` rather than `1/(upper - lower)`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Generative `CustomDist`

    Writing a log-density by hand is one option; often there is a better one. If your distribution can be *constructed* from existing distributions and PyTensor operations, pass a generative function via `dist=` instead of `logp=`. PyMC traces the function into a graph and, when that graph is invertible, derives the log-probability for you. Random draws come along for free, since the function *is* the simulator.

    Here is an exponential waiting time with a guaranteed minimum delay:
    """)
    return


@app.cell
def _():
    def shifted_exponential(lam, shift, size):
        return shift + pm.Exponential.dist(lam, size=size)

    with pm.Model():
        wait_time = pm.CustomDist("wait_time", 2.0, 1.0, dist=shifted_exponential)

    pm.draw(wait_time, draws=5, random_seed=RANDOM_SEED)
    return (wait_time,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That single definition also yields the correct log-probability, including `-inf` below the minimum delay, with no hand-written density in sight:
    """)
    return


@app.cell
def _(wait_time):
    pm.logp(wait_time, 1.5).eval(), pm.logp(wait_time, 0.5).eval()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Exercise: A Triangular Distribution

    Build a **triangular distribution** with `pm.CustomDist`. The triangular
    distribution on $[l, u]$ with mode $m$ has density

    $$f(x) = \begin{cases} \dfrac{2(x-l)}{(u-l)(m-l)} & l \le x < m \\[6pt] \dfrac{2(u-x)}{(u-l)(u-m)} & m \le x \le u \\[6pt] 0 & \text{otherwise} \end{cases}$$

    1. Write `triangular_logp(value, lower, mode, upper)` returning the **log**-density, using `pm.math.switch` as in the uniform example above.
    2. Create a `CustomDist` named `"tri"` with `lower=0`, `mode=2`, `upper=5`.
    3. Check your work with `pm.logp`: the log-density at the mode should be $\log(2/(u-l)) = \log(0.4)$, and any value outside $[0, 5]$ should give `-inf`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "Hint": mo.md(
                "Nest one `pm.math.switch` inside another: the outer switch "
                "handles out-of-support values (return `-np.inf`), the inner "
                "one picks the rising or falling branch. Remember to return "
                "the *log* of the density."
            ),
        }
    )
    return


@app.function
def exercise_triangular():
    def triangular_logp(value, lower, mode, upper):
        # YOUR CODE HERE — nested pm.math.switch: the outer switch handles
        # out-of-support values (-np.inf), the inner one picks the rising
        # or falling branch. Return the *log* of the density.
        ...

    with pm.Model():
        tri = ...
    return pm.logp(tri, 2).eval(), pm.logp(tri, -1).eval()


@app.cell(hide_code=True)
def _():
    run_triangular = mo.ui.run_button(label="▶ Run exercise")
    run_triangular
    return (run_triangular,)


@app.cell(hide_code=True)
def _(run_triangular):
    mo.stop(
        not run_triangular.value,
        mo.md("*Click ▶ Run exercise once your code is ready.*"),
    )
    exercise_triangular()
    return


@app.cell(hide_code=True)
def _():
    def solution_triangular():
        def triangular_logp(value, lower, mode, upper):
            return pm.math.switch(
                (value < lower) | (value > upper),
                -np.inf,
                pm.math.switch(
                    value < mode,
                    pm.math.log(2 * (value - lower))
                    - pm.math.log((upper - lower) * (mode - lower)),
                    pm.math.log(2 * (upper - value))
                    - pm.math.log((upper - lower) * (upper - mode)),
                ),
            )

        with pm.Model():
            tri = pm.CustomDist("tri", 0, 2, 5, logp=triangular_logp)
        return pm.logp(tri, 2).eval(), pm.logp(tri, -1).eval()

    mo.accordion(
        {
            "Solution": mo.vstack(
                [
                    mo.md(f"```python\n{inspect.getsource(solution_triangular)}\n```"),
                    mo.lazy(
                        lambda: mo.md(f"Result: `{solution_triangular()}`"),
                        show_loading_indicator=True,
                    ),
                ]
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
