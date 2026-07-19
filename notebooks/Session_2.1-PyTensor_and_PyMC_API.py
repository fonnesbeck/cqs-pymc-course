import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import base64
    import inspect
    from pathlib import Path
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
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


@app.cell(hide_code=True)
def header():
    mo.md("""
    # Session 2.1: PyTensor and the PyMC API

    In this session we look under PyMC's hood. PyMC is built on **PyTensor**, a library for defining and compiling computational graphs. We'll learn just enough PyTensor to read the graphs PyMC builds — then use that lens to understand what a PyMC model actually *is*, and tour the parts of the PyMC API you'll use in every model: distributions, `logp` and `draw`, `pm.math`, and the model-debugging toolkit.
    """)
    return


@app.cell(hide_code=True)
def _():
    def render_pytensor_logo():
        img_path = Path(__file__).parent / "images" / "PyTensor_RGB.png"
        img_html = ""
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            img_html = f'<img src="data:image/png;base64,{b64}" width="400">'
        return img_html

    pytensor_html = render_pytensor_logo()

    mo.md(f"""
    ## PyTensor Basics

    {pytensor_html}

    PyTensor is the computational backend of PyMC: every model you build with PyMC is, underneath, a PyTensor **computational graph**. You will rarely write PyTensor directly — the goal of this section is to learn just enough to *read* what PyMC builds for you. That literacy pays off throughout the course: it explains what model variables really are, what "compiling" a model means, and how to decode PyTensor's error messages when something goes wrong.

    In PyTensor, you define a computational graph explicitly. You start with input variables that are essentially placeholders, and from these build intermediate variables by applying operators. While PyTensor is designed to feel similar to NumPy, there is a key difference: PyTensor operations build a graph of computations to be executed **lazily**, rather than immediately returning values.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tensors and Basic Operations

    To begin, let's define some PyTensor tensors and show how to perform some basic operations.
    """)
    return


@app.cell(hide_code=True)
def _():
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
    buf = io.StringIO()
    w.dprint(file=buf)
    mo.md(f"```\n{buf.getvalue()}```")
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
    These automatic simplifications are called **graph rewrites**. Because your code builds a graph rather than executing immediately, PyTensor is free to replace pieces of that graph with cheaper or safer equivalents before anything runs. Beyond removing redundant arithmetic, PyTensor can recognize and replace many common expressions with more numerically stable equivalents — for example, `log(1 + x)` is rewritten to use `log1p`, which stays accurate when `x` is very close to zero. (This is not a blanket guarantee: custom expressions can still be numerically unstable.)

    When PyMC compiles your model's log-probability, all of these rewrites are applied automatically — you get the optimized, stabilized graph without asking for it.
    """)
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

    Defining variables, building expressions, compiling functions — everything we just did by hand is exactly what PyMC automates. Bayesian inference begins with a probability model that relates unknown parameters to observed data; PyMC provides high-level building blocks for constructing these models, and **every one of those building blocks is a PyTensor object underneath**. In this section we'll build the smallest possible model and inspect it with the graph-reading tools from the previous section.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model Contexts and Random Variables

    As we have seen, the canonical way to specify PyMC models is using a `Model` context manager. Generally speaking, a context manager is a Python idiom that define what happens when entering and exiting a with statement. They provide a clean, reliable way to set up and tear down resources,

    As an analogy, `Model` is a tape machine that records what is being added to the model; it keeps track the random variables (observed or unobserved) and other model components. The model context then computes some simple model properties, builds a **bijection** mapping that transforms between Python dictionaries and numpy/Pytensor ndarrays.

    More importantly, a `Model` contains methods to compile Pytensor functions that take Random Variables--that are also
    initialised within the same model--as input.
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
    What did `pm.Normal("z", ...)` actually create? Not a special PyMC object — a plain PyTensor `TensorVariable`, whose `owner.op` is a **`RandomVariable`** operation. This is the same graph anatomy we inspected earlier:
    """)
    return


@app.cell
def _(z_1):
    _buf = io.StringIO()
    z_1.dprint(file=_buf)
    mo.vstack(
        [
            mo.md(
                f"`type(z_1)` = `{type(z_1).__name__}`, "
                f"`z_1.owner.op` = `{type(z_1.owner.op).__name__}`"
            ),
            mo.md(f"```\n{_buf.getvalue()}```"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Asking for the log-probability of a value builds a **new** graph — the density expression — from the random variable's graph:
    """)
    return


@app.cell
def _(z_1):
    z_logp = pm.logp(z_1, 2.5)
    _buf = io.StringIO()
    z_logp.dprint(file=_buf)
    mo.md(f"```\n{_buf.getvalue()}```")
    return (z_logp,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A graph is not a number — to evaluate it we compile, exactly as before. `model.compile_logp()` applies the same compile-then-call pattern to the model's **joint** log-probability graph (the sum over all of the model's variables). Our model has a single variable, so the joint log-probability matches the single-variable graph we just printed:
    """)
    return


@app.cell
def _(model, z_logp):
    z_logp.eval(), model.compile_logp()({"z": 2.5})
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

    These building blocks connect in a directed acyclic graph (DAG) that completely specifies a joint probability distribution. PyMC leverages this graph structure to efficiently sample from the posterior distribution using its available inference algorithms.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Distributions and Random Variables

    Statistical distributions are provided in PyMC as subclasses of `Distribution` — `pm.Normal`, `pm.Binomial`, and so on. Calling one does **not** return a distribution object: as we saw in the bridge section, it creates a PyTensor `TensorVariable` whose `owner.op` is a `RandomVariable` (such as `NormalRV`), and registers that variable with the enclosing `Model`. This registration is why the named form is only usable inside a model context.

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
    Every distribution knows how to do two things: **generate random draws** and **compute the log-probability of a value**. Random generation is implemented by the underlying `RandomVariable` operation — the `NormalRV` we saw in the graph — and log-probabilities are provided through PyMC's `logp` dispatch system.

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
    Distributions will optionally have `cdf` and `icdf` methods, representing the cumulative distribution function and inverse cumulative distribution functions, respectively.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Sometimes we wish to use a distribution without adding a variable to a model — for example, just to generate random numbers. For this purpose, `Distribution` classes have a `.dist()` class method that returns an **unregistered** random-variable tensor: the same kind of `TensorVariable` as before, but not attached to any model, so it can be created and used anywhere.
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

    When you need mathematical operations on model variables — link functions, conditional logic, matrix products — reach for the **`pm.math`** namespace. These are the same symbolic PyTensor operations from the start of this session, collected under a PyMC-friendly name. Commonly used members include `pm.math.log`, `exp`, `sqrt`, `dot`, `sum`, `switch` (element-wise if/else), and the link functions `invlogit`/`logit`.

    Because they are symbolic, they add nodes to the graph rather than computing immediately — so they can be applied to model variables just like `+` or `*`. Later sessions use `pm.math.switch` and `pm.math.invlogit` without further introduction; this is where they come from.
    """)
    return


@app.cell
def _():
    grid = np.linspace(-3, 3, 7)
    pm.math.invlogit(grid).eval(), pm.math.switch(grid > 0, 1.0, 0.0).eval()
    return (grid,)


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
