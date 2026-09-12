"""Selected scientific references for the Bayesian inference learning pathway."""

from dash import html

from webapp.ui import markdown, note


def _link(label: str, href: str) -> html.A:
    return html.A(label, href=href, target="_blank", rel="noreferrer")


def _reference(title: str, citation: str, href: str, annotation: str, *links: tuple[str, str]) -> html.Li:
    return html.Li([
        html.Div(_link(title, href), className="barracuda-learning-reference-title"),
        html.P(citation),
        html.P(annotation, className="barracuda-learning-annotation"),
        *[html.Div(_link(label, url)) for label, url in links],
    ])


def _stage(number: str, title: str, explanation: str, start: int, references: list, equation: str = "") -> html.Details:
    children = [
        html.Summary([
            html.Span(number, className="barracuda-learning-stage-number"),
            html.H3(title),
        ]),
        markdown(explanation, mathjax=True),
    ]
    if equation:
        children.append(markdown(equation, class_name="barracuda-learning-equation", mathjax=True))
    for index, reference in enumerate(references, start=start):
        reference.id = f"learning-ref-{index}"
    children.append(html.Ol(references, start=start, className="barracuda-learning-references"))
    return html.Details(children, open=True, className="barracuda-learning-stage")


def learning_resources() -> html.Section:
    """Render an annotated pathway from statistical foundations to neural inference."""
    methods = [
        ("Analytic / finite exact inference", "Conjugacy, tractable algebra or a finite state space", "Exact distribution under the stated model; numerical evaluation still has finite precision. [1, 6]"),
        ("Quadrature / Laplace / INLA", "Evaluable densities with manageable numerical structure", "Discretization or distributional approximation; INLA exploits latent Gaussian structure. [1, 7, 8]"),
        ("Variational inference", "A tractable family and an optimization objective", "A fitted approximation whose restrictions and optimization affect accuracy. [9]"),
        ("MCMC, including HMC", "An evaluable unnormalized target; gradients for HMC", "Finite correlated draws estimate posterior quantities, with convergence and Monte Carlo uncertainty. [10–12]"),
        ("Sequential Monte Carlo", "A sequence of targets and suitable proposals / weights", "A finite weighted particle approximation; weight degeneracy and lost diversity require attention. [13]"),
        ("Approximate Bayesian computation", "Simulation is possible; likelihood evaluation is impractical", "Tolerance, summaries and finite simulation affect the approximation. [14–16]"),
        ("Neural simulation-based inference", "Simulated parameter–data pairs and a trained surrogate", "Neural approximation, simulation coverage and any subsequent sampling affect accuracy. [17–21]"),
    ]
    return html.Section([
        html.Span("06 · Further learning", className="barracuda-section-label"),
        html.H2("Selected references and learning pathway"),
        html.P(
            "A curated route through probability, statistical inference and Bayesian computation. "
            "The references combine textbooks, university teaching materials, original methods papers "
            "and runnable documentation. Follow the stages in order or expand a topic of interest.",
            className="barracuda-learning-intro",
        ),
        note(
            "One inferential framework, several computational routes",
            "Frequentist and Bayesian inference differ in their inferential interpretation. "
            "Analytic calculation, numerical approximation and simulation describe how an inference "
            "is computed. An exact calculation is exact for its assumptions; it does not make the model true.",
            tone="navy",
        ),
        html.Div(html.Table([
            html.Caption("Computational methods: requirements and sources of approximation"),
            html.Thead(html.Tr([html.Th(label, scope="col") for label in ("Method", "Requirements", "Interpretation")])),
            html.Tbody([html.Tr([html.Th(row[0], scope="row"), html.Td(row[1]), html.Td(row[2])]) for row in methods]),
        ], className="barracuda-simple-table"), className="barracuda-simple-table-wrap barracuda-learning-comparison", tabIndex=0,
            role="region", **{"aria-label": "Scrollable comparison of inference methods"}),
        _stage("01", "Probability, likelihood and statistical inference",
            "Begin with conditional probability, sampling models and the distinction between a likelihood "
            "and a probability distribution over parameters. Bayesian inference conditions a prior on observations; "
            "frequentist procedures are evaluated through repeated sampling. A 95% credible interval contains "
            "95% posterior probability under the specified model and prior. A 95% confidence procedure has "
            "95% coverage across repeated datasets under its assumptions. These interpretations differ even "
            "when numerical endpoints are similar. (References 1–5.)", 1, [
                _reference("Bayesian Data Analysis, third edition",
                    "Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2013). Chapman & Hall/CRC.",
                    "https://sites.stat.columbia.edu/gelman/book/",
                    "A comprehensive reference: probability and conjugate models (chapters 1–3), model checking and comparison (6–7), decision analysis (9), and computation (10–13). The author site links the book, lectures and code."),
                _reference("MIT 18.05: Introduction to Probability and Statistics",
                    "Orloff, J. & French Kamrin, J. (instructors, Spring 2022). MIT OpenCourseWare.",
                    "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/pages/classes-reading-and-in-class-materials/",
                    "A structured foundation in conditional probability, random variables, likelihood, Bayesian updating and frequentist inference, with worked problems."),
                _reference("MIT 18.650: Statistics for Applications",
                    "Rigollet, P. (2016). MIT OpenCourseWare.",
                    "https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/",
                    "Mathematical development of likelihood estimation, testing, regression and Bayesian statistics. Includes lecture videos, slides and assignments."),
                _reference("Probability intervals",
                    "Orloff, J. & Bloom, J. (2022). MIT 18.05, reading 16b.",
                    "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class16-prep-b.pdf",
                    "Connects the area under a posterior density to uncertainty about a parameter and compares intervals containing the same probability mass."),
                _reference("Confidence Intervals: Three Views",
                    "Orloff, J. & Bloom, J. (2022). MIT 18.05, reading 23a.",
                    "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class23-prep-a.pdf",
                    "Explains coverage through repeated experiments and contrasts confidence intervals with Bayesian probability intervals."),
            ]),
        _stage("02", "Exact solutions and deterministic approximations",
            "Use an analytic solution when the model permits it. For the coin experiment, Beta–Binomial "
            "conjugacy gives the posterior directly; sampling is unnecessary. Finite discrete models may also "
            "permit exact summation. Beyond these cases, numerical quadrature approximates integrals; Laplace "
            "methods use local curvature; INLA uses nested approximations for latent Gaussian models; "
            "variational inference optimizes a distribution within a chosen family. These are computational "
            "alternatives with different restrictions, not interchangeable guarantees of accuracy. (References 1, 6–9.)", 6, [
                _reference("Conjugate priors: Beta and normal",
                    "Orloff, J. & Bloom, J. (2022). MIT 18.05, reading 15.",
                    "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class15-prep.pdf",
                    "Derives Beta–Binomial and normal conjugate updates. A uniform Beta(1, 1) prior is flat in the success-probability parameter, not concentrated at a fair coin."),
                _reference("Accurate Approximations for Posterior Moments and Marginal Densities",
                    "Tierney, L. & Kadane, J. B. (1986). Journal of the American Statistical Association 81, 82–86.",
                    "https://www.math.mcgill.ca/dstephens/680/Handouts/OldPDFs/TierneyKadane-1986-JASA.pdf",
                    "Develops Laplace approximations to posterior integrals. Their accuracy depends on regularity and the geometry of the posterior, including concentration around a dominant mode."),
                _reference("Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations",
                    "Rue, H., Martino, S. & Chopin, N. (2009). Journal of the Royal Statistical Society B 71, 319–392.",
                    "https://doi.org/10.1111/j.1467-9868.2008.00700.x",
                    "Introduces INLA for latent Gaussian models, combining Laplace approximation with numerical integration over hyperparameters."),
                _reference("Variational Inference: A Review for Statisticians",
                    "Blei, D. M., Kucukelbir, A. & McAuliffe, J. D. (2017). Journal of the American Statistical Association 112, 859–877.",
                    "https://arxiv.org/abs/1601.00670",
                    "Explains approximation through optimization, including mean-field families and stochastic optimization. A converged optimizer need not imply an accurate posterior approximation."),
            ], r"$$\theta\sim\mathrm{Beta}(a,b),\quad h\mid\theta\sim\mathrm{Binomial}(n,\theta)"
               r"\quad\Longrightarrow\quad\theta\mid h,n\sim\mathrm{Beta}(a+h,b+n-h).$$"),
        _stage("03", "Why Monte Carlo? Markov chains and Hamiltonian dynamics",
            "Scientific questions often require posterior integrals, such as an expectation, a tail probability "
            "or a predictive distribution. These can be difficult even when the unnormalized density is evaluable. "
            "MCMC constructs a chain with the desired invariant distribution and estimates integrals from its "
            "states. HMC uses gradients to explore suitable continuous targets efficiently. Finite correlated "
            "draws remain an approximation: initialization, mixing, Monte Carlo error and numerical pathologies "
            "must be assessed. Multiple chains and diagnostics provide evidence, not proof of convergence. (References 10–12, 25.)", 10, [
                _reference("Monte Carlo sampling methods using Markov chains and their applications",
                    "Hastings, W. K. (1970). Biometrika 57, 97–109.",
                    "https://doi.org/10.1093/biomet/57.1.97",
                    "The general Metropolis–Hastings construction: account for both the target-density ratio and the forward/reverse proposal probabilities."),
                _reference("A Conceptual Introduction to Hamiltonian Monte Carlo",
                    "Betancourt, M. (2017; revised 2018). Preprint, arXiv:1701.02434.",
                    "https://arxiv.org/abs/1701.02434",
                    "Develops intuition for posterior geometry, the typical set and Hamiltonian exploration, including circumstances in which the method struggles."),
                _reference("Rank-normalization, folding, and localization: An improved R̂ for assessing convergence of MCMC",
                    "Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021). Bayesian Analysis 16, 667–718.",
                    "https://arxiv.org/abs/1903.08008",
                    "Modern multiple-chain diagnostics, rank plots and effective sample sizes. Report Monte Carlo uncertainty for the quantities that support the scientific conclusion."),
            ], r"$$\mathbb E[g(\theta)\mid y]=\int g(\theta)p(\theta\mid y)\,d\theta"
               r"\;\approx\;\frac{1}{M}\sum_{m=1}^{M}g(\theta^{(m)}).$$"),
        _stage("04", "Sequential Monte Carlo and particle populations",
            "SMC carries a weighted population through a sequence of targets. Importance weights represent the "
            "change in target; resampling allocates copies according to those weights; mutation moves particles "
            "to explore the new target. In tempered SMC, the same dataset is introduced gradually through a "
            "likelihood exponent β. This is distinct from filtering or updating with genuinely new observations. "
            "Resampling reduces weight imbalance but can remove diversity; it does not add information. (References 13.)", 13, [
                _reference("Sequential Monte Carlo samplers",
                    "Del Moral, P., Doucet, A. & Jasra, A. (2006). Journal of the Royal Statistical Society B 68, 411–436.",
                    "https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf",
                    "A general framework for particle approximations across distributions, including importance weights, resampling, MCMC mutation and normalizing-constant estimation."),
            ], r"$$\pi_t(\theta)\propto p(\theta)L(y\mid\theta)^{\beta_t},\qquad"
               r"0=\beta_0<\cdots<\beta_T=1.$$"),
        _stage("05", "Approximate Bayesian computation: infer through simulation",
            "When a simulator can generate data but its likelihood is impractical to evaluate, ABC can compare "
            "simulations with observations. Rejection ABC draws a parameter from the prior, simulates data, compares summary "
            "statistics and retains parameters whose discrepancy is below a tolerance ε. Smaller tolerances "
            "typically cost more simulations. Under suitable conditions, shrinking ε approaches inference "
            "conditional on the summaries; recovering the full-data posterior also requires retaining its "
            "relevant information. ABC-SMC improves exploration through decreasing tolerances, but does not "
            "eliminate approximation from the tolerance or summaries. (References 14–16.)", 14, [
                _reference("Approximate Bayesian computation in population genetics",
                    "Beaumont, M. A., Zhang, W. & Balding, D. J. (2002). Genetics 162, 2025–2035.",
                    "https://doi.org/10.1093/genetics/162.4.2025",
                    "An influential ABC development using simulated summaries and local regression adjustment; a starting point for understanding likelihood-free approximation."),
                _reference("Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems",
                    "Toni, T., Welch, D., Strelkowa, N., Ipsen, A. & Stumpf, M. P. H. (2009). Journal of the Royal Society Interface 6, 187–202.",
                    "https://arxiv.org/abs/0901.1925",
                    "ABC-SMC for dynamical models: resampling, perturbation, forward simulation and importance correction across decreasing tolerances."),
                _reference("Constructing summary statistics for approximate Bayesian computation: semi-automatic approximate Bayesian computation",
                    "Fearnhead, P. & Prangle, D. (2012). Journal of the Royal Statistical Society B 74, 419–474.",
                    "https://arxiv.org/abs/1004.1112",
                    "Constructs summaries for a defined inferential objective. Summaries suited to posterior-mean estimation are not automatically sufficient for the full posterior or model selection."),
            ], r"$$p_\varepsilon(\theta\mid s_{\mathrm{obs}})\propto p(\theta)"
               r"\int \mathbf{1}\!\left\{d(s(x),s_{\mathrm{obs}})\leq\varepsilon\right\}p(x\mid\theta)\,dx.$$"),
        _stage("06", "Neural Bayesian inference for simulation models",
            "Neural simulation-based inference learns from simulated parameter–data pairs. Neural posterior "
            "estimation (NPE) fits the conditional posterior directly; neural likelihood estimation (NLE) fits a "
            "likelihood surrogate; neural ratio estimation (NRE) fits a likelihood-to-evidence ratio. NLE and "
            "NRE require a further posterior integration or sampling step after combination with the prior. "
            "Amortization shares training cost across datasets; sequential methods focus simulations on one "
            "observation. These choices change computation, not Bayes’ rule. Finite training data, simulation "
            "coverage, network capacity, optimization and any subsequent sampler affect accuracy. (References 17–21.)", 17, [
                _reference("The frontier of simulation-based inference",
                    "Cranmer, K., Brehmer, J. & Louppe, G. (2020). Proceedings of the National Academy of Sciences USA 117, 30055–30062.",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7720103/",
                    "A map of implicit models, ABC, density and ratio estimation, amortization and sequential design. Distinguishes successful inference under a simulator from adequacy of that simulator."),
                _reference("Automatic Posterior Transformation for Likelihood-Free Inference",
                    "Greenberg, D., Nonnenmacher, M. & Macke, J. (2019). Proceedings of Machine Learning Research 97, 2404–2414.",
                    "https://proceedings.mlr.press/v97/greenberg19a.html",
                    "Sequential neural posterior estimation with correction for training proposals. Samples from a fitted neural density are not automatically exact posterior draws."),
                _reference("Sequential Neural Likelihood: Fast Likelihood-free Inference with Autoregressive Flows",
                    "Papamakarios, G., Sterratt, D. & Murray, I. (2019). Proceedings of Machine Learning Research 89, 837–848.",
                    "https://proceedings.mlr.press/v89/papamakarios19a.html",
                    "Learns a conditional likelihood with normalizing flows, then combines it with the prior for posterior sampling. Includes sequential simulation design."),
                _reference("Likelihood-free MCMC with Amortized Approximate Ratio Estimators",
                    "Hermans, J., Begy, V. & Louppe, G. (2020). Proceedings of Machine Learning Research 119, 4239–4248.",
                    "https://proceedings.mlr.press/v119/hermans20a.html",
                    "Uses classification of joint versus independently paired simulations to estimate a likelihood ratio. The ratio is a surrogate, and MCMC remains part of the inference procedure."),
                _reference("Benchmarking Simulation-Based Inference",
                    "Lueckmann, J.-M., Boelts, J., Greenberg, D., Goncalves, P. & Macke, J. (2021). Proceedings of Machine Learning Research 130, 343–351.",
                    "https://proceedings.mlr.press/v130/lueckmann21a.html",
                    "Compares methods using explicit simulation budgets, tasks and metrics. No method is uniformly best; speed and narrow intervals are not evidence of accuracy."),
            ], r"$$\begin{aligned}\mathrm{NPE}:&\quad q_\phi(\theta\mid x)\approx p(\theta\mid x),\\"
               r"\mathrm{NLE}:&\quad q_\phi(x\mid\theta)\approx p(x\mid\theta),\\"
               r"\mathrm{NRE}:&\quad r_\phi(\theta,x)\approx p(x\mid\theta)/p(x).\end{aligned}$$"),
        _stage("07", "Validation, scientific workflow and practical study",
            "Reliable inference requires more than an attractive posterior plot. Prior predictive checks examine "
            "what the model predicts before fitting; posterior predictive checks compare observable implications "
            "with data; simulation-based calibration checks computation over repeated simulated datasets. "
            "Sampler diagnostics, prior sensitivity and robustness to model choices answer further questions. "
            "Passing a finite diagnostic does not prove exact computation or establish that a simulator describes "
            "the real system. Report the model, observations, priors, computational settings, relevant diagnostics "
            "and limitations so the analysis can be evaluated and reproduced. (References 22–26.)", 22, [
                _reference("Bayesian Workflow",
                    "Gelman, A. et al. (2020). Preprint, arXiv:2011.01808.",
                    "https://arxiv.org/abs/2011.01808",
                    "Treats model building, checking, computational validation, comparison and revision as an iterative scientific process."),
                _reference("Validating Bayesian Inference Algorithms with Simulation-Based Calibration",
                    "Talts, S., Betancourt, M., Simpson, D., Vehtari, A. & Gelman, A. (2018; revised 2020). Preprint, arXiv:1804.06788.",
                    "https://arxiv.org/abs/1804.06788",
                    "Draw generating parameters from the prior, simulate data, refit and inspect ranks among posterior draws. Calibration under these simulations is distinct from adequacy for real observations."),
                _reference("Statistical Rethinking: 2024 Edition",
                    "McElreath, R. (2024). Author-maintained course materials.",
                    "https://github.com/rmcelreath/stat_rethinking_2024",
                    "Lectures, slides, exercises and code connecting scientific questions and causal models to Bayesian estimation, multilevel models and MCMC."),
                _reference("Stan documentation: posterior analysis and decision analysis",
                    "Stan Development Team. Living documentation; accessed 12 September 2026.",
                    "https://mc-stan.org/docs/reference-manual/analysis.html",
                    "Practical explanations of convergence, effective sample size and Monte Carlo error. Decision analysis connects posterior predictions with explicit utilities or losses.",
                    ("Decision analysis and expected utility", "https://mc-stan.org/docs/stan-users-guide/decision-analysis.html")),
                _reference("sbi tutorials: method choice, calibration and model checking",
                    "sbi team. Living documentation; accessed 12 September 2026.",
                    "https://sbi.readthedocs.io/en/stable/how_to_guide/06_choosing_inference_method.html",
                    "Runnable guidance for choosing NPE, NLE or NRE and assessing results. Use computational calibration alongside predictive checks and model-sensitivity assessment.",
                    ("Simulation-based calibration", "https://sbi.readthedocs.io/en/stable/advanced_tutorials/11_diagnostics_simulation_based_calibration.html"),
                    ("Posterior predictive checks", "https://sbi.readthedocs.io/en/stable/advanced_tutorials/10_diagnostics_posterior_predictive_checks.html"),
                    ("Model misspecification", "https://sbi.readthedocs.io/en/stable/how_to_guide/18_model_misspecification.html")),
            ]),
        html.P(
            "Reference links lead to author pages, university materials, original publications or official "
            "documentation. Preprints are identified explicitly; access to some publisher versions may require "
            "a subscription. This selection is a learning pathway, not an exhaustive review.",
            className="barracuda-learning-footer",
        ),
    ], id="learning-resources", className="barracuda-lesson-section barracuda-learning-resources")
