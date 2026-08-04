from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta as beta_distribution

from webapp.ui import hero, note, step_card


hero(
    "0 · Foundations",
    "Bayesian inference, from theorem to computation",
    "Build an intuition for priors, likelihoods and posteriors—then see why MCMC, SMC and Bayes factors are useful for real scientific models.",
    badge="Interactive primer · no data upload required",
)

note(
    "A small historical correction",
    "The theorem is named after the Reverend Thomas Bayes, not James Bayes. His unfinished inverse-probability work was published after his death in 1763.",
    tone="navy",
)

st.header("The update at the heart of Bayesian inference")
st.latex(r"p(\theta \mid y, M)=\frac{p(y\mid\theta,M)\,p(\theta\mid M)}{p(y\mid M)}")
st.markdown("**Posterior = likelihood × prior ÷ evidence.**")

term_columns = st.columns(4)
with term_columns[0]:
    step_card("01", "Prior", "Plausible parameter values before the current dataset is observed.")
with term_columns[1]:
    step_card("02", "Likelihood", "How compatible each parameter value is with the observed data under the model.")
with term_columns[2]:
    step_card("03", "Posterior", "The updated uncertainty after the prior and likelihood have been combined.")
with term_columns[3]:
    step_card("04", "Evidence", "The likelihood averaged over the prior; it normalises the posterior and compares models.")

st.subheader("Try a small exact update")
st.caption(
    "This Beta–Binomial example is for teaching. The ORCA event-count models use different likelihoods and numerical inference."
)

controls, chart = st.columns([0.32, 0.68], gap="large")
with controls:
    prior_alpha = st.slider("Prior α", 1, 20, 2, help="Prior event-like observations plus one.")
    prior_beta = st.slider("Prior β", 1, 30, 8, help="Prior non-event-like observations plus one.")
    opportunities = st.slider("Observed opportunities", 1, 100, 20)
    successes = st.slider("Observed events", 0, opportunities, min(8, opportunities))

posterior_alpha = prior_alpha + successes
posterior_beta = prior_beta + opportunities - successes
x = np.linspace(0.001, 0.999, 500)
prior_density = beta_distribution.pdf(x, prior_alpha, prior_beta)
posterior_density = beta_distribution.pdf(x, posterior_alpha, posterior_beta)

with chart:
    figure, axis = plt.subplots(figsize=(8, 3.8))
    axis.plot(x, prior_density, color="#647988", linewidth=2.2, label="Prior")
    axis.plot(x, posterior_density, color="#007C83", linewidth=2.8, label="Posterior")
    axis.fill_between(x, posterior_density, color="#007C83", alpha=0.12)
    axis.set(xlabel="Event probability θ", ylabel="Probability density", xlim=(0, 1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

prior_mean = prior_alpha / (prior_alpha + prior_beta)
posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
posterior_interval = beta_distribution.ppf([0.025, 0.975], posterior_alpha, posterior_beta)
metric_columns = st.columns(3)
metric_columns[0].metric("Prior mean", f"{prior_mean:.3f}")
metric_columns[1].metric("Posterior mean", f"{posterior_mean:.3f}")
metric_columns[2].metric("Posterior 95% interval", f"{posterior_interval[0]:.3f}–{posterior_interval[1]:.3f}")

st.header("When the posterior is not available on paper")
mcmc_tab, smc_tab = st.tabs(["MCMC", "SMC"])
with mcmc_tab:
    st.markdown(
        """
        **Markov chain Monte Carlo** builds one or more paths through parameter space. After successful
        warm-up, a well-behaved chain spends time in posterior regions in proportion to their probability.

        - **Warm-up/tuning** adapts the sampler and is normally discarded.
        - **Draws** are retained posterior samples; more draws reduce Monte Carlo noise.
        - **Chains** are independent runs that help reveal poor exploration.
        - **Target acceptance** adjusts the step behaviour of HMC/NUTS samplers.
        - **Cores** change speed, not the amount of information in the data.
        """
    )
with smc_tab:
    st.markdown(
        """
        **Sequential Monte Carlo** starts with a population of particles from the prior, then moves through
        intermediate distributions until the particles approximate the posterior.
        """
    )
    st.latex(r"\pi_\beta(\theta)\propto p(y\mid\theta)^\beta p(\theta),\qquad 0\leq\beta\leq1")
    st.markdown(
        """
        - **Particles** (called `draws` by PyMC SMC) control the approximation's resolution.
        - **Chains** provide independent SMC estimates for repeatability checks.
        - **Threshold** controls the tempering schedule and typically the number of stages.
        - **Correlation threshold** controls the mutation effort applied to particles.
        - SMC also estimates the marginal likelihood used in the model comparisons on this site.
        """
    )

note(
    "What more computation can—and cannot—do",
    "More draws or particles can reduce Monte Carlo error. They cannot create information absent from the data, identify an unidentifiable parameter, or repair a misspecified model.",
    tone="amber",
)

st.header("Marginal likelihoods and Bayes factors")
left, right = st.columns(2, gap="large")
with left:
    st.latex(r"p(y\mid M)=\int p(y\mid\theta,M)p(\theta\mid M)\,d\theta")
    st.markdown(
        "The **marginal likelihood** asks how well a model predicted the observed data on average over the parameter values allowed by its prior."
    )
with right:
    st.latex(r"BF_{12}=\frac{p(y\mid M_1)}{p(y\mid M_2)}")
    st.markdown(
        "A **Bayes factor** of 10 means the observed data have ten times the marginal likelihood under model 1 than model 2, given their stated priors."
    )

bf_table = pd.DataFrame(
    {
        "log10 BF₁₂": [2, 1, 0, -1, -2],
        "Marginal-likelihood ratio M₁:M₂": ["100:1", "10:1", "1:1", "1:10", "1:100"],
    }
)
st.dataframe(bf_table, hide_index=True, width="stretch")
st.caption(
    "A Bayes factor compares only the specified models, depends on their priors, and is not the probability that either model is true."
)

st.header("Can I trust the result?")
trust_columns = st.columns(3)
with trust_columns[0]:
    step_card("A", "Check computation", "Compare independent runs, inspect diagnostics, and increase particles or draws to test stability.")
with trust_columns[1]:
    step_card("B", "Check the model", "Use prior- and posterior-predictive checks for zeros, tails, dispersion and donor variation.")
with trust_columns[2]:
    step_card("C", "Check recovery", "Repeat simulations at known truths; one successful synthetic dataset is a demonstration, not calibration.")

st.header("Meet Thomas Bayes")
st.markdown(
    """
    The Reverend **Thomas Bayes** (c. 1701–1761) was an English nonconformist minister and Fellow of the
    Royal Society. His friend Richard Price communicated Bayes' unfinished work after his death, and it
    appeared in *Philosophical Transactions* in 1763. No portrait is definitively authenticated, so this
    page avoids presenting the familiar disputed image as fact.

    **Further reading:** [Bayes' original 1763 paper](https://doi.org/10.1098/rstl.1763.0053) ·
    [PyMC SMC documentation](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.smc.sample_smc.html) ·
    [Modern R-hat and effective sample size](https://doi.org/10.1214/20-BA1221) ·
    [Bayesian workflow](https://arxiv.org/abs/2011.01808)
    """
)
