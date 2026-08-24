# Evidence and Bayes factors

## Direction convention

For models `A` and `B`:

```text
log_BF_A_vs_B = log p(data | A) - log p(data | B)
BF_A_vs_B     = p(data | A) / p(data | B)
```

Positive log Bayes factors support `A`; negative values support `B`. The model
named before `_vs_` is always the numerator. BARRACUDA retains natural-log and
base-10 values because raw Bayes factors can overflow.

Tables comparing every candidate to the best model use two explicit columns:

- `log10_BF_model_vs_best` is non-positive except for ties;
- `log10_BF_best_vs_model` is non-negative except for numerical tolerance.

Do not infer direction from a plot title alone; preserve the column name.

## Pairwise comparisons

```python
from barracuda import pairwise_bayes_factors

table = pairwise_bayes_factors(
    {"homo": -120.4, "dis2p": -103.1, "hetero3": -101.8}
)
```

Each row is one unordered pair and contains both evidence values, directed log
and base-10 Bayes factors, the favored model, and a descriptive strength label.
Strength categories do not replace the magnitude, direction, or scientific
context.

## Posterior model probabilities

`posterior_model_probabilities` combines marginal likelihoods with explicit
model prior probabilities. Equal priors are the default. Supplied weights must
name every model, be finite and positive, and are normalized internally.

Posterior model probabilities are conditional on the candidate set: omitting a
plausible model changes their interpretation.

## Combining independent evidence

`combine_independent_evidence` sums log evidence by model across rows. This is
valid only when datasets are scientifically independent conditional on each
model and use the same model definitions and compatible priors. Repeated views,
cumulative prefixes, duplicated cells, or correlated conditions must not be
treated as independent evidence contributions.

When a `condition` column is present it is used as the dataset identifier.
Duplicate dataset/model rows and incomplete model coverage are rejected by
default, preventing a model from appearing to win because a difficult dataset
was silently omitted.

## Savage–Dickey point-null evidence

`savage_dickey_ratio` estimates prior and posterior densities at a reference
point using Gaussian KDE:

- `bf_01 = posterior_density / prior_density` supports the point null;
- `bf_10 = prior_density / posterior_density` supports the alternative.

The Savage–Dickey identity requires compatible nuisance-parameter priors in the
nested and encompassing models. KDE at bounded or boundary nulls is biased
without boundary correction; use an appropriate specialized method instead.
At least two finite, non-degenerate prior and posterior draws are required.

`history_effect_bayes_factors` applies the calculation to available trajectory
history coefficients and omits coefficients absent from a history-independent
model.

## SMC uncertainty

A marginal likelihood is an estimated quantity. Compare chain-level estimates
with `smc_log_evidence_by_chain` and `smc_evidence_summary`, repeat sensitive
comparisons with independent seeds, and report instability. A large estimated
Bayes factor from an unstable SMC run is not reliable evidence.
