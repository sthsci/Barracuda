# Model catalog

## Event-count likelihoods

For cell `i`, ORCA observes a non-negative event count `n_i` over a positive
observation time `T`. The models distinguish sampling variation, a structural
non-engaging fraction, and continuous cell-to-cell rate heterogeneity.

| Key | Structure | Parameters reported by donor-ignorant fits |
|---|---|---|
| `homo` | One shared Poisson rate | `lambda` |
| `z2p` | Structural zero component plus one shared engaging-cell rate | `lambda`, `p_zero` |
| `dis2p` | Engaging-cell rates follow a Gamma distribution | `mu_lambda`, `sigma_lambda` |
| `hetero3` | Gamma rate heterogeneity plus structural zeros | `mu_lambda`, `sigma_lambda`, `p_zero` |

`p_zero` is a model component, not a generic explanation for every observed
zero. A zero can still arise from the count likelihood for an engaging cell.

## Donor hierarchy

Donor-aware versions use the same four scientific mechanisms but introduce a
population distribution over donor-level parameters. They separate:

- population-level location and dispersion;
- between-donor parameter variation; and
- within-donor cell/count variation.

The public donor utilities canonicalize the historical backend name `phi_0` to
`p_zero`. Variance decomposition depends on the supplied donor weights. A
leave-one-donor-out moment calculation recomputes weighted posterior mixtures;
it is **not** a refit and must not be described as leave-one-out predictive
cross-validation.

## Ordered contact-kill trajectories

Trajectory models jointly represent contact opportunity and the binary lethal
decision. The decision log odds may vary between cells (`sigma_eta`) and may
change with previous non-lethal (`beta_f`) or lethal (`beta_s`) contacts.

| Key | Heterogeneous baseline decision propensity | History effects |
|---|---:|---:|
| `homogeneous_history_independent` | No | No |
| `homogeneous_history_dependent` | No | `beta_f`, `beta_s` |
| `heterogeneous_history_independent` | `sigma_eta` | No |
| `heterogeneous_history_dependent` | `sigma_eta` | `beta_f`, `beta_s` |

Public notation uses `beta_f` and `beta_s`; research backends may store those
variables as `beta_x` and `beta_y`. Public helpers translate the names.

## Nested boundaries

Several simpler mechanisms sit on boundaries of larger models—for example
`sigma_lambda = 0`, `p_zero = 0`, `sigma_eta = 0`, or a history coefficient of
zero. Recovery near a boundary can be asymmetric and weakly identified.
Report boundary-aware summaries and inspect posterior mass rather than relying
only on a posterior mean.

## Priors are part of the model

Bayes factors compare prior-predictive models, not likelihood labels alone.
Changing prior bounds or scales changes the scientific model and its marginal
likelihood. Preserve the complete settings object and do not combine evidence
from fits that used incompatible model definitions or priors.
