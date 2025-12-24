# Orca



---

<br>

## Method

### Model

Cytotoxic killing is represented as a **continuous-time stochastic process** in which each killer cell generates target-contact events and, at each contact, makes a **kill vs non-kill** decision.

**State (per killer cell \(i\))**
- \(x_i\): cumulative number of **non-lethal** contacts  
- \(y_i\): cumulative number of **lethal** contacts  
- \(K_i\): **killing capacity** (maximum number of kills available); \(K^{\text{rem}}_i\) decreases by 1 after each kill

**Event generation (global Gillespie step)**
- Each alive killer has an encounter rate \(\lambda_i = r(x_i, y_i)\) (default: constant rate matrix).
- The next event time is sampled as  
  \[
  \Delta t \sim \mathrm{Exp}\!\left(\sum_i \lambda_i\right).
  \]
- The killer responsible for the event is selected with probability  
  \[
  \Pr(i\ \text{chosen})=\lambda_i / \sum_j \lambda_j.
  \]

**Target depletion (finite target pool)**
- Initial target count \(T_0\) is set either explicitly or as a multiple of killers (default: \(T_0 = 4\,N_{\text{killers}}\)).
- Remaining targets \(T\) decrease by 1 after each successful kill.
- Encounter rates are scaled by a depletion factor  
  \[
  f(T)=\max\!\left(\frac{T}{T_0},\ \text{target\_rate\_floor}\right),
  \qquad \lambda_i \leftarrow \lambda_i \, f(T),
  \]
  so encounters become less frequent as targets are depleted (with an optional floor).

**Decision rule at each contact**
- **Deterministic mode:** \(p_{\text{kill}}(x_i,y_i)=1\) while \(K^{\text{rem}}_i>0\); otherwise \(p_{\text{kill}}=0\).
- **Stochastic mode:** \(p_{\text{kill}}(x_i,y_i)=p(x_i,y_i)\) from a probability matrix:
  - **Constant:** \(p(x,y)=p_0\).
  - **History-dependent (logistic):**
    \[
    \mathrm{logit}\,p(x,y)=\mathrm{logit}(p_0)+\alpha x+\beta y,
    \qquad p(x,y)=\sigma(\mathrm{logit}\,p(x,y)),
    \]
    where \(\sigma(\cdot)\) is the sigmoid function. This guarantees \(p\in(0,1)\) (with optional numerical clipping).

**Killing capacity \(K_i\) generation**
- **Homogeneous:** \(K_i = K_0\) for all killers.
- **Heterogeneous (continuous):** \(K_i\) sampled from Normal / Lognormal / Gamma, rounded to integers.
- **Heterogeneous (discrete mixture):** \(K_i\) sampled from a mixture over communities with specified proportions and community-specific capacities.
- Optional clipping to \([K_{\min}, K_{\max}]\).

**Outputs**
- Per-killer decision sequences (0 = non-lethal, 1 = lethal) and corresponding event times
- Final \((x_i, y_i)\), capacities \(K_i\), remaining capacities \(K^{\text{rem}}_i\)
- Target trajectory \(T(t)\) over event times

### Bayesian Inference

---

