# Orca



---

<br>

## Method

### Model

Cytotoxic killing is represented as a continuous-time stochastic process. Each killer cell generates target-contact events, and each contact results in either a kill or a non-kill outcome.

**State (per killer cell $i$)**  
- $x_i$: cumulative number of non-lethal contacts  
- $y_i$: cumulative number of lethal contacts  
- $K_i$: killing capacity (maximum number of kills available); remaining capacity $K_i^{\mathrm{rem}}$ decreases by 1 after each kill  

**Event generation (global Gillespie step)**  
- Each alive killer has an encounter rate $\lambda_i = r(x_i, y_i)$ (default: constant rate matrix).  
- Next event time is sampled as $\Delta t \sim \mathrm{Exp}\!\left(\sum_i \lambda_i\right)$.  
- The killer responsible for the next event is chosen with probability $\Pr(i\ \text{chosen}) = \lambda_i / \sum_j \lambda_j$.  

**Target depletion (finite target pool)**  
- The initial target count is $T_0$, set either explicitly or as a multiple of killers (default: $T_0 = 4\,N_{\mathrm{killers}}$).  
- Remaining targets $T$ decrease by 1 after each successful kill.  
- Encounter rates are scaled by a depletion factor $f(T) = \max\!\left(T/T_0,\ f_{\min}\right)$, implemented as $\lambda_i \leftarrow \lambda_i\, f(T)$, where `f_min` corresponds to `target_rate_floor`.  

**Decision rule at each contact**  
- Deterministic mode: $p_{\mathrm{kill}}(x_i,y_i)=1$ while $K_i^{\mathrm{rem}}>0$; otherwise $p_{\mathrm{kill}}=0$.  
- Stochastic mode: $p_{\mathrm{kill}}(x_i,y_i)=p(x_i,y_i)$ from a probability matrix:  
  - Constant: $p(x,y)=p_0$.  
  - History-dependent (logistic): $\mathrm{logit}\,p(x,y)=\mathrm{logit}(p_0)+\alpha x+\beta y$, with $p(x,y)=\sigma(\mathrm{logit}\,p(x,y))$, where $\sigma(\cdot)$ is the sigmoid (ensuring $p\in(0,1)$ up to numerical clipping).  

**Killing capacity $K_i$ generation**  
- Homogeneous: $K_i = K_0$ for all killers.  
- Heterogeneous (continuous): $K_i$ sampled from Normal / Lognormal / Gamma distributions and rounded to integers.  
- Heterogeneous (discrete mixture): $K_i$ sampled from a mixture of communities with specified proportions and community-specific capacities.  
- Optional clipping to $[K_{\min}, K_{\max}]$.  

**Outputs**  
- Per-killer decision sequences (0 = non-lethal, 1 = lethal) and corresponding event times  
- Final $(x_i, y_i)$, capacities $K_i$, remaining capacities $K_i^{\mathrm{rem}}$  
- Target trajectory $T(t)$ evaluated at event times  



---
**Description**
This simulator models cytotoxic killing as a continuous-time stochastic process in which multiple killer cells generate target-contact events and, at each contact, decide whether the interaction is lethal or non-lethal. Each killer keeps track of its cumulative non-lethal contacts and lethal kills, and is assigned a finite killing capacity that limits how many kills it can perform before becoming incapable of killing. Contact events are scheduled using a global Gillespie step: at any moment, all alive killers contribute an encounter rate (from a rate matrix that can depend on their contact history), these rates are summed to determine the waiting time to the next event, and the killer responsible for that event is sampled in proportion to its rate. When an event occurs, the chosen killer attempts a kill with a probability given by a killing-probability matrix: this can be deterministic (always killing until capacity is exhausted) or stochastic, either constant or history-dependent via a logistic rule where the log-odds of killing change with accumulated non-lethal and lethal contacts. The model also includes a finite pool of targets; each successful kill reduces the number of remaining targets and proportionally decreases future encounter rates, so contacts become less frequent as targets are depleted. The outputs are per-killer sequences of lethal/non-lethal decisions with their event times, final contact histories, remaining capacities, and the trajectory of targets remaining over time.

---

### Bayesian Inference

---

<br>
<br>

## Results

[You can find the results folder here.](./1_Result_modelExplore/)






