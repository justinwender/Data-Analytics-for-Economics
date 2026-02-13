# Audit 02: Deconstructing Statistical Lies  
**A Forensic Analysis of Metrics Under Skew, Selection, and Conditioning**

## Overview

This project examines how statistically valid metrics can produce misleading conclusions when evaluated without attention to distributional structure, base rates, or selection effects.

Using simulated case studies across infrastructure, machine learning, experimentation, and financial markets, this audit demonstrates how averages and aggregate statistics can obscure operational risk, inflate performance claims, and distort inference.

The objective is not to “debunk” metrics, but to interrogate the assumptions that give them meaning.

---

## Executive Summary

Across four simulated environments, we identify three recurring failure modes:

1. **Distributional Skew** – Means conceal tail risk in heavy-tailed systems.
2. **Base Rate Neglect** – Model accuracy collapses in low-prevalence settings.
3. **Selection Bias** – Performance appears inflated when non-survivors are excluded.

The unifying theme:  
**Metrics are conditional objects. Without the correct denominator, distribution, or prior, they mislead.**

---

# Case Study 1: Latency Metrics Under Heavy-Tailed Distributions

### Claim
A cloud infrastructure firm reports a mean latency of 35ms.

### Analytical Approach
We simulated 1,000 request logs with:
- 98% normal traffic (20–50ms)
- 2% extreme spikes (1000–5000ms)

This structure approximates a heavy-tailed (Pareto-like) system common in distributed environments.

### Findings

- The **mean** was highly sensitive to rare latency spikes.
- The **median** remained stable and representative of typical user experience.
- The **Median Absolute Deviation (MAD)** confirmed robustness.
- Tail metrics (P95/P99) captured operational risk far better than the mean.

### Interpretation

In skewed systems, the mean is a poor summary statistic.  
Operational stability depends on tail behavior, not central tendency.

**Implication:** Reporting only mean latency constitutes incomplete disclosure in a heavy-tailed environment.

---

# Case Study 2: False Positives in Low-Prevalence Settings

### Claim
An AI plagiarism detector reports:
- Sensitivity: 98%
- Specificity: 98%

### Analytical Approach
Using Bayes’ Theorem, we evaluated the posterior probability:

P(Cheater | Flagged)

under three base-rate environments:

| Scenario | Base Rate | Posterior Probability |
|-----------|-----------|------------------------|
| Bootcamp | 50% | ~98% |
| Econ Course | 5% | ~72% |
| Honors Seminar | 0.1% | ~4.7% |

### Findings

When cheating is rare (0.1% prevalence):

- 95% of flagged students are statistically innocent.

The model’s performance metrics did not change.  
The environment did.

### Interpretation

Accuracy metrics are conditional on prevalence.  
High sensitivity and specificity do not guarantee high positive predictive value.

**Implication:** Model performance must be evaluated at the relevant base rate. Reporting aggregate accuracy without contextual prevalence is incomplete.

---

# Case Study 3: Experimental Integrity in A/B Testing

### Claim
A fintech firm ran a 100,000-user A/B test with a 50/50 split and reports a significant uplift.

### Observation
Observed allocation:
- Control: 50,250
- Treatment: 49,750

A Chi-Square Goodness-of-Fit test was conducted to assess allocation randomness.

### Result
The imbalance was not statistically significant at the 5% level.

### Interpretation

While the allocation deviation falls within expected random fluctuation, operational explanations (e.g., server crashes, exposure filtering, assignment bias) remain plausible.

**Implication:** Statistical insignificance does not eliminate the need for engineering review. Randomization integrity is both a statistical and operational question.

---

# Case Study 4: Survivorship Bias in Crypto Markets

### Simulation
10,000 token launches were simulated using a power-law distribution for peak market capitalization.

Two datasets were constructed:
- **Full Population (The Graveyard)**
- **Top 1% Survivors**

### Findings

- The mean market cap of survivors was orders of magnitude higher than the full population.
- Excluding failed tokens dramatically inflated perceived performance.

### Interpretation

Survivorship bias fundamentally alters distributional conclusions.  
Studying only surviving assets produces systematically overstated returns.

**Implication:** Financial analysis must reconstruct the full denominator to avoid inflated inference.

---

# Cross-Case Synthesis

Across infrastructure systems, AI classification, experimentation, and financial markets, the same structural weaknesses recur:

| Failure Mode | Mechanism | Consequence |
|--------------|------------|-------------|
| Skew | Heavy-tailed distributions | Mean understates risk |
| Base Rate Neglect | Ignoring prevalence | False positive inflation |
| Selection Bias | Filtering denominator | Performance overstatement |
| Conditioning | Reporting partial metrics | Misleading inference |

The metrics themselves were not incorrect.

The conditioning was.

---

# Methodology

- Synthetic data generated using NumPy distributions (uniform and Pareto)
- Bayesian posterior calculations performed analytically
- Chi-Square tests implemented manually
- Distributional visualizations via Matplotlib
- Robust dispersion measured using Median Absolute Deviation

All simulations are reproducible and included in the repository.

---

# Conclusion

Statistical misrepresentation rarely arises from fabricated numbers.

It arises from:
- Reporting central tendency in skewed systems
- Ignoring base rates in classification
- Omitting denominators in financial analysis
- Equating statistical insignificance with operational validity

The central discipline of economic consulting is not computing metrics.  
It is interrogating the assumptions that make those metrics meaningful.

