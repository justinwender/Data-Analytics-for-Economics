# Hypothesis Testing & Causal Evidence Architecture

## Objective

This lab pivots from traditional **estimation** (fitting models to maximize predictive accuracy) to **falsification** (rigorously testing causal claims). Rather than asking "what's the best-fit model?", we ask "can we reject the null hypothesis that this intervention has no effect?" This shift from parameter estimation to hypothesis testing is foundational to causal inference and separates spurious correlations from genuine causal effects.

## What I Did

Using the **Lalonde (1986) dataset**, I operationalized the scientific method to adjudicate between competing narratives of causality. The analysis evaluated whether job training participation causally increased 1978 real earnings.

### Key Findings
- **Treatment Effect (ATE)**: ~$1,795 increase in real earnings for training participants
- **T-Statistic**: Statistically significant at α = 0.05
- **Conclusion**: Rejected the Null Hypothesis via Proof by Statistical Contradiction

## Technical Approach

### Parametric Testing: Welch's T-Test
- Calculated the Signal-to-Noise ratio using **Welch's T-Test** to estimate the Average Treatment Effect (ATE)
- Used Welch's variant (not Student's) to account for **unequal variances** between treatment and control groups
- Controlled for **Type I errors** by setting significance level at α = 0.05

### Non-Parametric Validation: Permutation Testing
- Conducted a **Non-Parametric Permutation Test** with 10,000 resamples to validate results
- This approach makes no distributional assumptions—critical since earnings data violates normality assumptions
- Simulated "What if treatment labels were meaningless?" to empirically derive the null distribution
- Compared permutation p-value against parametric t-test p-value for consistency

### Implementation
- **Tool**: SciPy's `stats.ttest_ind()` and `stats.permutation_test()` for robust hypothesis testing
- **Data Visualization**: KDE plots to visualize the empirical distributions of treated vs. control earnings

## Why This Matters: The Safety Valve of the Algorithmic Economy

In modern data science organizations, hypothesis testing serves as the **safety valve** preventing unfounded claims from driving business decisions. Without rigorous falsification:

- **Data Grubbing**: Running countless analyses until one shows significance (p-hacking)
- **Spurious Correlations**: Mistaking noise for signal in high-dimensional datasets
- **Opportunity Costs**: Investing in interventions that appear effective by chance alone

By enforcing pre-specified hypotheses and controlling for multiple comparisons, hypothesis testing ensures that business decisions rest on genuine causal effects, not statistical artifacts. This discipline is what separates evidence-based organizations from those vulnerable to costly false positives.

---

**Dataset**: Lalonde (1986) – National Supported Work (NSW) Demonstration Program
**Methods**: Welch's T-Test, Permutation Testing (scipy.stats)
**Language**: Python 3
