# Dempster-Shafer Sensor Fusion

Evidence fusion for classifying air contacts from multiple conflicting sensors. A tactical control system receives radar, imaging, and electronic support measures (ESM) reports that are noisy, often coarse, and sometimes deliberately misleading. Operators and track management logic act on the confidence attached to the fused class, so that confidence needs to be well calibrated, not just accurate.

This project asks whether Dempster-Shafer evidence fusion preserves accuracy and calibration better than naive Bayesian fusion in that setting.

## Approach

Dempster-Shafer fusion, naive Bayesian fusion, Yager's combination rule, and an oracle Bayesian posterior that knows the true generative model are implemented from first principles as a tested Python package. All four run on identical simulated evidence across 36 sensor reliability and conflict conditions, with 2,000 contacts per condition and five independent seeds, and are scored on accuracy and Brier score calibration.

## Findings

- Accuracy never separates the methods. Even the oracle is at most 0.016 more accurate, so the evidence limits accuracy rather than the combination rule.
- Calibration does separate them. Dempster-Shafer improves the Brier score over plain naive Bayes by up to 0.29, but about nine tenths of that gain comes from reliability discounting.
- Once both rules are discounted, the methods form a ladder of increasing caution. The best calibrated choice depends on how much conflict the sensor model fails to anticipate. A discounted Bayesian rule is best when unmodeled conflict is rare, Dempster's rule at moderate conflict, and Yager's rule at the highest conflict.
- The crossover reproduces under every seed, with no reversal of sign in any condition.

## Status

The code, experiment results, walkthrough notebook, and case study paper will be published here after the Fall 2026 term ends.

## Built with

Python, NumPy, pandas, Matplotlib

## Context

Research project for EN.605.745 Reasoning Under Uncertainty at Johns Hopkins University (Fall 2026).
