# Lab 18 — Fraud Detection: Model Evaluation Beyond Accuracy

## Objective
Evaluate a logistic regression fraud detector using the full imbalanced-classification toolkit — confusion matrices, Precision, Recall, F1, ROC-AUC, PR-AUC, and threshold analysis — then make a capacity-constrained operating-point decision that connects the model math to the business cost of missed fraud versus false alarms.

## Methodology
- Pulled the Kaggle `mlg-ulb/creditcardfraud` dataset (284,807 European credit-card transactions; 492 confirmed fraud cases; 0.172 percent positive class) via a local file or `kagglehub` fallback.
- Standardised the `Amount` feature, dropped `Time`, and built an 80/20 stratified train/test split with `random_state=42` to preserve the fraud rate.
- Established the naive baseline: a constant "not fraud" predictor hits 99.83 percent accuracy and zero recall, which is the accuracy paradox made concrete.
- Trained `LogisticRegression(max_iter=1000)` on the training split and extracted predicted probabilities on the held-out test set.
- Computed the confusion matrix and classification report at the default threshold of 0.5, then plotted the ROC curve (AUC), the Precision-Recall curve (PR-AUC), and confusion matrices across three thresholds (0.5, 0.3, 0.1) to make the precision-recall tradeoff visible.
- Swept thresholds from 0.01 to 0.98 to identify the F1-maximising operating point.
- Part 3 (capacity constraint): found the lowest threshold that keeps the daily flagged-transaction count within a 500-investigation budget, reported recall and precision at that operating point, and translated the resulting FN and FP counts into a dollar cost under the assumption that missed fraud costs about 800 dollars and a false alarm costs about 12 dollars.
- AI expansion: shipped `streamlit_app.py`, an interactive dashboard that lets a fraud analyst drag the threshold slider and see the confusion matrix, Precision / Recall / F1, and a cost curve update live. The same dashboard adds a Random Forest comparison panel on ROC-AUC and PR-AUC, with user-tunable FN and FP costs so the business tradeoff is explicit.

## Key Findings
- The accuracy paradox is stark: the naive baseline beats the logistic regression on accuracy alone while catching zero frauds. Accuracy is a useless metric here.
- Logistic regression achieves a strong ROC-AUC on the test set, but its PR-AUC is the more honest number on a 0.17 percent positive-rate problem, and it is comfortably above the random-classifier baseline (which equals the prevalence).
- The F1-maximising threshold is meaningfully below 0.5 — consistent with lecture, the default threshold is an arbitrary convention, not a law of nature.
- At a 500-investigation daily capacity the model catches roughly seven out of every ten frauds while keeping precision much higher than at τ = 0.5. The cost-minimising threshold is *lower* than that capacity-constrained threshold once FN and FP are priced asymmetrically, so the real bottleneck is analyst capacity, not the model.
- Random Forest offers a modest PR-AUC bump over logistic regression on this dataset — enough to matter for production, but not so big that it changes the qualitative story about thresholds and cost.

## Reproducing
1. Drop `creditcard.csv` into the Lab18 folder (or let `kagglehub` fetch it on first run).
2. `pip install -r requirements.txt`.
3. Run `lab_18_model_evaluation.ipynb` top to bottom.
4. For the dashboard: from the `Lab18/` folder run `streamlit run streamlit_app.py`.

## Stack
Python 3.13, pandas, numpy, scikit-learn, matplotlib, seaborn, kagglehub, streamlit.
