# Lab 23 — FedSpeak 2.0: NLP Pipeline for Central Bank Communications

## Objective
Diagnose and repair a broken sentiment pipeline over Federal Reserve FOMC minutes, extend it with sentence-transformer embeddings, and compare the two representations on a Fed-rate-regime classification task.

## Methodology
- Pulled the `vtasca/fomc-statements-minutes` Hugging Face dataset and filtered to `Type == 'Minute'`.
- Part 1 (diagnose): identified three planted bugs: `text.split()` as a tokenizer, the Harvard General Inquirer dictionary on financial text, and a TF-IDF configuration with `min_df=1, max_df=1.0, ngram_range=(1,1)`.
- Part 2 (fix): corrected preprocessing with regex-strip plus `nltk.word_tokenize` and lemmatization; replaced the GI dictionary with Loughran-McDonald negative, positive, and uncertainty word lists; tightened TF-IDF to `min_df=5, max_df=0.85, ngram_range=(1, 2)` with a 5,000-term vocabulary cap.
- Part 3 (extend): encoded each document with `SentenceTransformer('all-MiniLM-L6-v2')` (384-d dense embeddings) using the first 2,000 characters of each minute. Clustered both the embeddings and the TF-IDF + TruncatedSVD representation with K-Means (`K=3`, `random_state=42`) and compared silhouette scores and year composition.
- Part 4 (module): packaged `preprocess_fomc`, `compute_lm_sentiment`, and `build_tfidf_matrix` into `src/fomc_sentiment.py`, each with type hints, docstrings, and an idempotent `nltk.download` guard. Exported the LM word lists as module constants for downstream reuse.
- Challenge: ran a 5-fold `TimeSeriesSplit` comparison of logistic regression AUC-ROC on TF-IDF (SVD-50) features versus sentence-transformer embeddings for predicting a tightening-regime indicator.

## Key Findings
- The corrected preprocessing eliminates non-alpha tokens entirely and drops the vocabulary size enough that "the", "committee", and "meeting" no longer dominate the TF-IDF matrix.
- Switching from Harvard GI to Loughran-McDonald removes the bulk of false-positive negative sentiment driven by neutral finance vocabulary (liability, tax, capital, debt, expense).
- Embedding-based clustering aligns better with known Fed regimes (2004–2007 tightening, 2008–2015 easing and ZLB, 2015–2019 normalization, 2022–present tightening) than TF-IDF + SVD clustering does.
- On the tightening-regime prediction task, sentence-transformer embeddings edge out the TF-IDF + SVD baseline on mean AUC, but the gap is narrow. TF-IDF remains more interpretable because the top bigrams map directly to policy concepts like "balance sheet" and "federal funds".

## Reproducing
1. `pip install -r requirements.txt`.
2. Run `lab-ch23-diagnostic.ipynb` top to bottom. First run downloads the FOMC dataset, the NLTK tokenizer, and the sentence-transformer weights.

## Stack
Python 3.13, pandas, numpy, scikit-learn, nltk, datasets (Hugging Face), sentence-transformers, matplotlib.
