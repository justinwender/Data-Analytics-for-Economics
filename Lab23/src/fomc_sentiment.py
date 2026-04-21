"""fomc_sentiment.py - FOMC text preprocessing, LM sentiment, and TF-IDF utilities.

Course: ECON 5200, Lab 23.

The Loughran-McDonald (LM) dictionary here is a compact version suitable for
lab work. For publication-quality finance research, pull the full word lists
from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ and drop them in.
"""
from __future__ import annotations

import re
from typing import Iterable

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def _ensure_nltk() -> None:
    for package in ("punkt_tab", "stopwords", "wordnet"):
        try:
            nltk.data.find(f"tokenizers/{package}" if package.startswith("punkt") else f"corpora/{package}")
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk()
_STOP = set(stopwords.words("english"))
_LEM = WordNetLemmatizer()
_NON_ALPHA = re.compile(r"[^a-z\s]")


# Compact LM word lists (enough to separate LM from Harvard-GI in the lab)
LM_NEGATIVE: frozenset[str] = frozenset({
    "adverse", "adversely", "against", "concern", "concerned", "concerns",
    "decline", "declined", "declining", "decrease", "decreased", "deficit",
    "deteriorate", "deteriorated", "deteriorating", "difficult", "difficulty",
    "downturn", "fail", "failure", "falling", "loss", "losses", "negative",
    "negatively", "recession", "recessionary", "risk", "risks", "risky",
    "severe", "severely", "slowdown", "sluggish", "stress", "stressed",
    "threat", "threatened", "threatening", "weak", "weakened", "weakening",
    "weakness", "worse", "worsening", "unfavorable", "unfavorably",
})
LM_POSITIVE: frozenset[str] = frozenset({
    "achieve", "achieved", "achieving", "advance", "advanced", "advancing",
    "benefit", "benefits", "boost", "boosted", "confident", "confidence",
    "effective", "effectively", "efficient", "efficiently", "favorable",
    "favorably", "gain", "gains", "good", "great", "growth", "improve",
    "improved", "improvement", "improvements", "increase", "increased",
    "increasing", "positive", "positively", "profitable", "progress",
    "strong", "stronger", "strengthen", "strengthened", "strengthening",
    "success", "successful", "successfully", "upturn",
})
LM_UNCERTAINTY: frozenset[str] = frozenset({
    "approximate", "approximated", "approximately", "assume", "assumed",
    "assuming", "assumption", "assumptions", "believe", "believed",
    "believes", "could", "depend", "dependent", "depending", "may",
    "might", "possible", "possibility", "possibly", "probable", "probably",
    "risk", "risks", "tentative", "tentatively", "uncertain", "uncertainty",
    "variable", "volatile", "volatility", "vague",
})


def preprocess_fomc(text: str) -> str:
    """Lowercase, strip non-alpha characters, tokenize, drop stopwords, lemmatize.

    Returns a space-joined string so the output can be dropped straight into a
    ``TfidfVectorizer`` via ``token_pattern`` default splitting.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    lowered = text.lower()
    cleaned = _NON_ALPHA.sub(" ", lowered)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in _STOP and len(t) > 2]
    return " ".join(_LEM.lemmatize(t) for t in tokens)


def compute_lm_sentiment(text: str) -> dict:
    """Compute Loughran-McDonald sentiment and uncertainty scores.

    ``net_sentiment = (pos - neg) / total_words`` is the standard LM tone
    measure in the accounting / central-bank communications literature.
    ``uncertainty = unc_count / total_words`` tracks linguistic hedging.

    Returns:
        Dict with ``net_sentiment``, ``uncertainty``, ``neg_count``,
        ``pos_count``, ``unc_count``, ``total_words``.
    """
    cleaned = preprocess_fomc(text)
    tokens = cleaned.split()
    total = len(tokens) or 1
    neg = sum(1 for t in tokens if t in LM_NEGATIVE)
    pos = sum(1 for t in tokens if t in LM_POSITIVE)
    unc = sum(1 for t in tokens if t in LM_UNCERTAINTY)
    return {
        "net_sentiment": (pos - neg) / total,
        "uncertainty": unc / total,
        "neg_count": neg,
        "pos_count": pos,
        "unc_count": unc,
        "total_words": total,
    }


def build_tfidf_matrix(
    texts: Iterable[str],
    min_df: int = 5,
    max_df: float = 0.85,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[object, np.ndarray, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer on preprocessed texts.

    Returns:
        Tuple of (sparse matrix, feature-names array, fitted vectorizer).
    """
    vectorizer = TfidfVectorizer(
        min_df=min_df, max_df=max_df,
        max_features=max_features, ngram_range=ngram_range,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out(), vectorizer


if __name__ == "__main__":
    sample = (
        "The Committee sees downside risks to growth remaining elevated; "
        "however, recent gains in employment and strengthening manufacturing "
        "activity suggest the recovery is on a firmer footing. Uncertainty "
        "remains around the pace of disinflation."
    )
    print("Preprocessed:", preprocess_fomc(sample)[:120], "...")
    print("LM sentiment:", compute_lm_sentiment(sample))
    matrix, names, vec = build_tfidf_matrix([sample, sample.replace("growth", "recession")])
    print("Vocabulary size:", len(names))
    print("Matrix shape:", matrix.shape)
