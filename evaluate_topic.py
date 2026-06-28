"""
Topic Modeling Evaluation — Coherence and Diversity
====================================================

Evaluates LDA, NMF, and BERTopic across 10 Malay meeting transcripts using
two intrinsic metrics (no ground truth needed):

  Coherence (c_v) — how semantically related the top keywords within each
    topic are. Computed via gensim CoherenceModel against the transcript
    itself as the reference corpus. Higher = more meaningful topics.

  Diversity — proportion of unique words across all detected topics.
    Score of 1.0 means every keyword is unique across all topics (no overlap).
    Score near 0 means topics are nearly identical (bad).

Usage:
  python evaluate_topic.py
  python evaluate_topic.py --methods lda,nmf
  python evaluate_topic.py --files berita,podcast
"""

import os
import sys
import re
import argparse
import statistics
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from preprocess import preprocess_malay_transcript
from topic_modeling import (
    perform_topic_modeling, get_malay_stopwords,
    filter_content_words, group_mwe,
)
import gensim
from gensim import corpora

INPUT_DIR = ROOT / "stt_transcription_trimmed"
OUT_DIR   = ROOT / "Testing" / "TOPIC"
NORMALIZATION = "hybrid"

FILES = [
    "berita", "culture_shock", "debat_perdana", "iv_menteri_kewangan",
    "MeetingSample", "mesyuarat_hari_sukan", "pengalaman_camping",
    "podcast", "parlimen", "sembang_pengalaman",
]

METHODS = ["lda", "nmf", "bertopic"]


# ── text helpers ─────────────────────────────────────────────────────
def tokenize_for_coherence(text):
    """
    Tokenize WITHOUT stemming so resolved (original-form) topic words
    can be looked up directly in the corpus for coherence scoring.
    """
    stopwords = get_malay_stopwords()
    text = group_mwe(text)
    docs = []
    for sent in re.split(r'[.!?]+', text):
        tokens = re.findall(r'\b[a-zA-Z_]+\b', sent.lower())
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        tokens = filter_content_words(tokens)
        if tokens:
            docs.append(tokens)
    return docs


# ── metrics ──────────────────────────────────────────────────────────
def compute_coherence(topics, texts):
    """
    c_v coherence via gensim CoherenceModel.
    topic words are resolved (original form) and texts are unstemmed
    so both use the same vocabulary.
    Returns float or None on failure.
    """
    if not topics or not texts:
        return None
    dictionary = corpora.Dictionary(texts)
    if len(dictionary) < 2:
        return None
    # Only keep words that exist in the dictionary (filters MWEs with
    # underscores and any resolved word not present in the plain corpus).
    valid = set(dictionary.token2id.keys())
    filtered = []
    for t in topics:
        words = [w.lower() for w in t.get("words", []) if w.lower() in valid]
        if len(words) >= 2:
            filtered.append(words)
    if not filtered:
        return None
    try:
        cm = gensim.models.CoherenceModel(
            topics=filtered,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        return cm.get_coherence()
    except Exception as e:
        print(f"    [coherence error] {e}")
        return None


def compute_diversity(topics):
    """
    Proportion of unique words across all topic keyword lists.
    1.0 = all keywords unique, 0.0 = all identical.
    """
    if not topics:
        return 0.0
    all_words = [w.lower() for t in topics for w in t.get("words", [])]
    if not all_words:
        return 0.0
    return round(len(set(all_words)) / len(all_words), 4)


# ── report ───────────────────────────────────────────────────────────
def _fmt(val, decimals=3):
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def build_report(results, files, methods):
    lines = []
    lines.append("# Topic Modeling Evaluation — Coherence and Diversity\n")
    lines.append(f"Files: {len(files)}  |  Methods: {', '.join(methods)}\n")
    lines.append(f"Preprocessing: meeting mode, **{NORMALIZATION}** normalization "
                 "(matches web app default)\n\n")
    lines.append(
        "> **Coherence (c_v):** measures how semantically related the top keywords "
        "within each topic are — higher is better. Computed via gensim CoherenceModel "
        "using each transcript as the reference corpus.\n"
    )
    lines.append(
        "> **Diversity:** proportion of unique words across all detected topics — "
        "higher means less repetition between topics (max 1.0).\n\n"
    )
    lines.append("---\n")

    # ── aggregate table ──────────────────────────────────────────────
    lines.append("## Aggregate Results (mean across all files)\n")
    lines.append("| Model | Avg Coherence (c_v) | Avg Diversity | Avg Topics Detected |\n")
    lines.append("|-------|---------------------|---------------|---------------------|\n")
    for m in methods:
        fr = results[m]
        avg_coh = _safe_mean([r["coherence"] for r in fr.values()])
        avg_div = _safe_mean([r["diversity"] for r in fr.values()])
        avg_n   = _safe_mean([r["n_topics"]  for r in fr.values()])
        lines.append(
            f"| **{m}** | {_fmt(avg_coh)} | {_fmt(avg_div)} | {_fmt(avg_n, 1)} |\n"
        )
    lines.append("\n")

    # ── key findings ─────────────────────────────────────────────────
    agg = {}
    for m in methods:
        fr = results[m]
        agg[m] = {
            "coherence": _safe_mean([r["coherence"] for r in fr.values()]),
            "diversity": _safe_mean([r["diversity"] for r in fr.values()]),
        }

    best_coh = max((m for m in methods if agg[m]["coherence"] is not None),
                   key=lambda m: agg[m]["coherence"], default=None)
    best_div = max(methods, key=lambda m: agg[m]["diversity"] or 0, default=None)

    lines.append("---\n")
    lines.append("## Key Findings\n")
    if best_coh:
        lines.append(
            f"- **Highest coherence:** {best_coh} ({_fmt(agg[best_coh]['coherence'])}) "
            f"— produces the most semantically related keyword groups per topic.\n"
        )
    if best_div:
        lines.append(
            f"- **Highest diversity:** {best_div} ({_fmt(agg[best_div]['diversity'])}) "
            f"— least repetition of keywords across topics.\n"
        )
    lines.append(
        "- Coherence is computed against each individual transcript (not a large "
        "external corpus), so scores reflect within-document co-occurrence. "
        "Use these numbers for relative comparison across models, not as "
        "absolute quality thresholds.\n"
    )
    lines.append(
        "- Topic detection is an unsupervised task with no ground truth. "
        "Coherence and diversity are standard intrinsic metrics used when "
        "human-annotated topic labels are unavailable.\n\n"
    )

    # ── per-file detail ───────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Per-File Detail\n")
    for name in files:
        if not any(name in results[m] for m in methods):
            continue
        lines.append(f"### {name}\n")
        lines.append("| Model | Coherence | Diversity | Topics | Keywords |\n")
        lines.append("|-------|-----------|-----------|--------|----------|\n")
        for m in methods:
            r = results[m].get(name)
            if r is None:
                continue
            kw_col = "  /  ".join(", ".join(t["words"]) for t in r["topics"])
            lines.append(
                f"| {m} | {_fmt(r['coherence'])} | {_fmt(r['diversity'])} "
                f"| {r['n_topics']} | {kw_col} |\n"
            )
        lines.append("\n")

    return "".join(lines)


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate topic modeling quality")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="Comma-separated methods (default: lda,nmf,bertopic)")
    parser.add_argument("--files", default=None,
                        help="Comma-separated file stems (default: all)")
    parser.add_argument("--output", default=str(OUT_DIR / "TOPIC_1.md"))
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    files   = [f.strip() for f in args.files.split(",")] if args.files else FILES

    results = {m: {} for m in methods}

    for name in files:
        path = INPUT_DIR / f"{name}.txt"
        if not path.exists():
            print(f"[skip] {name}: transcript not found")
            continue

        print(f"\n=== {name} ===")
        raw     = path.read_text(encoding="utf-8")
        cleaned = preprocess_malay_transcript(raw, normalization=NORMALIZATION)
        texts   = tokenize_for_coherence(cleaned)

        for m in methods:
            print(f"  [{m}] running ...")
            topics = perform_topic_modeling(cleaned, method=m)
            coh = compute_coherence(topics, texts)
            div = compute_diversity(topics)
            results[m][name] = {
                "coherence": coh,
                "diversity": div,
                "n_topics":  len(topics),
                "topics":    topics,
            }
            print(f"  [{m}] topics={len(topics)}  coherence={_fmt(coh)}  "
                  f"diversity={_fmt(div)}")

    report = build_report(results, files, methods)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
