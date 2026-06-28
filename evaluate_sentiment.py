"""
Sentiment Analysis Evaluation — Human Labels vs Model Predictions
=================================================================

Compares BERT, Multinomial NB, and Lexicon predictions against
manually assigned ground-truth labels for 10 Malay transcripts.

Preprocessing uses hybrid normalization to match the web app default.

Usage:
  python evaluate_sentiment.py
"""

import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from preprocess import preprocess_malay_transcript
from sentiment_analysis import analyze_sentiment

INPUT_DIR = ROOT / "stt_transcription_trimmed"
OUT_DIR   = ROOT / "Testing" / "SENTIMENT"
NORMALIZATION = "hybrid"

# ── Human ground-truth labels ────────────────────────────────────────
LABELS = {
    "berita":               "negative",
    "culture_shock":        "positive",
    "debat_perdana":        "negative",
    "iv_menteri_kewangan":  "positive",
    "MeetingSample":        "positive",
    "mesyuarat_hari_sukan": "neutral",
    "pengalaman_camping":   "negative",
    "podcast":              "negative",
    "parlimen":             "neutral",
    "sembang_pengalaman":   "positive",
}

METHODS = ["bert", "multinomial_nb", "lexicon"]


def main():
    predictions = {m: {} for m in METHODS}

    for name, true_label in LABELS.items():
        path = INPUT_DIR / f"{name}.txt"
        if not path.exists():
            print(f"[skip] {name}: transcript not found")
            continue

        print(f"\n=== {name} (label: {true_label}) ===")
        raw     = path.read_text(encoding="utf-8")
        cleaned = preprocess_malay_transcript(raw, normalization=NORMALIZATION)

        for method in METHODS:
            result = analyze_sentiment(cleaned, method=method)
            pred   = result["sentiment"]
            predictions[method][name] = pred
            correct = "correct" if pred == true_label else "WRONG"
            print(f"  [{method}] {pred} ({correct})")

    report = build_report(predictions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "SENTIMENT_5.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {out_path}")


def build_report(predictions):
    lines = []
    lines.append("# Sentiment Analysis Evaluation — Human Labels vs Model Predictions\n")
    lines.append(f"Files: {len(LABELS)}  |  Models: {', '.join(METHODS)}\n")
    lines.append(f"Preprocessing: meeting mode, **{NORMALIZATION}** normalization "
                 "(matches web app default)\n")
    lines.append("Ground truth: manually assigned document-level labels "
                 "(positive / negative / neutral) by the system developer "
                 "based on transcript content.\n\n")

    # ── per-file comparison table ─────────────────────────────────────
    lines.append("## Per-File Results\n")
    lines.append("| File | Ground Truth | BERT | Multinomial NB | Lexicon |\n")
    lines.append("|------|-------------|------|----------------|--------|\n")

    correct_counts = {m: 0 for m in METHODS}

    for name, true_label in LABELS.items():
        row = f"| {name} | **{true_label}** |"
        for method in METHODS:
            pred = predictions[method].get(name, "N/A")
            if pred == true_label:
                correct_counts[method] += 1
                row += f" {pred} |"
            else:
                row += f" ~~{pred}~~ |"
        lines.append(row + "\n")

    lines.append("\n")

    # ── accuracy summary ──────────────────────────────────────────────
    n = len(LABELS)
    lines.append("## Accuracy Summary\n")
    lines.append("| Model | Correct | Accuracy |\n")
    lines.append("|-------|---------|----------|\n")
    for method in METHODS:
        c = correct_counts[method]
        lines.append(f"| **{method}** | {c} / {n} | {c/n:.0%} |\n")
    lines.append("\n")

    # ── per-class breakdown ───────────────────────────────────────────
    lines.append("## Per-Class Breakdown\n")
    classes = ["positive", "negative", "neutral"]
    for method in METHODS:
        lines.append(f"### {method}\n")
        lines.append("| Class | True Positives | Total in GT | Recall |\n")
        lines.append("|-------|---------------|-------------|--------|\n")
        for cls in classes:
            gt_count   = sum(1 for v in LABELS.values() if v == cls)
            tp         = sum(1 for name, v in LABELS.items()
                             if v == cls and predictions[method].get(name) == cls)
            recall     = tp / gt_count if gt_count > 0 else 0.0
            lines.append(f"| {cls} | {tp} | {gt_count} | {recall:.0%} |\n")
        lines.append("\n")

    # ── key findings ──────────────────────────────────────────────────
    best = max(METHODS, key=lambda m: correct_counts[m])
    lines.append("## Key Findings\n")
    lines.append(
        f"- **Best overall accuracy:** {best} "
        f"({correct_counts[best]}/{n} = {correct_counts[best]/n:.0%})\n"
    )
    lines.append(
        "- Ground truth labels are document-level (one label per transcript), "
        "which matches the overall sentiment output shown to users in the web app.\n"
    )
    lines.append(
        f"- Evaluation set is small ({n} files) — results indicate relative "
        "model performance on this dataset and should not be generalised. "
        "Acknowledged as a limitation in the thesis.\n"
    )

    return "".join(lines)


if __name__ == "__main__":
    main()
