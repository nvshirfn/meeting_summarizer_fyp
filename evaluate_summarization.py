"""
Summarization Evaluation Harness (LLM silver-standard references)
=================================================================

Scores each summarization method's output against an LLM-generated reference
summary using ROUGE-1/2/L (lexical overlap) and an embedding-based semantic
cosine similarity (multilingual MiniLM — robust to paraphrasing).

IMPORTANT — what these numbers mean
-----------------------------------
The references in reference_summaries/ are LLM-GENERATED, not human gold
(see reference_summaries/PROMPT.md). So a score here measures *similarity to
a strong LLM's summary*, a pragmatic proxy for quality under time constraints.
Treat the results as a RELATIVE comparison across your own methods against a
common reference — NOT an absolute quality grade. ROUGE (lexical) penalises
abstractive paraphrasing; read it alongside the semantic score, which does not.

Outputs per method:
  - ROUGE-1 / ROUGE-2 / ROUGE-L  (F-measure)
  - Semantic  (cosine of multilingual sentence embeddings)

System outputs are cached in system_outputs/ so re-scoring after editing a
reference does not re-run the (slow) models.

Usage:
  python evaluate_summarization.py                       # all files, all methods
  python evaluate_summarization.py --methods textrank,lsa # subset of extractive
  python evaluate_summarization.py --no-abstractive       # skip ms-t5 (faster)
  python evaluate_summarization.py --files berita,podcast  # subset of files
  python evaluate_summarization.py --refresh              # ignore cache, regenerate
"""

import os
import sys
import re
import argparse
import statistics
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from preprocess import preprocess_malay_transcript
from extractive import run_extractive
from abstractive import abstractive_summarize

# ── paths ────────────────────────────────────────────────────────────
INPUT_DIR = ROOT / "stt_transcription_trimmed"
REF_DIR   = ROOT / "reference_summaries"
CACHE_DIR = ROOT / "system_outputs"
OUT_DIR   = ROOT / "Testing" / "SUMMARIZATION"

# Same 10 files as the sentiment benchmark, for a consistent dataset.
FILES = [
    "berita", "culture_shock", "debat_perdana", "iv_menteri_kewangan",
    "MeetingSample", "mesyuarat_hari_sukan", "pengalaman_camping",
    "podcast", "parlimen", "sembang_pengalaman",
]

EXTRACTIVE_METHODS = ["textrank", "lsa", "electra"]
PLACEHOLDER_PREFIX = "<PASTE"

# ── lazy semantic model ──────────────────────────────────────────────
_sent_model = None


def _semantic_model():
    global _sent_model
    if _sent_model is None:
        from sentence_transformers import SentenceTransformer
        print("[Eval] Loading multilingual sentence model for semantic score …")
        _sent_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _sent_model


# ── text helpers ─────────────────────────────────────────────────────
def clean_for_scoring(text):
    """Strip bullet/header decoration so ROUGE compares prose, not markup."""
    text = re.sub(r"^\s*Ringkasan AI\s*", "", text)
    text = text.replace("•", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_reference(name):
    """Return reference text, or None if the file is missing/empty/placeholder."""
    path = REF_DIR / f"{name}.txt"
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    if not content or content.startswith(PLACEHOLDER_PREFIX):
        return None
    return content


# ── system-output generation (with caching) ──────────────────────────
def generate_outputs(name, cleaned, methods, do_abstractive, refresh):
    outputs = {}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for m in methods:
        cache = CACHE_DIR / f"{name}.{m}.txt"
        if cache.exists() and not refresh:
            outputs[m] = cache.read_text(encoding="utf-8")
        else:
            print(f"  [{name}] generating extractive: {m} …")
            outputs[m] = run_extractive(cleaned, method=m)["combined"]
            cache.write_text(outputs[m], encoding="utf-8")

    if do_abstractive:
        cache = CACHE_DIR / f"{name}.abstractive.txt"
        if cache.exists() and not refresh:
            outputs["abstractive"] = cache.read_text(encoding="utf-8")
        else:
            # App default path: textrank extractive → ms-t5 abstractive.
            base = outputs.get("textrank")
            if base is None:
                base = run_extractive(cleaned, method="textrank")["combined"]
            print(f"  [{name}] generating abstractive (textrank → ms-t5) …")
            outputs["abstractive"] = abstractive_summarize(
                base, mode="beam", postprocess=True
            )
            cache.write_text(outputs["abstractive"], encoding="utf-8")

    return outputs


# ── scoring ──────────────────────────────────────────────────────────
def score(reference, hypothesis, scorer):
    ref = clean_for_scoring(reference)
    hyp = clean_for_scoring(hypothesis)
    r = scorer.score(ref, hyp)

    from sentence_transformers import util
    emb = _semantic_model().encode([ref, hyp], convert_to_tensor=True,
                                    show_progress_bar=False)
    semantic = float(util.cos_sim(emb[0], emb[1]))

    return {
        "rouge1": r["rouge1"].fmeasure,
        "rouge2": r["rouge2"].fmeasure,
        "rougeL": r["rougeL"].fmeasure,
        "semantic": semantic,
    }


# ── report ───────────────────────────────────────────────────────────
def build_report(results, skipped, methods, do_abstractive):
    method_order = methods + (["abstractive"] if do_abstractive else [])
    lines = []
    lines.append("# Summarization Evaluation — ROUGE vs LLM Reference\n")
    lines.append(
        f"Files scored: {len(results)}  |  "
        f"Methods: {', '.join(method_order)}\n"
    )
    lines.append(
        "\n> **References are LLM-generated (silver standard), not human gold.** "
        "Scores measure similarity to a strong LLM's summary — a relative "
        "comparison across methods, not an absolute quality grade. ROUGE is "
        "lexical (penalises paraphrase); the semantic column is embedding-based "
        "(robust to paraphrase). See `reference_summaries/PROMPT.md` for the "
        "exact model and prompt used.\n\n"
    )
    if skipped:
        lines.append(
            f"_Skipped (no reference filled in yet): {', '.join(skipped)}_\n\n"
        )
    lines.append("---\n")

    # Per-file tables
    lines.append("## Per-File Results\n")
    for name, method_scores in results.items():
        lines.append(f"### {name}\n")
        lines.append("| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |\n")
        lines.append("|--------|---------|---------|---------|----------|\n")
        for m in method_order:
            s = method_scores.get(m)
            if not s:
                continue
            lines.append(
                f"| {m} | {s['rouge1']:.3f} | {s['rouge2']:.3f} | "
                f"{s['rougeL']:.3f} | {s['semantic']:.3f} |\n"
            )
        lines.append("\n")

    # Aggregate (mean across files)
    lines.append("---\n")
    lines.append("## Aggregate (mean across scored files)\n")
    lines.append("| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |\n")
    lines.append("|--------|---------|---------|---------|----------|\n")
    best = {}
    for m in method_order:
        cols = {k: [] for k in ("rouge1", "rouge2", "rougeL", "semantic")}
        for method_scores in results.values():
            s = method_scores.get(m)
            if s:
                for k in cols:
                    cols[k].append(s[k])
        if not cols["rouge1"]:
            continue
        means = {k: statistics.mean(v) for k, v in cols.items()}
        best[m] = means
        lines.append(
            f"| **{m}** | {means['rouge1']:.3f} | {means['rouge2']:.3f} | "
            f"{means['rougeL']:.3f} | {means['semantic']:.3f} |\n"
        )

    # Winners
    if best:
        lines.append("\n### Best method per metric\n")
        for k, label in [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"),
                         ("rougeL", "ROUGE-L"), ("semantic", "Semantic")]:
            winner = max(best, key=lambda m: best[m][k])
            lines.append(f"- **{label}:** {winner} ({best[winner][k]:.3f})\n")

    return "".join(lines)


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization vs LLM references")
    parser.add_argument("--methods", default=",".join(EXTRACTIVE_METHODS),
                        help="Comma-separated extractive methods (default: all)")
    parser.add_argument("--no-abstractive", action="store_true",
                        help="Skip the ms-t5 abstractive summary")
    parser.add_argument("--files", default=None,
                        help="Comma-separated file stems to score (default: all)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cached system outputs and regenerate")
    parser.add_argument("--output", default=str(OUT_DIR / "SUMMARIZATION_1.md"),
                        help="Report output path")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    do_abstractive = not args.no_abstractive
    files = [f.strip() for f in args.files.split(",")] if args.files else FILES

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=False)

    results = {}
    skipped = []

    for name in files:
        ref = load_reference(name)
        if ref is None:
            skipped.append(name)
            print(f"[skip] {name}: no reference filled in yet.")
            continue

        transcript_path = INPUT_DIR / f"{name}.txt"
        if not transcript_path.exists():
            skipped.append(name)
            print(f"[skip] {name}: transcript not found at {transcript_path}.")
            continue

        print(f"\n=== {name} ===")
        raw = transcript_path.read_text(encoding="utf-8")
        cleaned = preprocess_malay_transcript(raw, mode="meeting",
                                              normalization="dictionary")
        outputs = generate_outputs(name, cleaned, methods, do_abstractive,
                                   args.refresh)

        method_scores = {}
        for m, hyp in outputs.items():
            method_scores[m] = score(ref, hyp, scorer)
            s = method_scores[m]
            print(f"  {m:>11}:  R1={s['rouge1']:.3f}  R2={s['rouge2']:.3f}  "
                  f"RL={s['rougeL']:.3f}  Sem={s['semantic']:.3f}")
        results[name] = method_scores

    if not results:
        print("\nNo references filled in yet — nothing scored.")
        print(f"Add LLM summaries to {REF_DIR}/ (see PROMPT.md) and re-run.")
        return

    report = build_report(results, skipped, methods, do_abstractive)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
