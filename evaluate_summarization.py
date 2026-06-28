"""
Summarization Evaluation Harness (per-LLM silver-standard references)
=====================================================================

Scores each summarization method's output against LLM-generated reference
summaries using ROUGE-1/2/L (lexical overlap) and an embedding-based semantic
cosine similarity (multilingual MiniLM — robust to paraphrasing).

References come from THREE LLMs (Claude, Gemini, ChatGPT), each kept as its
own reference set in reference_summaries/<llm>/<name>.txt. Every system method
is scored against each LLM's reference separately, so you can see whether the
ranking of your methods depends on which LLM you trusted.

IMPORTANT — what these numbers mean
-----------------------------------
The references are LLM-GENERATED, not human gold (see reference_summaries/
PROMPT.md). A score measures *similarity to a strong LLM's summary*, a pragmatic
proxy for quality under time constraints. Treat results as a RELATIVE comparison
across your own methods against a common reference — NOT an absolute quality
grade. ROUGE (lexical) penalises abstractive paraphrasing; read it alongside the
semantic score, which does not.

System outputs are generated ONCE per file and cached in system_outputs/ (they
do not depend on the reference), so scoring against 3 LLMs does not re-run the
slow models.

Usage:
  python evaluate_summarization.py                        # all files/methods/LLMs
  python evaluate_summarization.py --llms claude,gemini    # subset of references
  python evaluate_summarization.py --methods textrank,lsa  # subset of extractive
  python evaluate_summarization.py --no-abstractive        # skip ms-t5 (faster)
  python evaluate_summarization.py --files berita,podcast   # subset of files
  python evaluate_summarization.py --refresh               # regenerate cached outputs
"""

import os
import sys
import re
import argparse
import statistics
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Windows consoles default to cp1252 and crash on non-ASCII prints.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

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
LLMS = ["claude", "gemini", "chatgpt"]
METRICS = ("rouge1", "rouge2", "rougeL", "semantic")
PLACEHOLDER_PREFIX = "<PASTE"

# Must match the web app's defaults (app.py) so the evaluation scores the
# SAME outputs a user gets from the webpage.
NORMALIZATION = "hybrid"

# Files where Malaya's ROUGE postprocessor (postprocess=True, the webpage
# default) crashed and the abstractive step fell back to postprocess=False.
ABSTRACTIVE_FALLBACKS = []

# ── lazy semantic model ──────────────────────────────────────────────
_sent_model = None


def _semantic_model():
    global _sent_model
    if _sent_model is None:
        from sentence_transformers import SentenceTransformer
        print("[Eval] Loading multilingual sentence model for semantic score ...")
        _sent_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _sent_model


# ── text helpers ─────────────────────────────────────────────────────
def clean_for_scoring(text):
    """Strip bullet/header decoration so ROUGE compares prose, not markup."""
    text = re.sub(r"^\s*Ringkasan AI\s*", "", text)
    text = text.replace("•", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_reference(name, llm):
    """Return reference text for (file, llm), or None if missing/placeholder."""
    path = REF_DIR / llm / f"{name}.txt"
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
            print(f"  [{name}] generating extractive: {m} ...")
            outputs[m] = run_extractive(cleaned, method=m)["combined"]
            cache.write_text(outputs[m], encoding="utf-8")

    if do_abstractive:
        # Generate one abstractive output per extractive method so we can see
        # how the base extractor affects the abstractive quality.
        for m in methods:
            key = f"abstractive_{m}"
            cache = CACHE_DIR / f"{name}.abstractive_{m}.txt"
            if cache.exists() and not refresh:
                outputs[key] = cache.read_text(encoding="utf-8")
            else:
                base = outputs.get(m)
                if base is None:
                    base = run_extractive(cleaned, method=m)["combined"]
                print(f"  [{name}] generating abstractive ({m} -> ms-t5) ...")
                try:
                    outputs[key] = abstractive_summarize(
                        base, mode="beam", postprocess=True
                    )
                except Exception as e:
                    print(f"  [{name}] postprocess failed ({type(e).__name__}); "
                          f"retrying with postprocess=False")
                    outputs[key] = abstractive_summarize(
                        base, mode="beam", postprocess=False
                    )
                    ABSTRACTIVE_FALLBACKS.append(f"{name}({m})")
                cache.write_text(outputs[key], encoding="utf-8")

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


# ── report helpers ───────────────────────────────────────────────────
def _mean_table(per_file_scores, method_order):
    """per_file_scores: {name: {method: scores}} → {method: mean_scores}."""
    table = {}
    for m in method_order:
        cols = {k: [] for k in METRICS}
        for method_scores in per_file_scores.values():
            s = method_scores.get(m)
            if s:
                for k in METRICS:
                    cols[k].append(s[k])
        if cols["rouge1"]:
            table[m] = {k: statistics.mean(v) for k, v in cols.items()}
    return table


def _emit_mean_table(lines, table, method_order):
    lines.append("| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |\n")
    lines.append("|--------|---------|---------|---------|----------|\n")
    for m in method_order:
        s = table.get(m)
        if s:
            lines.append(
                f"| **{m}** | {s['rouge1']:.3f} | {s['rouge2']:.3f} | "
                f"{s['rougeL']:.3f} | {s['semantic']:.3f} |\n"
            )
    lines.append("\n")


def build_report(results, skipped, methods, do_abstractive, llms):
    """results: {llm: {name: {method: scores}}}."""
    method_order = methods + ([f"abstractive_{m}" for m in methods] if do_abstractive else [])
    lines = []
    lines.append("# Summarization Evaluation — ROUGE vs LLM References (per-LLM)\n")
    lines.append(f"Methods: {', '.join(method_order)}\n")
    lines.append(f"Reference LLMs: {', '.join(llms)} (scored as separate reference sets)\n")
    lines.append(f"Preprocessing: meeting mode, **{NORMALIZATION}** normalization "
                 f"(matches the web app default)\n")
    lines.append(
        "\n> **References are LLM-generated (silver standard), not human gold.** "
        "Scores measure similarity to each LLM's summary — a relative comparison "
        "across methods, not an absolute quality grade. ROUGE is lexical "
        "(penalises paraphrase); the semantic column is embedding-based (robust "
        "to paraphrase). See `reference_summaries/PROMPT.md` for the exact models "
        "and prompt.\n\n"
    )
    if skipped:
        sk = "; ".join(f"{llm}: {', '.join(fs)}" for llm, fs in skipped.items() if fs)
        if sk:
            lines.append(f"_Skipped (no reference filled in yet) — {sk}_\n\n")
    if ABSTRACTIVE_FALLBACKS:
        lines.append(
            f"_⚠️ Abstractive postprocess (Malaya ROUGE filter, the webpage "
            f"default) crashed on: {', '.join(ABSTRACTIVE_FALLBACKS)} — these "
            f"used `postprocess=False` instead. The live app would error on "
            f"these inputs; worth noting as a robustness limitation._\n\n"
        )
    lines.append("---\n")

    # 1. Aggregate per reference LLM
    lines.append("## Aggregate by Reference LLM (mean across scored files)\n")
    per_llm_tables = {}
    for llm in llms:
        if not results.get(llm):
            continue
        n = len(results[llm])
        lines.append(f"### Reference: {llm}  _(files scored: {n})_\n")
        table = _mean_table(results[llm], method_order)
        per_llm_tables[llm] = table
        _emit_mean_table(lines, table, method_order)

    # 2. Average across all reference LLMs (pool every file×llm comparison)
    lines.append("---\n")
    lines.append("## Average Across All Reference LLMs\n")
    pooled = {}
    for llm in llms:
        for name, method_scores in results.get(llm, {}).items():
            pooled[f"{llm}/{name}"] = method_scores
    grand = _mean_table(pooled, method_order)
    _emit_mean_table(lines, grand, method_order)

    if grand:
        lines.append("### Best method per metric (averaged over LLMs)\n")
        for k, label in [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"),
                         ("rougeL", "ROUGE-L"), ("semantic", "Semantic")]:
            winner = max(grand, key=lambda m: grand[m][k])
            lines.append(f"- **{label}:** {winner} ({grand[winner][k]:.3f})\n")
        lines.append("\n")

    # 3. Per-file detail, grouped by LLM
    lines.append("---\n")
    lines.append("## Per-File Detail\n")
    for llm in llms:
        if not results.get(llm):
            continue
        lines.append(f"### Reference: {llm}\n")
        for name, method_scores in results[llm].items():
            lines.append(f"#### {name}\n")
            lines.append("| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |\n")
            lines.append("|--------|---------|---------|---------|----------|\n")
            for m in method_order:
                s = method_scores.get(m)
                if s:
                    lines.append(
                        f"| {m} | {s['rouge1']:.3f} | {s['rouge2']:.3f} | "
                        f"{s['rougeL']:.3f} | {s['semantic']:.3f} |\n"
                    )
            lines.append("\n")

    return "".join(lines)


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization vs per-LLM references")
    parser.add_argument("--methods", default=",".join(EXTRACTIVE_METHODS),
                        help="Comma-separated extractive methods (default: all)")
    parser.add_argument("--no-abstractive", action="store_true",
                        help="Skip the ms-t5 abstractive summary")
    parser.add_argument("--files", default=None,
                        help="Comma-separated file stems to score (default: all)")
    parser.add_argument("--llms", default=",".join(LLMS),
                        help="Comma-separated reference LLMs (default: claude,gemini,chatgpt)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cached system outputs and regenerate")
    parser.add_argument("--output", default=str(OUT_DIR / "SUMMARIZATION_1.md"),
                        help="Report output path")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    do_abstractive = not args.no_abstractive
    files = [f.strip() for f in args.files.split(",")] if args.files else FILES
    llms = [l.strip() for l in args.llms.split(",") if l.strip()]

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=False)

    results = {llm: {} for llm in llms}
    skipped = {llm: [] for llm in llms}

    for name in files:
        # Which LLMs have a real reference for this file?
        refs = {llm: load_reference(name, llm) for llm in llms}
        present = {llm: r for llm, r in refs.items() if r is not None}
        for llm in llms:
            if refs[llm] is None:
                skipped[llm].append(name)

        if not present:
            print(f"[skip] {name}: no references filled in for any LLM.")
            continue

        transcript_path = INPUT_DIR / f"{name}.txt"
        if not transcript_path.exists():
            print(f"[skip] {name}: transcript not found at {transcript_path}.")
            continue

        print(f"\n=== {name} ===  (references: {', '.join(present)})")
        raw = transcript_path.read_text(encoding="utf-8")
        cleaned = preprocess_malay_transcript(raw, normalization=NORMALIZATION)
        # Generate system outputs ONCE — independent of the reference LLM.
        outputs = generate_outputs(name, cleaned, methods, do_abstractive,
                                   args.refresh)

        for llm, ref in present.items():
            method_scores = {}
            for m, hyp in outputs.items():
                method_scores[m] = score(ref, hyp, scorer)
            results[llm][name] = method_scores
            best = ", ".join(
                f"{m}:R1={s['rouge1']:.2f}/Sem={s['semantic']:.2f}"
                for m, s in method_scores.items()
            )
            print(f"  [{llm}] {best}")

    if not any(results.values()):
        print("\nNo references filled in yet — nothing scored.")
        print(f"Add LLM summaries under {REF_DIR}/<llm>/ (see PROMPT.md) and re-run.")
        return

    report = build_report(results, skipped, methods, do_abstractive, llms)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
