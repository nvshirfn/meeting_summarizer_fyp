"""
Sentiment benchmark: run all 10 stt_transcription_trimmed files through
bert, multinomial_nb, and lexicon models, then write a report to
Testing/SENTIMENT/SENTIMENT_3.md.

Text is preprocessed with preprocess_malay_transcript (meeting mode,
dictionary normalization) before sentiment analysis to remove fillers,
normalize slang, and reduce noise from raw STT transcripts.

BERT now uses word-based chunking (300 words, 50-word overlap) instead
of punctuation-based sentence splitting, to avoid silent truncation on
long STT transcripts.
"""

import os
import sys
import time
from pathlib import Path

# ── ensure project root is on the path ─────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from sentiment_analysis import analyze_sentiment
from preprocess import preprocess_malay_transcript

# ── files ───────────────────────────────────────────────────────────
INPUT_DIR = ROOT / "stt_transcription_trimmed"
FILES = [
    "berita.txt",
    "culture_shock.txt",
    "debat_perdana.txt",
    "iv_menteri_kewangan.txt",
    "MeetingSample.txt",
    "mesyuarat_hari_sukan.txt",
    "pengalaman_camping.txt",
    "podcast.txt",
    "parlimen.txt",
    "sembang_pengalaman.txt",
]
MODELS = ["bert", "multinomial_nb", "lexicon"]

OUTPUT_DIR = ROOT / "Testing" / "SENTIMENT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "SENTIMENT_4.md"
PREV_FILE  = OUTPUT_DIR / "SENTIMENT_3.md"

# ── helpers ─────────────────────────────────────────────────────────

def load_text(filename):
    path = INPUT_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    cleaned = preprocess_malay_transcript(raw, mode="meeting", normalization="dictionary")
    return raw, cleaned


def fmt_label(label):
    icons = {"positive": "✅ POSITIVE", "negative": "❌ NEGATIVE", "neutral": "⚪ NEUTRAL"}
    return icons.get(label, label.upper())


# ── main ─────────────────────────────────────────────────────────────

def run():
    results = {}  # results[filename][model] = result_dict

    for filename in FILES:
        print(f"\n{'='*60}")
        print(f"File: {filename}")
        print(f"{'='*60}")
        raw, text = load_text(filename)
        print(f"  Raw: {len(raw)} chars  ->  Preprocessed: {len(text)} chars")
        results[filename] = {}
        results[filename]["_raw_len"] = len(raw)
        results[filename]["_clean_len"] = len(text)

        for model in MODELS:
            print(f"  Running [{model}] ...", end=" ", flush=True)
            t0 = time.time()
            result = analyze_sentiment(text, method=model)
            elapsed = time.time() - t0
            result["_elapsed"] = round(elapsed, 2)
            results[filename][model] = result
            print(f"-> {result['sentiment'].upper()}  ({elapsed:.1f}s)")

    # ── build report ────────────────────────────────────────────────
    lines = []
    lines.append("# Sentiment Analysis Benchmark — All 10 Files × 3 Models\n")
    lines.append(f"Files processed: {len(FILES)}  |  Models: {', '.join(MODELS)}\n")
    lines.append("**Preprocessing:** `preprocess_malay_transcript(mode='meeting', normalization='dictionary')` applied before sentiment analysis.\n\n")
    lines.append("**BERT chunking:** word-based chunks of 300 words (50-word overlap) — replaces punctuation-based sentence split to avoid silent truncation.\n\n")
    lines.append("---\n")

    # Per-file detailed results
    lines.append("## Per-File Results\n")
    for filename in FILES:
        name = Path(filename).stem
        raw_len = results[filename]["_raw_len"]
        clean_len = results[filename]["_clean_len"]
        reduction = round((1 - clean_len / raw_len) * 100, 1)
        lines.append(f"### {name}\n")
        lines.append(f"_Preprocessing: {raw_len} chars → {clean_len} chars ({reduction}% reduction)_\n\n")
        lines.append("| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |\n")
        lines.append("|-------|-----------|----------|----------|---------|------------|------|\n")
        for model in MODELS:
            r = results[filename][model]
            label = r["sentiment"]
            if model == "lexicon":
                pos = r.get("positive_score", 0)
                neg = r.get("negative_score", 0)
                neu = "—"
                conf = "—"
                row = f"| {model} | **{label}** | {pos} words | {neg} words | {neu} | {conf} | {r['_elapsed']}s |\n"
            else:
                pos = f"{r['positive_score']:.1%}"
                neg = f"{r['negative_score']:.1%}"
                neu = f"{r['neutral_score']:.1%}"
                conf = f"{r['confidence']:.1%}"
                row = f"| {model} | **{label}** | {pos} | {neg} | {neu} | {conf} | {r['_elapsed']}s |\n"
            lines.append(row)
        lines.append("\n")

    # ── Cross-model agreement summary ────────────────────────────────
    lines.append("---\n")
    lines.append("## Cross-Model Agreement\n")
    lines.append("Shows whether all 3 models agree on the sentiment for each file.\n\n")
    lines.append("| File | BERT | Multinomial NB | Lexicon | Agreement |\n")
    lines.append("|------|------|----------------|---------|----------|\n")

    agree_counts = {m: 0 for m in MODELS}
    total_agree = 0

    for filename in FILES:
        name = Path(filename).stem
        r = results[filename]
        bert_label = r["bert"]["sentiment"]
        nb_label = r["multinomial_nb"]["sentiment"]
        lex_label = r["lexicon"]["sentiment"]
        labels = {bert_label, nb_label, lex_label}
        agreed = "✅ Full" if len(labels) == 1 else ("⚠️ Partial" if len(labels) == 2 else "❌ None")
        if len(labels) == 1:
            total_agree += 1
        lines.append(f"| {name} | {bert_label} | {nb_label} | {lex_label} | {agreed} |\n")

    lines.append(f"\n**Full agreement rate:** {total_agree}/{len(FILES)} files\n\n")

    # ── Per-model confidence summary ─────────────────────────────────
    lines.append("---\n")
    lines.append("## Model Performance Summary\n\n")

    # BERT avg confidence
    bert_confs = [results[f]["bert"]["confidence"] for f in FILES]
    nb_confs = [results[f]["multinomial_nb"]["confidence"] for f in FILES]
    bert_avg = sum(bert_confs) / len(bert_confs)
    nb_avg = sum(nb_confs) / len(nb_confs)

    # Lexicon: compute ratio metric (pos/(pos+neg+1))
    lex_ratios = []
    for f in FILES:
        r = results[f]["lexicon"]
        pos = r.get("positive_score", 0)
        neg = r.get("negative_score", 0)
        total = pos + neg
        lex_ratios.append(pos / total if total > 0 else 0.5)

    # Sentiment distribution per model
    lines.append("### Sentiment Distribution\n")
    lines.append("| Model | Positive | Negative | Neutral | Avg Confidence |\n")
    lines.append("|-------|----------|----------|---------|----------------|\n")
    for model in MODELS:
        labels = [results[f][model]["sentiment"] for f in FILES]
        pos_n = labels.count("positive")
        neg_n = labels.count("negative")
        neu_n = labels.count("neutral")
        if model == "bert":
            avg_conf = f"{bert_avg:.1%}"
        elif model == "multinomial_nb":
            avg_conf = f"{nb_avg:.1%}"
        else:
            avg_conf = "N/A (word count)"
        lines.append(f"| {model} | {pos_n} | {neg_n} | {neu_n} | {avg_conf} |\n")

    # ── Best model analysis ──────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Best Model Analysis\n\n")

    lines.append("### Scoring Criteria\n")
    lines.append("Since no ground-truth labels exist, models are compared on:\n")
    lines.append("1. **Average confidence** — higher means more decisive predictions (BERT, NB only)\n")
    lines.append("2. **Sentiment plausibility** — does the predicted sentiment match expected content type?\n")
    lines.append("3. **Cross-model agreement** — agreement with other models as a proxy for reliability\n\n")

    # Plausibility check: expected sentiments per file
    expected = {
        "berita.txt": "neutral",
        "culture_shock.txt": "negative",
        "debat_perdana.txt": "negative",
        "iv_menteri_kewangan.txt": "neutral",
        "MeetingSample.txt": "neutral",
        "mesyuarat_hari_sukan.txt": "positive",
        "pengalaman_camping.txt": "positive",
        "podcast.txt": "neutral",
        "parlimen.txt": "negative",
        "sembang_pengalaman.txt": "positive",
    }

    lines.append("### Plausibility Score (vs Expected Sentiment)\n")
    lines.append("Expected sentiments are rough heuristics based on content type.\n\n")
    lines.append("| File | Expected | BERT | NB | Lexicon |\n")
    lines.append("|------|----------|------|----|---------|\n")

    plaus = {m: 0 for m in MODELS}
    for filename in FILES:
        name = Path(filename).stem
        exp = expected.get(filename, "?")
        row_parts = [name, exp]
        for model in MODELS:
            pred = results[filename][model]["sentiment"]
            match = "✅" if pred == exp else "❌"
            row_parts.append(f"{pred} {match}")
            if pred == exp:
                plaus[model] += 1
        lines.append("| " + " | ".join(row_parts) + " |\n")

    lines.append(f"\n**Plausibility matches out of {len(FILES)}:**\n")
    for model in MODELS:
        lines.append(f"- **{model}**: {plaus[model]}/{len(FILES)}\n")

    # Overall ranking
    lines.append("\n### Overall Ranking\n\n")

    scores = {}
    scores["bert"] = {"confidence": bert_avg, "plausibility": plaus["bert"] / len(FILES)}
    scores["multinomial_nb"] = {"confidence": nb_avg, "plausibility": plaus["multinomial_nb"] / len(FILES)}
    scores["lexicon"] = {"confidence": None, "plausibility": plaus["lexicon"] / len(FILES)}

    # Composite: weight confidence 50%, plausibility 50% (lexicon gets only plausibility)
    def composite(m):
        if scores[m]["confidence"] is not None:
            return 0.5 * scores[m]["confidence"] + 0.5 * scores[m]["plausibility"]
        return scores[m]["plausibility"]

    ranked = sorted(MODELS, key=composite, reverse=True)

    lines.append("| Rank | Model | Avg Confidence | Plausibility | Composite Score |\n")
    lines.append("|------|-------|----------------|--------------|----------------|\n")
    for rank, model in enumerate(ranked, 1):
        conf_str = f"{scores[model]['confidence']:.1%}" if scores[model]["confidence"] is not None else "N/A"
        plaus_str = f"{scores[model]['plausibility']:.1%}"
        comp_str = f"{composite(model):.1%}"
        lines.append(f"| {rank} | **{model}** | {conf_str} | {plaus_str} | {comp_str} |\n")

    best = ranked[0]
    lines.append(f"\n### Winner: `{best}`\n\n")

    conclusions = {
        "bert": (
            "**BERT (Malaya NanoT5 transformer)** is the best performer. "
            "It achieves the highest average confidence and the most plausible sentiment predictions "
            "for Malay meeting/speech text. It handles nuanced Malay phrasing better than the rule-based approaches."
        ),
        "multinomial_nb": (
            "**Multinomial Naive Bayes** is the best performer. "
            "It strikes a good balance between speed and accuracy for Malay text, "
            "outperforming both the heavier transformer model and the simple lexicon approach."
        ),
        "lexicon": (
            "**Lexicon** is the best performer by plausibility. "
            "While it lacks probabilistic confidence scores, its word-based matching "
            "aligned most closely with the expected sentiment for these Malay speech files."
        ),
    }
    lines.append(conclusions[best] + "\n\n")

    lines.append("**Key observations:**\n")
    lines.append(f"- BERT average confidence: {bert_avg:.1%}\n")
    lines.append(f"- Multinomial NB average confidence: {nb_avg:.1%}\n")
    lines.append(f"- Full 3-model agreement: {total_agree}/{len(FILES)} files\n")
    lines.append(f"- BERT plausibility: {plaus['bert']}/{len(FILES)} | NB: {plaus['multinomial_nb']}/{len(FILES)} | Lexicon: {plaus['lexicon']}/{len(FILES)}\n")

    # ── compare against previous run ────────────────────────────────
    prev_bert_plaus = None
    improved = None
    if PREV_FILE.exists():
        import re as _re
        prev_text = PREV_FILE.read_text(encoding="utf-8")
        m = _re.search(r'\*\*bert\*\*: (\d+)/10', prev_text)
        if m:
            prev_bert_plaus = int(m.group(1))
            improved = plaus["bert"] > prev_bert_plaus

    lines.append("\n---\n")
    lines.append("## vs Previous Run (SENTIMENT_3.md)\n\n")
    if prev_bert_plaus is not None:
        change_str = "**improved**" if improved else ("same" if plaus["bert"] == prev_bert_plaus else "**worse**")
        nb_prev = None
        if PREV_FILE.exists():
            import re as _re2
            prev_text2 = PREV_FILE.read_text(encoding="utf-8")
            m2 = _re2.search(r'\*\*multinomial_nb\*\*: (\d+)/10', prev_text2)
            if m2:
                nb_prev = int(m2.group(1))
        nb_change = ""
        if nb_prev is not None:
            nb_diff = plaus["multinomial_nb"] - nb_prev
            nb_change = f"**{'improved' if nb_diff > 0 else ('same' if nb_diff == 0 else 'worse')}**"
        lines.append(f"| Metric | SENTIMENT_3 | SENTIMENT_4 | Change |\n")
        lines.append(f"|--------|-------------|-------------|--------|\n")
        lines.append(f"| BERT plausibility | {prev_bert_plaus}/10 | {plaus['bert']}/10 | {change_str} |\n")
        lines.append(f"| NB plausibility | {nb_prev}/10 | {plaus['multinomial_nb']}/10 | {nb_change} |\n")
        lines.append(f"| BERT avg confidence | — | {bert_avg:.1%} | — |\n")
        bert_ok = plaus["bert"] >= prev_bert_plaus
        nb_ok = nb_prev is None or plaus["multinomial_nb"] >= nb_prev
        verdict = "KEEP changes" if (bert_ok and nb_ok) else "REVERT — no improvement"
        lines.append(f"\n**Verdict: {verdict}**\n")
    else:
        lines.append("_(No previous run to compare against)_\n")

    # ── write file ───────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n{'='*60}")
    print(f"Report written to: {OUTPUT_FILE}")
    print(f"Best model: {best.upper()}")
    if prev_bert_plaus is not None:
        change = plaus['bert'] - prev_bert_plaus
        sign = "+" if change >= 0 else ""
        print(f"BERT plausibility: {prev_bert_plaus}/10 -> {plaus['bert']}/10 ({sign}{change})")
        print(f"NB   plausibility: ?/10 -> {plaus['multinomial_nb']}/10")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
