"""
Unified Summarization Pipeline for Malay Text
Chains: Preprocess → Extractive → Abstractive → Report

Supports both meeting transcripts and written Malay text (news, articles).
"""

import os
import argparse
from datetime import datetime

from preprocess import NORMALIZATION_OPTIONS, preprocess_malay_transcript
from extractive import run_extractive
from abstractive import abstractive_summarize, AVAILABLE_MODELS, DEFAULT_MODEL_KEY
from topic_modeling import perform_topic_modeling
from sentiment_analysis import analyze_sentiment


def run_pipeline(input_path, extractive_method="textrank", mode="meeting",
                 output_dir="summaries", skip_preprocess=False,
                 abs_model=DEFAULT_MODEL_KEY, abs_mode="beam",
                 postprocess=True, normalization="hybrid"):
    """
    Full summarization pipeline: Preprocess → Extractive → Abstractive → Save Report.
    
    Args:
        input_path: Path to the raw input text file
        extractive_method: "textrank", "lsa", or "electra"
        mode: "meeting" for spoken transcripts, "written" for news/articles
        output_dir: Directory to save the output report
        skip_preprocess: If True, skip preprocessing (input is already clean)
        abs_model: Abstractive model key ("t5-small", "t5-base", "ms-t5-small", "ms-t5-base")
        abs_mode: Decoding strategy ("beam" or "sampling")
        postprocess: If True, use Malaya's built-in ROUGE postprocessing
        normalization: "dictionary", "malaya", or "hybrid"
    
    Returns:
        dict with keys: cleaned_text, extractive_result, abstractive_summary, report_path
    """
    # ── STEP 0: Load Input ──────────────────────────────────────
    abs_info = AVAILABLE_MODELS.get(abs_model, {})
    abs_label = abs_info.get('description', abs_model)

    print(f"\n{'='*60}")
    print(f"  MALAY TEXT SUMMARIZATION PIPELINE")
    print(f"{'='*60}")
    print(f"  Input:       {input_path}")
    print(f"  Mode:        {mode}")
    print(f"  Extractive:  {extractive_method.upper()}")
    print(f"  Abstractive: {abs_model} ({abs_mode})")
    print(f"  Normalize:   {normalization}")
    print(f"  Postprocess: {postprocess}")
    print(f"  Preprocess:  {'Skipped' if skip_preprocess else 'Enabled'}")
    print(f"{'='*60}\n")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    original_word_count = len(raw_text.split())

    # ── STEP 1: Preprocess ──────────────────────────────────────
    if skip_preprocess:
        print("[Step 1/3] Preprocessing: SKIPPED (using input as-is)\n")
        cleaned_text = raw_text
    else:
        print(f"[Step 1/3] Preprocessing ({mode} mode, {normalization} normalization)...")
        cleaned_text = preprocess_malay_transcript(
            raw_text,
            mode=mode,
            normalization=normalization,
        )
        cleaned_word_count = len(cleaned_text.split())
        reduction = ((original_word_count - cleaned_word_count) / original_word_count * 100) if original_word_count > 0 else 0
        print(f"  Original: {original_word_count} words → Cleaned: {cleaned_word_count} words ({reduction:.1f}% reduced)\n")

    # ── STEP 2: Topic Modeling ──────────────────────────────────
    print(f"[Step 2/5] Topic Modeling (Gensim LDA)...")
    topics = perform_topic_modeling(cleaned_text)
    if topics:
        for t in topics:
            print(f"  Topic {t['topic_id']}: {', '.join(t['words'])}")
    else:
        print("  Not enough text for topic modeling.")
    print()

    # ── STEP 3: Sentiment Analysis ──────────────────────────────
    print(f"[Step 3/5] Sentiment Analysis (Malaya)...")
    sentiment = analyze_sentiment(cleaned_text)
    print(f"  Overall Sentiment: {sentiment}\n")

    # ── STEP 4: Extractive Summarization ────────────────────────
    print(f"[Step 4/5] Extractive Summarization ({extractive_method.upper()})...")
    extractive_result = run_extractive(cleaned_text, method=extractive_method)

    print(f"  Total sentences: {extractive_result['total_sentences']}")
    print(f"  Extracted: {extractive_result['extracted_count']} sentences\n")
    
    for i, sent in enumerate(extractive_result['sentences'], 1):
        print(f"    {i}. {sent}")
    print()

    # ── STEP 5: Abstractive Summarization ───────────────────────
    print(f"[Step 5/5] Abstractive Summarization ({abs_model}, {abs_mode})...")
    print(f"  Rewriting for natural flow and accuracy (this may take a moment)...\n")

    abstractive_summary = abstractive_summarize(
        extractive_result['combined'],
        model=abs_model,
        mode=abs_mode,
        postprocess=postprocess
    )

    print(f"  FINAL SUMMARY:")
    print(f"  {abstractive_summary}\n")

    # ── STEP 4: Save Report ─────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename from input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{base_name}_{extractive_method}_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"MALAY TEXT SUMMARIZATION REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Input:              {input_path}\n")
        f.write(f"Mode:               {mode}\n")
        f.write(f"Extractive Method:  {extractive_method.upper()}\n")
        f.write(f"Abstractive Model:  {abs_model} ({abs_mode})\n")
        f.write(f"Normalization:      {NORMALIZATION_OPTIONS.get(normalization, normalization)}\n")
        f.write(f"Postprocess:        {postprocess}\n")
        f.write(f"Generated:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"--- DOCUMENT METADATA ---\n")
        f.write(f"Sentiment Analysis: {str(sentiment).upper()}\n")
        f.write(f"Identified Topics:\n")
        if topics:
            for t in topics:
                f.write(f"  Topic {t['topic_id']}: {', '.join(t['words'])}\n")
        else:
            f.write("  None\n")
        f.write("\n")

        f.write(f"--- EXTRACTIVE KEY POINTS ({extractive_method.upper()}) ---\n")
        for i, sent in enumerate(extractive_result['sentences'], 1):
            f.write(f"{i}. {sent}\n")

        f.write(f"\n--- FINAL ABSTRACTIVE SUMMARY ({abs_model.upper()}, {abs_mode.upper()}) ---\n")
        f.write(abstractive_summary)
        f.write("\n")

    print(f"{'='*60}")
    print(f"  Report saved to: {report_path}")
    print(f"{'='*60}\n")

    return {
        "cleaned_text": cleaned_text,
        "topics": topics,
        "sentiment": sentiment,
        "extractive_result": extractive_result,
        "abstractive_summary": abstractive_summary,
        "report_path": report_path
    }


# --- CLI ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Malay Text Summarization Pipeline: Preprocess → Extractive → Abstractive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --input stt_transcription/culture_shock.txt --method textrank
  python pipeline.py --input stt_transcription/culture_shock.txt --method electra --abs-model ms-t5-base
  python pipeline.py --input cleaned_text/news.txt --method lsa --mode written --skip-preprocess
  python pipeline.py --input stt_transcription/culture_shock.txt --method textrank --abs-mode sampling
        """
    )
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--method", choices=["textrank", "lsa", "electra"], default="textrank",
                        help="Extractive summarization method (default: textrank)")
    parser.add_argument("--mode", choices=["meeting", "written"], default="meeting",
                        help="Text type: 'meeting' for transcripts, 'written' for news/articles (default: meeting)")
    parser.add_argument("--output-dir", default="summaries",
                        help="Directory to save output reports (default: summaries)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (use if input is already cleaned)")
    parser.add_argument("--normalization", choices=list(NORMALIZATION_OPTIONS.keys()), default="hybrid",
                        help="Normalization strategy for preprocessing (default: dictionary)")
    parser.add_argument("--abs-model", default=DEFAULT_MODEL_KEY,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help=f"Abstractive model (default: {DEFAULT_MODEL_KEY})")
    parser.add_argument("--abs-mode", choices=["beam", "sampling"], default="beam",
                        help="Decoding strategy: 'beam' (accurate) or 'sampling' (natural) (default: beam)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable Malaya's built-in ROUGE postprocessing")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    run_pipeline(
        input_path=args.input,
        extractive_method=args.method,
        mode=args.mode,
        output_dir=args.output_dir,
        skip_preprocess=args.skip_preprocess,
        abs_model=args.abs_model,
        abs_mode=args.abs_mode,
        postprocess=not args.no_postprocess,
        normalization=args.normalization
    )
