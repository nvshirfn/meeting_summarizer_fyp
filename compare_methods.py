"""
Compare all 3 extractive methods (TextRank, LSA, ELECTRA) side-by-side
on the same input text, then run abstractive summarization on each.

Usage:
  python compare_methods.py --input stt_transcription/culture_shock.txt
  python compare_methods.py --input cleaned_text/news.txt --mode written --skip-preprocess
"""

import os
import argparse
from datetime import datetime

from preprocess import preprocess_malay_transcript
from extractive import run_extractive
from abstractive import abstractive_summarize


METHODS = ["textrank", "lsa", "electra"]


def compare_all(input_path, mode="meeting", output_dir="summaries",
                skip_preprocess=False, skip_abstractive=False):
    """
    Run all 3 extractive methods on the same input, then abstractive on each.
    
    Args:
        input_path: Path to the raw input text file
        mode: "meeting" or "written"
        output_dir: Directory to save the comparison report
        skip_preprocess: If True, skip preprocessing
        skip_abstractive: If True, only compare extractive outputs (faster)
    
    Returns:
        dict mapping method name to results
    """
    # ── Load & Preprocess ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON: ALL EXTRACTIVE METHODS")
    print(f"{'='*60}")
    print(f"  Input: {input_path}")
    print(f"  Mode:  {mode}")
    print(f"{'='*60}\n")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if skip_preprocess:
        print("[Preprocess] Skipped\n")
        cleaned_text = raw_text
    else:
        print(f"[Preprocess] Cleaning text ({mode} mode)...")
        cleaned_text = preprocess_malay_transcript(raw_text, mode=mode)
        original_len = len(raw_text.split())
        cleaned_len = len(cleaned_text.split())
        reduction = ((original_len - cleaned_len) / original_len * 100) if original_len > 0 else 0
        print(f"  {original_len} words → {cleaned_len} words ({reduction:.1f}% reduced)\n")

    # ── Run Each Extractive Method ──────────────────────────────
    results = {}

    for method in METHODS:
        print(f"{'─'*60}")
        print(f"  EXTRACTIVE: {method.upper()}")
        print(f"{'─'*60}")

        try:
            ext_result = run_extractive(cleaned_text, method=method)

            print(f"  Sentences: {ext_result['extracted_count']} / {ext_result['total_sentences']}\n")
            for i, sent in enumerate(ext_result['sentences'], 1):
                print(f"    {i}. {sent}")
            print()

            # Run abstractive on this extractive output
            if not skip_abstractive:
                print(f"  [Abstractive] Generating summary from {method.upper()} output...")
                abs_summary = abstractive_summarize(ext_result['combined'])
                print(f"  [Abstractive] Result: {abs_summary}\n")
            else:
                abs_summary = "(skipped)"

            results[method] = {
                "extractive": ext_result,
                "abstractive": abs_summary
            }

        except Exception as e:
            print(f"  ERROR: {e}\n")
            results[method] = {
                "extractive": None,
                "abstractive": None,
                "error": str(e)
            }

    # ── Save Comparison Report ──────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"{base_name}_comparison_{timestamp}.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EXTRACTIVE METHOD COMPARISON REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Input:     {input_path}\n")
        f.write(f"Mode:      {mode}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        for method in METHODS:
            r = results.get(method, {})
            f.write(f"{'─'*60}\n")
            f.write(f"METHOD: {method.upper()}\n")
            f.write(f"{'─'*60}\n")

            if "error" in r:
                f.write(f"ERROR: {r['error']}\n\n")
                continue

            ext = r.get("extractive")
            if ext:
                f.write(f"Extracted {ext['extracted_count']} / {ext['total_sentences']} sentences\n\n")
                f.write(f"EXTRACTIVE OUTPUT:\n")
                for i, sent in enumerate(ext['sentences'], 1):
                    f.write(f"  {i}. {sent}\n")

            abs_text = r.get("abstractive", "(skipped)")
            f.write(f"\nABSTRACTIVE SUMMARY:\n")
            f.write(f"  {abs_text}\n\n")

        # ── Summary Table ───────────────────────────────────────
        f.write(f"{'='*60}\n")
        f.write(f"QUICK COMPARISON\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Method':<12} {'Extracted':<12} {'Abstractive Length':<20}\n")
        f.write(f"{'─'*44}\n")

        for method in METHODS:
            r = results.get(method, {})
            if "error" in r:
                f.write(f"{method:<12} {'ERROR':<12} {'—':<20}\n")
            else:
                ext_count = r['extractive']['extracted_count'] if r.get('extractive') else 0
                abs_len = len(r.get('abstractive', '').split()) if r.get('abstractive') else 0
                f.write(f"{method:<12} {ext_count:<12} {abs_len} words\n")

    print(f"\n{'='*60}")
    print(f"  Comparison report saved to: {report_path}")
    print(f"{'='*60}\n")

    return results


# --- CLI ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare all 3 extractive methods on the same Malay text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_methods.py --input stt_transcription/culture_shock.txt
  python compare_methods.py --input cleaned_text/news.txt --mode written --skip-preprocess
  python compare_methods.py --input stt_transcription/culture_shock.txt --extractive-only
        """
    )
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--mode", choices=["meeting", "written"], default="meeting",
                        help="Text type (default: meeting)")
    parser.add_argument("--output-dir", default="summaries",
                        help="Directory to save comparison report (default: summaries)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing")
    parser.add_argument("--extractive-only", action="store_true",
                        help="Only compare extractive outputs (skip abstractive, much faster)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    compare_all(
        input_path=args.input,
        mode=args.mode,
        output_dir=args.output_dir,
        skip_preprocess=args.skip_preprocess,
        skip_abstractive=args.extractive_only
    )
