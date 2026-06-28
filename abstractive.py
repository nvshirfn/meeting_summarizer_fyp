"""
Abstractive Summarization Module for Malay Text
Uses mesolitica/finetune-summarization-ms-t5-base-standard-bahasa-cased
(highest ROUGE-L among Malaya T5 models, best overall coherence).
"""

import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

MODEL_NAME = "mesolitica/finetune-summarization-ms-t5-base-standard-bahasa-cased"

_cached_model = None
_patches_applied = False


def _apply_patches():
    global _patches_applied
    if _patches_applied:
        return

    import inspect
    import numpy as np

    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec

    try:
        import scipy
        if not hasattr(scipy, 'asarray'):
            scipy.asarray = np.asarray
        if not hasattr(scipy, 'ones'):
            scipy.ones = np.ones
        if not hasattr(scipy, 'zeros'):
            scipy.zeros = np.zeros
        if not hasattr(scipy, 'array'):
            scipy.array = np.array
    except ImportError:
        pass

    _patches_applied = True


def _load_model():
    global _cached_model

    _apply_patches()
    import malaya

    if _cached_model is not None:
        return _cached_model

    print(f"[Abstractive] Loading model: {MODEL_NAME}")
    _cached_model = malaya.summarization.abstractive.huggingface(model=MODEL_NAME)
    return _cached_model


def abstractive_summarize(text, mode="beam",
                          max_length=512, min_length=100, length_penalty=2.0,
                          postprocess=True, bullet_points=True,
                          num_beams=5, no_repeat_ngram_size=3,
                          repetition_penalty=2.0, early_stopping=True,
                          top_p=0.92, top_k=0, temperature=0.7,
                          n=2, threshold=0.3, reject_similarity=0.85):
    """
    Generate an abstractive summary from Malay text using ms-t5-base.

    Args:
        text:      Input text (typically extractive summary output)
        mode:      "beam" (deterministic, higher accuracy) or
                   "sampling" (more varied, higher hallucination risk)
        max_length: Max output tokens
        min_length: Min output tokens
        postprocess: Filter low-ROUGE sentences via Malaya postprocessor
    """
    loaded_model = _load_model()

    gen_kwargs = {
        "max_length": max_length,
        "min_length": min_length,
        "length_penalty": length_penalty,
    }

    if mode == "beam":
        gen_kwargs.update({
            "num_beams": num_beams,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
            "early_stopping": early_stopping,
        })
    elif mode == "sampling":
        gen_kwargs.update({
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        })
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'beam' or 'sampling'.")

    pp_kwargs = {}
    if postprocess:
        pp_kwargs = {
            "postprocess": True,
            "n": n,
            "threshold": threshold,
            "reject_similarity": reject_similarity,
        }

    result = loaded_model.generate([text], **gen_kwargs, **pp_kwargs)

    summary_text = result[0]

    if bullet_points:
        import re
        sentences = [s.strip(' \t\n\r"""\'''') for s in re.split(r'\.\s+', summary_text) if s.strip(' \t\n\r"""\'''')]
        bullets = []
        for s in sentences:
            if s.endswith('.'):
                s = s[:-1]
            bullets.append(f"•\t{s}")
        return "Ringkasan AI\n" + "\n".join(bullets)

    return summary_text


# --- STANDALONE EXECUTION ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Abstractive summarization for Malay text (ms-t5-base)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python abstractive.py --input cleaned_text/news.txt
  python abstractive.py --input cleaned_text/news.txt --mode sampling
        """
    )
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--mode", choices=["beam", "sampling"], default="beam",
                        help="Decoding strategy (default: beam)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max summary length in tokens")
    parser.add_argument("--min-length", type=int, default=100,
                        help="Min summary length in tokens")
    parser.add_argument("--length-penalty", type=float, default=2.0,
                        help="Length penalty (>1.0 encourages longer output)")
    parser.add_argument("--no-bullet-points", action="store_false", dest="bullet_points",
                        help="Disable bullet point formatting")
    parser.set_defaults(bullet_points=True)
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable Malaya's ROUGE postprocessing")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n{'='*60}")
    print(f"  ABSTRACTIVE SUMMARIZATION")
    print(f"{'='*60}")
    print(f"  Model:       ms-t5-base (mesolitica, reported ROUGE-L 0.377)")
    print(f"  Decoding:    {args.mode}")
    print(f"  Postprocess: {not args.no_postprocess}")
    print(f"  Input:       {len(text.split())} words")
    print(f"  Generating summary...\n")

    summary = abstractive_summarize(
        text,
        mode=args.mode,
        max_length=args.max_length,
        min_length=args.min_length,
        length_penalty=args.length_penalty,
        postprocess=not args.no_postprocess,
        bullet_points=args.bullet_points,
    )

    print(f"  SUMMARY:\n")
    print(f"  {summary}")
    print(f"\n{'='*60}")
