"""
Abstractive Summarization Module for Malay Text
Supports all 4 Mesolitica T5 models fine-tuned for Bahasa Malaysia summarization.
Includes beam search and nucleus sampling decoding strategies.
"""

import os
import warnings

# Suppress noisy warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")


# ── Available Models ────────────────────────────────────────────
# Official ROUGE scores from Malaya docs (tested on translated CNN DailyMail)
AVAILABLE_MODELS = {
    "t5-small": {
        "name": "mesolitica/finetune-summarization-t5-small-standard-bahasa-cased",
        "size_mb": 242,
        "rouge1": 0.757, "rouge2": 0.497, "rougeL": 0.304,
        "description": "Fastest. Highest ROUGE-1 (best word overlap)."
    },
    "t5-base": {
        "name": "mesolitica/finetune-summarization-t5-base-standard-bahasa-cased",
        "size_mb": 892,
        "rouge1": 0.713, "rouge2": 0.470, "rougeL": 0.367,
        "description": "Good balance of speed and coherence."
    },
    "ms-t5-small": {
        "name": "mesolitica/finetune-summarization-ms-t5-small-standard-bahasa-cased",
        "size_mb": 242,
        "rouge1": 0.743, "rouge2": 0.502, "rougeL": 0.374,
        "description": "More Malay-specific training. Highest ROUGE-2 (best bigram match)."
    },
    "ms-t5-base": {
        "name": "mesolitica/finetune-summarization-ms-t5-base-standard-bahasa-cased",
        "size_mb": 892,
        "rouge1": 0.728, "rouge2": 0.497, "rougeL": 0.377,
        "description": "Best overall coherence. Highest ROUGE-L (best long-range flow)."
    },
}

# Default: ms-t5-base for best coherence (ROUGE-L)
DEFAULT_MODEL_KEY = "ms-t5-base"

# Module-level model cache to avoid reloading
_cached_model = None
_cached_model_name = None
_patches_applied = False


def list_models():
    """Print all available models with their ROUGE scores."""
    print(f"\n{'='*70}")
    print(f"  AVAILABLE ABSTRACTIVE MODELS")
    print(f"{'='*70}")
    print(f"  {'Key':<14} {'Size':<8} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}")
    print(f"  {'─'*62}")
    for key, info in AVAILABLE_MODELS.items():
        marker = " ◀ default" if key == DEFAULT_MODEL_KEY else ""
        print(f"  {key:<14} {info['size_mb']}MB   {info['rouge1']:<10.3f} {info['rouge2']:<10.3f} {info['rougeL']:<10.3f}{marker}")
    print(f"{'='*70}\n")


def _resolve_model_name(model_key_or_name):
    """Resolve a short key (e.g. 'ms-t5-base') to a full HuggingFace model name."""
    if model_key_or_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_key_or_name]["name"]
    # If it's already a full HF model path, use as-is
    return model_key_or_name


def _apply_patches():
    """Apply compatibility patches for Malaya (only once)."""
    global _patches_applied
    if _patches_applied:
        return

    import inspect
    import numpy as np

    # Python 3.11+ compatibility patch for Malaya
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec

    # SciPy compatibility patches
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


def _load_model(model_name):
    """Load and cache the abstractive model."""
    global _cached_model, _cached_model_name

    _apply_patches()
    import malaya

    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model

    print(f"[Abstractive] Loading model: {model_name}")
    _cached_model = malaya.summarization.abstractive.huggingface(model=model_name)
    _cached_model_name = model_name
    return _cached_model


def abstractive_summarize(text, model=DEFAULT_MODEL_KEY, mode="beam",
                          max_length=256, postprocess=True,
                          # Beam search params
                          num_beams=4, no_repeat_ngram_size=3,
                          repetition_penalty=2.0, early_stopping=True,
                          # Sampling params
                          top_p=0.92, top_k=0, temperature=0.7,
                          # Postprocess params
                          n=2, threshold=0.1, reject_similarity=0.85):
    """
    Generate an abstractive summary from the input text using Malaya T5.

    Two decoding strategies:
      - "beam": Beam search — deterministic, higher factual accuracy
      - "sampling": Nucleus sampling — more natural/diverse output

    Args:
        text: Input text (typically the extractive summary output)
        model: Model key ("t5-small", "t5-base", "ms-t5-small", "ms-t5-base")
               or a full HuggingFace model path
        mode: "beam" for beam search, "sampling" for nucleus sampling
        max_length: Maximum length of generated summary in tokens
        postprocess: If True, Malaya filters output using ROUGE and removes
                     repetitive/biased sentences (recommended)

        Beam search params (used when mode="beam"):
            num_beams: Number of beams (higher = better quality, slower)
            no_repeat_ngram_size: Prevents repeating n-grams of this size
            repetition_penalty: Penalizes copying exact input (higher = more paraphrasing)
            early_stopping: Stop when all beams produce EOS

        Sampling params (used when mode="sampling"):
            top_p: Nucleus sampling probability threshold (0.0-1.0)
            top_k: Top-K filtering (0 = disabled, let top_p handle it)
            temperature: Controls randomness (lower = more focused)

        Postprocess params (used when postprocess=True):
            n: N-gram size for ROUGE filtering
            threshold: Minimum ROUGE-N score to keep a sentence
            reject_similarity: Reject sentences with similarity above this

    Returns:
        str: The abstractive summary text
    """
    model_name = _resolve_model_name(model)
    loaded_model = _load_model(model_name)

    # Build generation kwargs based on decoding mode
    gen_kwargs = {"max_length": max_length}

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

    # Postprocess kwargs
    pp_kwargs = {}
    if postprocess:
        pp_kwargs = {
            "postprocess": True,
            "n": n,
            "threshold": threshold,
            "reject_similarity": reject_similarity,
        }

    result = loaded_model.generate([text], **gen_kwargs, **pp_kwargs)

    return result[0]


# --- STANDALONE EXECUTION ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Abstractive summarization for Malay text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  t5-small      242MB  Fastest, highest ROUGE-1
  t5-base       892MB  Good balance (previous default)
  ms-t5-small   242MB  More Malay-specific, highest ROUGE-2
  ms-t5-base    892MB  Best coherence, highest ROUGE-L (new default)

Examples:
  python abstractive.py --input cleaned_text/news.txt
  python abstractive.py --input cleaned_text/news.txt --model ms-t5-base --mode sampling
  python abstractive.py --list-models
        """
    )
    parser.add_argument("--input", help="Path to input text file")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help=f"Model to use (default: {DEFAULT_MODEL_KEY})")
    parser.add_argument("--mode", choices=["beam", "sampling"], default="beam",
                        help="Decoding strategy (default: beam)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max summary length in tokens")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable Malaya's built-in ROUGE postprocessing")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models and exit")

    args = parser.parse_args()

    if args.list_models:
        list_models()
        exit(0)

    if not args.input:
        parser.error("--input is required (unless using --list-models)")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    model_info = AVAILABLE_MODELS.get(args.model, {})
    model_label = model_info.get("description", args.model)

    print(f"\n{'='*60}")
    print(f"  ABSTRACTIVE SUMMARIZATION")
    print(f"{'='*60}")
    print(f"  Model:       {args.model} — {model_label}")
    print(f"  Decoding:    {args.mode}")
    print(f"  Postprocess: {not args.no_postprocess}")
    print(f"  Input:       {len(text.split())} words")
    print(f"  Generating summary...\n")

    summary = abstractive_summarize(
        text,
        model=args.model,
        mode=args.mode,
        max_length=args.max_length,
        postprocess=not args.no_postprocess,
    )

    print(f"  SUMMARY:\n")
    print(f"  {summary}")
    print(f"\n{'='*60}")
