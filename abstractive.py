"""
Abstractive Summarization Module for Malay Text
Uses Malaya's T5-Base model fine-tuned for Bahasa Malaysia summarization.
"""

import os
import warnings

# Suppress noisy warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

# Default model — T5-Base for best accuracy
DEFAULT_MODEL = 'mesolitica/finetune-summarization-t5-base-standard-bahasa-cased'

# Module-level model cache to avoid reloading
_cached_model = None
_cached_model_name = None
_patches_applied = False


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


def abstractive_summarize(text, model_name=DEFAULT_MODEL, max_length=256,
                          num_beams=4, no_repeat_ngram_size=3,
                          repetition_penalty=2.0, early_stopping=True):
    """
    Generate an abstractive summary from the input text using Malaya T5.
    
    The model rewrites the extractive output into natural, fluent Malay prose
    while preserving key information and meaning.
    
    Args:
        text: Input text (typically the extractive summary output)
        model_name: HuggingFace model identifier for Malaya abstractive model
        max_length: Maximum length of generated summary in tokens
        num_beams: Number of beams for beam search (higher = better quality, slower)
        no_repeat_ngram_size: Prevents repeating n-grams of this size
        repetition_penalty: Penalizes copying exact input phrases (higher = more paraphrasing)
        early_stopping: Stop beam search when all beams produce EOS token
    
    Returns:
        str: The abstractive summary text
    """
    model = _load_model(model_name)

    result = model.generate(
        [text],
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        early_stopping=early_stopping
    )

    return result[0]


# --- STANDALONE EXECUTION ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Abstractive summarization for Malay text")
    parser.add_argument("--input", required=True, help="Path to input text file (extractive output)")
    parser.add_argument("--max-length", type=int, default=256, help="Max summary length in tokens")
    parser.add_argument("--repetition-penalty", type=float, default=2.0,
                        help="Repetition penalty (higher = more paraphrasing)")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n{'='*60}")
    print(f"  ABSTRACTIVE SUMMARIZATION — Malaya T5-Base")
    print(f"{'='*60}")
    print(f"  Input length: {len(text.split())} words")
    print(f"  Generating summary...\n")

    summary = abstractive_summarize(
        text,
        max_length=args.max_length,
        repetition_penalty=args.repetition_penalty
    )

    print(f"  SUMMARY:\n")
    print(f"  {summary}")
    print(f"\n{'='*60}")
