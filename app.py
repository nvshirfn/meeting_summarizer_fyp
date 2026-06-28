"""
Web Interface for Malay Meeting Summarization
Flask app providing upload + results pages.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import torch
import warnings
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session

# Pre-import heavily threaded ML libraries to avoid importlib KeyErrors in Flask worker threads
import transformers
import tokenizers

# Suppress noisy warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

from preprocess import NORMALIZATION_OPTIONS, preprocess_malay_transcript
from extractive import run_extractive
from abstractive import abstractive_summarize, MODEL_NAME
from topic_modeling import perform_topic_modeling
from sentiment_analysis import analyze_sentiment

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── STT (Speech-to-Text) Setup ─────────────────────────────────
_stt_pipe = None


def get_stt_pipeline():
    """Lazy-load the Whisper STT pipeline (GPU-accelerated if available)."""
    global _stt_pipe
    if _stt_pipe is not None:
        return _stt_pipe

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "mesolitica/Malaysian-whisper-large-v3-turbo-v3"

    print(f"[STT] Loading {model_id} on {device} ...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    _stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    return _stt_pipe


# ── Dropdown options ────────────────────────────────────────────
SUMMARIZATION_MODELS = {
    "textrank":  "TextRank (TF-IDF + PageRank)",
    "lsa":       "LSA (Latent Semantic Analysis)",
    "electra":   "ELECTRA (Deep Semantic Encoder)",
}

SENTIMENT_MODELS = {
    "lexicon": "Lexicon-Based Polarity Analysis",
    "bert": "Malaya BERT (Transformer-Based)",
    "multinomial_nb": "Multinomial Naive Bayes (ML-Based)",
}

TOPIC_MODELS = {
    "lda": "LDA (Latent Dirichlet Allocation)",
    "bertopic": "BERTopic (Transformer-based Clusters)",
    "nmf": "NMF (Non-Negative Matrix Factorization)"
}

ABSTRACTIVE_MODELS = {
    "ms-t5-base": "ms-t5-base  —  Malay abstractive summarizer (mesolitica)"
}


# ── Routes ──────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def upload_page():
    return render_template(
        "upload.html",
        summarization_models=SUMMARIZATION_MODELS,
        sentiment_models=SENTIMENT_MODELS,
        topic_models=TOPIC_MODELS,
        abstractive_models=ABSTRACTIVE_MODELS,
        normalization_options=NORMALIZATION_OPTIONS,
    )


@app.route("/process", methods=["POST"])
def process():
    # ── Collect form data ───────────────────────────────────────
    uploaded_file = request.files.get("upload_file")
    if not uploaded_file or uploaded_file.filename == "":
        return redirect(url_for("upload_page"))

    ext_method     = request.form.get("summarization_model", "textrank")
    sentiment_key  = request.form.get("sentiment_model", "lexicon")
    topic_key      = request.form.get("topic_model", "lda")
    normalization_key = request.form.get("normalization", "hybrid")

    # Save uploaded file
    filename = uploaded_file.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    uploaded_file.save(save_path)

    start_time = time.time()

    # ── STEP 1  Speech-to-Text or Read Text ─────────────────────
    if filename.lower().endswith(".txt"):
        print("[Web] Step 1/5 — Reading text file (skipping STT) …")
        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_transcript = f.read()
    else:
        print("[Web] Step 1/5 — Transcribing audio …")
        stt = get_stt_pipeline()
        stt_result = stt(
            save_path,
            generate_kwargs={"language": "ms", "task": "transcribe"},
            return_timestamps=True,
        )
        raw_transcript = stt_result["text"]

    # ── STEP 2  Preprocessing ───────────────────────────────────
    print(f"[Web] Step 2/5 — Preprocessing ({normalization_key}) …")
    cleaned_transcript = preprocess_malay_transcript(
        raw_transcript,
        normalization=normalization_key,
    )

    # ── STEP 3  Topic Modeling ──────────────────────────────────
    print(f"[Web] Step 3/5 — Topic modeling ({topic_key}) …")
    topics = perform_topic_modeling(cleaned_transcript, method=topic_key)

    # ── STEP 4  Sentiment ───────────────────────────────────────
    print(f"[Web] Step 4/5 — Sentiment analysis ({sentiment_key}) …")
    sentiment = analyze_sentiment(cleaned_transcript, method=sentiment_key)

    # ── STEP 5  Extractive → Abstractive ────────────────────────
    print(f"[Web] Step 5/5 — Summarization ({ext_method} → ms-t5-base) …")
    extractive_result = run_extractive(cleaned_transcript, method=ext_method)
    summary = abstractive_summarize(
        extractive_result["combined"],
        mode="beam",
        postprocess=True,
    )

    elapsed = round(time.time() - start_time, 2)

    # ── Build results context ───────────────────────────────────
    results = {
        "filename": filename,
        "raw_transcript": raw_transcript,
        "cleaned_transcript": cleaned_transcript,
        "summary": summary,
        "sentiment": sentiment,
        "topics": topics,
        "extractive_method": SUMMARIZATION_MODELS.get(ext_method, ext_method),
        "abstractive_model": "ms-t5-base  —  mesolitica finetune-summarization",
        "sentiment_model": SENTIMENT_MODELS.get(sentiment_key, sentiment_key),
        "topic_model": TOPIC_MODELS.get(topic_key, topic_key),
        "normalization": NORMALIZATION_OPTIONS.get(normalization_key, normalization_key),
        "processing_time": elapsed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return render_template("results.html", r=results)


# ── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
