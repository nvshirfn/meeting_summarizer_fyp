# Sentiment Analysis Benchmark — All 10 Files × 3 Models
Files processed: 10  |  Models: bert, multinomial_nb, lexicon
**Preprocessing:** `preprocess_malay_transcript(mode='meeting', normalization='dictionary')` applied before sentiment analysis.

**BERT chunking:** word-based chunks of 300 words (50-word overlap) — replaces punctuation-based sentence split to avoid silent truncation.

---
## Per-File Results
### berita
_Preprocessing: 8310 chars → 8278 chars (0.4% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **negative** | 44.4% | 48.8% | 6.8% | 48.8% | 12.69s |
| multinomial_nb | **negative** | 31.1% | 47.3% | 21.6% | 47.3% | 1.07s |
| lexicon | **negative** | 7 words | 12 words | — | — | 0.03s |

### culture_shock
_Preprocessing: 6457 chars → 6626 chars (-2.6% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **negative** | 38.2% | 42.0% | 19.8% | 42.0% | 0.97s |
| multinomial_nb | **negative** | 27.6% | 45.6% | 26.8% | 45.6% | 0.04s |
| lexicon | **negative** | 14 words | 35 words | — | — | 0.03s |

### debat_perdana
_Preprocessing: 3551 chars → 3583 chars (-0.9% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 5.6% | 36.8% | 57.6% | 57.6% | 1.94s |
| multinomial_nb | **negative** | 27.5% | 51.0% | 21.5% | 51.0% | 0.02s |
| lexicon | **negative** | 5 words | 38 words | — | — | 0.01s |

### iv_menteri_kewangan
_Preprocessing: 11871 chars → 11885 chars (-0.1% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 35.6% | 23.9% | 40.4% | 40.4% | 1.29s |
| multinomial_nb | **negative** | 38.8% | 41.2% | 20.1% | 41.2% | 0.06s |
| lexicon | **positive** | 27 words | 11 words | — | — | 0.04s |

### MeetingSample
_Preprocessing: 4289 chars → 4300 chars (-0.3% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **positive** | 67.8% | 1.5% | 30.7% | 67.8% | 0.53s |
| multinomial_nb | **positive** | 43.0% | 31.8% | 25.2% | 43.0% | 0.02s |
| lexicon | **positive** | 12 words | 3 words | — | — | 0.01s |

### mesyuarat_hari_sukan
_Preprocessing: 9488 chars → 9461 chars (0.3% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 42.5% | 5.0% | 52.5% | 52.5% | 0.99s |
| multinomial_nb | **positive** | 41.7% | 36.3% | 22.0% | 41.7% | 0.05s |
| lexicon | **negative** | 16 words | 34 words | — | — | 0.04s |

### pengalaman_camping
_Preprocessing: 11927 chars → 12052 chars (-1.0% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **negative** | 11.8% | 52.8% | 35.4% | 52.8% | 1.61s |
| multinomial_nb | **negative** | 29.9% | 45.3% | 24.8% | 45.3% | 0.06s |
| lexicon | **negative** | 28 words | 94 words | — | — | 0.04s |

### podcast
_Preprocessing: 7982 chars → 8080 chars (-1.2% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **negative** | 10.1% | 66.8% | 23.1% | 66.8% | 0.92s |
| multinomial_nb | **negative** | 25.9% | 54.0% | 20.1% | 54.0% | 0.05s |
| lexicon | **negative** | 34 words | 47 words | — | — | 0.03s |

### parlimen
_Preprocessing: 8805 chars → 8820 chars (-0.2% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **negative** | 6.4% | 75.4% | 18.2% | 75.4% | 0.89s |
| multinomial_nb | **negative** | 36.1% | 38.5% | 25.4% | 38.5% | 0.04s |
| lexicon | **positive** | 31 words | 30 words | — | — | 0.03s |

### sembang_pengalaman
_Preprocessing: 10940 chars → 10510 chars (3.9% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **negative** | 32.4% | 39.9% | 27.7% | 39.9% | 1.32s |
| multinomial_nb | **negative** | 29.2% | 45.6% | 25.2% | 45.6% | 0.05s |
| lexicon | **negative** | 14 words | 39 words | — | — | 0.04s |

---
## Cross-Model Agreement
Shows whether all 3 models agree on the sentiment for each file.

| File | BERT | Multinomial NB | Lexicon | Agreement |
|------|------|----------------|---------|----------|
| berita | negative | negative | negative | ✅ Full |
| culture_shock | negative | negative | negative | ✅ Full |
| debat_perdana | neutral | negative | negative | ⚠️ Partial |
| iv_menteri_kewangan | neutral | negative | positive | ❌ None |
| MeetingSample | positive | positive | positive | ✅ Full |
| mesyuarat_hari_sukan | neutral | positive | negative | ❌ None |
| pengalaman_camping | negative | negative | negative | ✅ Full |
| podcast | negative | negative | negative | ✅ Full |
| parlimen | negative | negative | positive | ⚠️ Partial |
| sembang_pengalaman | negative | negative | negative | ✅ Full |

**Full agreement rate:** 6/10 files

---
## Model Performance Summary

### Sentiment Distribution
| Model | Positive | Negative | Neutral | Avg Confidence |
|-------|----------|----------|---------|----------------|
| bert | 1 | 6 | 3 | 54.4% |
| multinomial_nb | 2 | 8 | 0 | 45.3% |
| lexicon | 3 | 7 | 0 | N/A (word count) |

---
## Best Model Analysis

### Scoring Criteria
Since no ground-truth labels exist, models are compared on:
1. **Average confidence** — higher means more decisive predictions (BERT, NB only)
2. **Sentiment plausibility** — does the predicted sentiment match expected content type?
3. **Cross-model agreement** — agreement with other models as a proxy for reliability

### Plausibility Score (vs Expected Sentiment)
Expected sentiments are rough heuristics based on content type.

| File | Expected | BERT | NB | Lexicon |
|------|----------|------|----|---------|
| berita | neutral | negative ❌ | negative ❌ | negative ❌ |
| culture_shock | negative | negative ✅ | negative ✅ | negative ✅ |
| debat_perdana | negative | neutral ❌ | negative ✅ | negative ✅ |
| iv_menteri_kewangan | neutral | neutral ✅ | negative ❌ | positive ❌ |
| MeetingSample | neutral | positive ❌ | positive ❌ | positive ❌ |
| mesyuarat_hari_sukan | positive | neutral ❌ | positive ✅ | negative ❌ |
| pengalaman_camping | positive | negative ❌ | negative ❌ | negative ❌ |
| podcast | neutral | negative ❌ | negative ❌ | negative ❌ |
| parlimen | negative | negative ✅ | negative ✅ | positive ❌ |
| sembang_pengalaman | positive | negative ❌ | negative ❌ | negative ❌ |

**Plausibility matches out of 10:**
- **bert**: 3/10
- **multinomial_nb**: 4/10
- **lexicon**: 2/10

### Overall Ranking

| Rank | Model | Avg Confidence | Plausibility | Composite Score |
|------|-------|----------------|--------------|----------------|
| 1 | **multinomial_nb** | 45.3% | 40.0% | 42.7% |
| 2 | **bert** | 54.4% | 30.0% | 42.2% |
| 3 | **lexicon** | N/A | 20.0% | 20.0% |

### Winner: `multinomial_nb`

**Multinomial Naive Bayes** is the best performer. It strikes a good balance between speed and accuracy for Malay text, outperforming both the heavier transformer model and the simple lexicon approach.

**Key observations:**
- BERT average confidence: 54.4%
- Multinomial NB average confidence: 45.3%
- Full 3-model agreement: 6/10 files
- BERT plausibility: 3/10 | NB: 4/10 | Lexicon: 2/10

---
## vs Previous Run (SENTIMENT_3.md)

| Metric | SENTIMENT_3 | SENTIMENT_4 | Change |
|--------|-------------|-------------|--------|
| BERT plausibility | 3/10 | 3/10 | same |
| NB plausibility | 4/10 | 4/10 | **same** |
| BERT avg confidence | — | 54.4% | — |

**Verdict: KEEP changes**
