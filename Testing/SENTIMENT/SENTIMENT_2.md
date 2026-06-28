# Sentiment Analysis Benchmark — All 10 Files × 3 Models
Files processed: 10  |  Models: bert, multinomial_nb, lexicon
**Preprocessing:** `preprocess_malay_transcript(mode='meeting', normalization='dictionary')` applied before sentiment analysis.

---
## Per-File Results
### berita
_Preprocessing: 8310 chars → 8278 chars (0.4% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 35.7% | 26.3% | 38.0% | 38.0% | 42.74s |
| multinomial_nb | **negative** | 31.2% | 47.2% | 21.6% | 47.2% | 1.6s |
| lexicon | **negative** | 7 words | 11 words | — | — | 0.04s |

### culture_shock
_Preprocessing: 6457 chars → 6626 chars (-2.6% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 17.8% | 24.1% | 58.1% | 58.1% | 2.25s |
| multinomial_nb | **negative** | 27.7% | 45.6% | 26.7% | 45.6% | 0.04s |
| lexicon | **negative** | 13 words | 35 words | — | — | 0.02s |

### debat_perdana
_Preprocessing: 3551 chars → 3583 chars (-0.9% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 17.1% | 25.4% | 57.5% | 57.5% | 5.03s |
| multinomial_nb | **negative** | 27.5% | 51.0% | 21.5% | 51.0% | 1.97s |
| lexicon | **negative** | 5 words | 38 words | — | — | 0.01s |

### iv_menteri_kewangan
_Preprocessing: 11871 chars → 11885 chars (-0.1% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **positive** | 48.0% | 9.1% | 42.9% | 48.0% | 8.13s |
| multinomial_nb | **negative** | 38.8% | 41.2% | 20.1% | 41.2% | 1.14s |
| lexicon | **positive** | 27 words | 11 words | — | — | 0.07s |

### MeetingSample
_Preprocessing: 4289 chars → 4300 chars (-0.3% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **positive** | 54.8% | 4.0% | 41.2% | 54.8% | 1.74s |
| multinomial_nb | **positive** | 43.0% | 31.8% | 25.2% | 43.0% | 0.04s |
| lexicon | **positive** | 12 words | 3 words | — | — | 0.01s |

### mesyuarat_hari_sukan
_Preprocessing: 9488 chars → 9461 chars (0.3% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 29.8% | 5.9% | 64.3% | 64.3% | 21.23s |
| multinomial_nb | **positive** | 41.7% | 36.3% | 22.0% | 41.7% | 4.5s |
| lexicon | **negative** | 16 words | 34 words | — | — | 0.03s |

### pengalaman_camping
_Preprocessing: 11927 chars → 12052 chars (-1.0% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 15.5% | 29.4% | 55.2% | 55.2% | 26.71s |
| multinomial_nb | **negative** | 30.0% | 45.3% | 24.8% | 45.3% | 4.49s |
| lexicon | **negative** | 25 words | 94 words | — | — | 0.03s |

### podcast
_Preprocessing: 7982 chars → 8080 chars (-1.2% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 11.7% | 33.7% | 54.6% | 54.6% | 32.04s |
| multinomial_nb | **negative** | 26.0% | 54.0% | 20.0% | 54.0% | 4.95s |
| lexicon | **negative** | 32 words | 47 words | — | — | 0.03s |

### parlimen
_Preprocessing: 8805 chars → 8820 chars (-0.2% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 31.2% | 14.7% | 54.1% | 54.1% | 26.62s |
| multinomial_nb | **negative** | 36.1% | 38.5% | 25.4% | 38.5% | 4.99s |
| lexicon | **positive** | 31 words | 30 words | — | — | 0.02s |

### sembang_pengalaman
_Preprocessing: 10940 chars → 10510 chars (3.9% reduction)_

| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 16.8% | 23.4% | 59.8% | 59.8% | 25.0s |
| multinomial_nb | **negative** | 29.2% | 45.6% | 25.2% | 45.6% | 5.19s |
| lexicon | **negative** | 14 words | 39 words | — | — | 0.03s |

---
## Cross-Model Agreement
Shows whether all 3 models agree on the sentiment for each file.

| File | BERT | Multinomial NB | Lexicon | Agreement |
|------|------|----------------|---------|----------|
| berita | neutral | negative | negative | ⚠️ Partial |
| culture_shock | neutral | negative | negative | ⚠️ Partial |
| debat_perdana | neutral | negative | negative | ⚠️ Partial |
| iv_menteri_kewangan | positive | negative | positive | ⚠️ Partial |
| MeetingSample | positive | positive | positive | ✅ Full |
| mesyuarat_hari_sukan | neutral | positive | negative | ❌ None |
| pengalaman_camping | neutral | negative | negative | ⚠️ Partial |
| podcast | neutral | negative | negative | ⚠️ Partial |
| parlimen | neutral | negative | positive | ❌ None |
| sembang_pengalaman | neutral | negative | negative | ⚠️ Partial |

**Full agreement rate:** 1/10 files

---
## Model Performance Summary

### Sentiment Distribution
| Model | Positive | Negative | Neutral | Avg Confidence |
|-------|----------|----------|---------|----------------|
| bert | 2 | 0 | 8 | 54.4% |
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
| berita | neutral | neutral ✅ | negative ❌ | negative ❌ |
| culture_shock | negative | neutral ❌ | negative ✅ | negative ✅ |
| debat_perdana | negative | neutral ❌ | negative ✅ | negative ✅ |
| iv_menteri_kewangan | neutral | positive ❌ | negative ❌ | positive ❌ |
| MeetingSample | neutral | positive ❌ | positive ❌ | positive ❌ |
| mesyuarat_hari_sukan | positive | neutral ❌ | positive ✅ | negative ❌ |
| pengalaman_camping | positive | neutral ❌ | negative ❌ | negative ❌ |
| podcast | neutral | neutral ✅ | negative ❌ | negative ❌ |
| parlimen | negative | neutral ❌ | negative ✅ | positive ❌ |
| sembang_pengalaman | positive | neutral ❌ | negative ❌ | negative ❌ |

**Plausibility matches out of 10:**
- **bert**: 2/10
- **multinomial_nb**: 4/10
- **lexicon**: 2/10

### Overall Ranking

| Rank | Model | Avg Confidence | Plausibility | Composite Score |
|------|-------|----------------|--------------|----------------|
| 1 | **multinomial_nb** | 45.3% | 40.0% | 42.7% |
| 2 | **bert** | 54.4% | 20.0% | 37.2% |
| 3 | **lexicon** | N/A | 20.0% | 20.0% |

### Winner: `multinomial_nb`

**Multinomial Naive Bayes** is the best performer. It strikes a good balance between speed and accuracy for Malay text, outperforming both the heavier transformer model and the simple lexicon approach.

**Key observations:**
- BERT average confidence: 54.4%
- Multinomial NB average confidence: 45.3%
- Full 3-model agreement: 1/10 files
- BERT plausibility: 2/10 | NB: 4/10 | Lexicon: 2/10
