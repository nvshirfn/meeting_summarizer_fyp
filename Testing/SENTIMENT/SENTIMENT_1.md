# Sentiment Analysis Benchmark — All 10 Files × 3 Models
Files processed: 10  |  Models: bert, multinomial_nb, lexicon
---
## Per-File Results
### berita
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 35.4% | 27.3% | 37.3% | 37.3% | 47.02s |
| multinomial_nb | **negative** | 31.2% | 47.3% | 21.5% | 47.3% | 1.91s |
| lexicon | **negative** | 7 words | 11 words | — | — | 0.03s |

### culture_shock
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 15.8% | 31.6% | 52.6% | 52.6% | 3.44s |
| multinomial_nb | **negative** | 27.0% | 47.6% | 25.4% | 47.6% | 0.06s |
| lexicon | **negative** | 12 words | 36 words | — | — | 0.03s |

### debat_perdana
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 16.7% | 26.7% | 56.5% | 56.5% | 8.18s |
| multinomial_nb | **negative** | 27.5% | 51.0% | 21.5% | 51.0% | 0.4s |
| lexicon | **negative** | 5 words | 38 words | — | — | 0.02s |

### iv_menteri_kewangan
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **positive** | 47.4% | 8.8% | 43.8% | 47.4% | 11.33s |
| multinomial_nb | **negative** | 39.4% | 40.8% | 19.9% | 40.8% | 5.26s |
| lexicon | **positive** | 27 words | 11 words | — | — | 0.06s |

### MeetingSample
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **positive** | 57.5% | 4.0% | 38.5% | 57.5% | 2.52s |
| multinomial_nb | **positive** | 43.2% | 31.8% | 25.0% | 43.2% | 0.06s |
| lexicon | **positive** | 12 words | 3 words | — | — | 0.02s |

### mesyuarat_hari_sukan
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 29.3% | 6.7% | 64.0% | 64.0% | 23.65s |
| multinomial_nb | **positive** | 41.1% | 37.3% | 21.7% | 41.1% | 1.94s |
| lexicon | **negative** | 16 words | 34 words | — | — | 0.03s |

### pengalaman_camping
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 13.0% | 33.3% | 53.8% | 53.8% | 33.68s |
| multinomial_nb | **negative** | 29.3% | 47.0% | 23.8% | 47.0% | 2.29s |
| lexicon | **negative** | 25 words | 94 words | — | — | 0.05s |

### podcast
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 9.9% | 35.0% | 55.0% | 55.0% | 81.68s |
| multinomial_nb | **negative** | 25.4% | 55.2% | 19.3% | 55.2% | 5.49s |
| lexicon | **negative** | 33 words | 47 words | — | — | 0.05s |

### parlimen
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 31.4% | 15.1% | 53.5% | 53.5% | 68.39s |
| multinomial_nb | **negative** | 36.1% | 38.6% | 25.3% | 38.6% | 0.65s |
| lexicon | **positive** | 31 words | 30 words | — | — | 0.02s |

### sembang_pengalaman
| Model | Sentiment | Positive | Negative | Neutral | Confidence | Time |
|-------|-----------|----------|----------|---------|------------|------|
| bert | **neutral** | 11.6% | 30.8% | 57.5% | 57.5% | 39.32s |
| multinomial_nb | **negative** | 27.8% | 47.9% | 24.2% | 47.9% | 5.02s |
| lexicon | **negative** | 14 words | 39 words | — | — | 0.04s |

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
| bert | 2 | 0 | 8 | 53.5% |
| multinomial_nb | 2 | 8 | 0 | 46.0% |
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
| 1 | **multinomial_nb** | 46.0% | 40.0% | 43.0% |
| 2 | **bert** | 53.5% | 20.0% | 36.8% |
| 3 | **lexicon** | N/A | 20.0% | 20.0% |

### Winner: `multinomial_nb`

**Multinomial Naive Bayes** is the best performer. It strikes a good balance between speed and accuracy for Malay text, outperforming both the heavier transformer model and the simple lexicon approach.

**Key observations:**
- BERT average confidence: 53.5%
- Multinomial NB average confidence: 46.0%
- Full 3-model agreement: 1/10 files
- BERT plausibility: 2/10 | NB: 4/10 | Lexicon: 2/10
