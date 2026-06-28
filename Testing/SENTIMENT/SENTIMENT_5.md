# Sentiment Analysis Evaluation — Human Labels vs Model Predictions
Files: 10  |  Models: bert, multinomial_nb, lexicon
Preprocessing: meeting mode, **hybrid** normalization (matches web app default)
Ground truth: manually assigned document-level labels (positive / negative / neutral) by the system developer based on transcript content.

## Per-File Results
| File | Ground Truth | BERT | Multinomial NB | Lexicon |
|------|-------------|------|----------------|--------|
| berita | **negative** | negative | negative | negative |
| culture_shock | **positive** | ~~negative~~ | ~~negative~~ | ~~negative~~ |
| debat_perdana | **negative** | ~~neutral~~ | negative | negative |
| iv_menteri_kewangan | **positive** | ~~neutral~~ | ~~negative~~ | positive |
| MeetingSample | **positive** | positive | positive | positive |
| mesyuarat_hari_sukan | **neutral** | neutral | ~~positive~~ | ~~negative~~ |
| pengalaman_camping | **negative** | ~~neutral~~ | negative | negative |
| podcast | **negative** | negative | negative | negative |
| parlimen | **neutral** | ~~negative~~ | ~~negative~~ | ~~positive~~ |
| sembang_pengalaman | **positive** | ~~negative~~ | ~~negative~~ | ~~negative~~ |

## Accuracy Summary
| Model | Correct | Accuracy |
|-------|---------|----------|
| **bert** | 4 / 10 | 40% |
| **multinomial_nb** | 5 / 10 | 50% |
| **lexicon** | 6 / 10 | 60% |

## Per-Class Breakdown
### bert
| Class | True Positives | Total in GT | Recall |
|-------|---------------|-------------|--------|
| positive | 1 | 4 | 25% |
| negative | 2 | 4 | 50% |
| neutral | 1 | 2 | 50% |

### multinomial_nb
| Class | True Positives | Total in GT | Recall |
|-------|---------------|-------------|--------|
| positive | 1 | 4 | 25% |
| negative | 4 | 4 | 100% |
| neutral | 0 | 2 | 0% |

### lexicon
| Class | True Positives | Total in GT | Recall |
|-------|---------------|-------------|--------|
| positive | 2 | 4 | 50% |
| negative | 4 | 4 | 100% |
| neutral | 0 | 2 | 0% |

## Key Findings
- **Best overall accuracy:** lexicon (6/10 = 60%)
- Ground truth labels are document-level (one label per transcript), which matches the overall sentiment output shown to users in the web app.
- Evaluation set is small (10 files) — results indicate relative model performance on this dataset and should not be generalised. Acknowledged as a limitation in the thesis.
