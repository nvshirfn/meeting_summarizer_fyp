# Summarization Evaluation — ROUGE vs LLM References (per-LLM)
Methods: textrank, lsa, electra, abstractive
Reference LLMs: claude, gemini, chatgpt (scored as separate reference sets)
Preprocessing: meeting mode, **hybrid** normalization (matches the web app default)

> **References are LLM-generated (silver standard), not human gold.** Scores measure similarity to each LLM's summary — a relative comparison across methods, not an absolute quality grade. ROUGE is lexical (penalises paraphrase); the semantic column is embedding-based (robust to paraphrase). See `reference_summaries/PROMPT.md` for the exact models and prompt.

_⚠️ Abstractive postprocess (Malaya ROUGE filter, the webpage default) crashed on: MeetingSample — these used `postprocess=False` instead. The live app would error on these inputs; worth noting as a robustness limitation._

---
## Aggregate by Reference LLM (mean across scored files)
### Reference: claude  _(files scored: 10)_
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| **textrank** | 0.300 | 0.085 | 0.166 | 0.695 |
| **lsa** | 0.324 | 0.103 | 0.145 | 0.615 |
| **electra** | 0.253 | 0.065 | 0.129 | 0.592 |
| **abstractive** | 0.195 | 0.061 | 0.116 | 0.618 |

### Reference: gemini  _(files scored: 10)_
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| **textrank** | 0.280 | 0.069 | 0.153 | 0.682 |
| **lsa** | 0.257 | 0.067 | 0.126 | 0.571 |
| **electra** | 0.230 | 0.045 | 0.121 | 0.578 |
| **abstractive** | 0.191 | 0.041 | 0.107 | 0.602 |

### Reference: chatgpt  _(files scored: 10)_
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| **textrank** | 0.278 | 0.076 | 0.168 | 0.677 |
| **lsa** | 0.267 | 0.086 | 0.140 | 0.619 |
| **electra** | 0.251 | 0.056 | 0.127 | 0.569 |
| **abstractive** | 0.205 | 0.055 | 0.124 | 0.602 |

---
## Average Across All Reference LLMs
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| **textrank** | 0.286 | 0.077 | 0.162 | 0.684 |
| **lsa** | 0.282 | 0.085 | 0.137 | 0.601 |
| **electra** | 0.244 | 0.056 | 0.125 | 0.580 |
| **abstractive** | 0.197 | 0.053 | 0.116 | 0.607 |

### Best method per metric (averaged over LLMs)
- **ROUGE-1:** textrank (0.286)
- **ROUGE-2:** lsa (0.085)
- **ROUGE-L:** textrank (0.162)
- **Semantic:** textrank (0.684)

---
## Per-File Detail
### Reference: claude
#### berita
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.301 | 0.124 | 0.207 | 0.426 |
| lsa | 0.354 | 0.180 | 0.171 | 0.267 |
| electra | 0.270 | 0.086 | 0.146 | 0.379 |
| abstractive | 0.196 | 0.090 | 0.107 | 0.347 |

#### culture_shock
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.306 | 0.066 | 0.170 | 0.818 |
| lsa | 0.282 | 0.043 | 0.097 | 0.826 |
| electra | 0.167 | 0.000 | 0.084 | 0.701 |
| abstractive | 0.180 | 0.025 | 0.105 | 0.707 |

#### debat_perdana
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.392 | 0.148 | 0.209 | 0.806 |
| lsa | 0.342 | 0.125 | 0.195 | 0.747 |
| electra | 0.166 | 0.050 | 0.091 | 0.619 |
| abstractive | 0.261 | 0.071 | 0.134 | 0.837 |

#### iv_menteri_kewangan
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.400 | 0.187 | 0.279 | 0.733 |
| lsa | 0.501 | 0.226 | 0.188 | 0.554 |
| electra | 0.398 | 0.165 | 0.185 | 0.506 |
| abstractive | 0.328 | 0.189 | 0.269 | 0.452 |

#### MeetingSample
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.296 | 0.083 | 0.198 | 0.672 |
| lsa | 0.492 | 0.209 | 0.299 | 0.788 |
| electra | 0.419 | 0.133 | 0.251 | 0.776 |
| abstractive | 0.063 | 0.021 | 0.052 | 0.476 |

#### mesyuarat_hari_sukan
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.276 | 0.053 | 0.112 | 0.661 |
| lsa | 0.168 | 0.006 | 0.081 | 0.639 |
| electra | 0.167 | 0.019 | 0.084 | 0.562 |
| abstractive | 0.185 | 0.021 | 0.082 | 0.626 |

#### pengalaman_camping
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.226 | 0.012 | 0.115 | 0.621 |
| lsa | 0.241 | 0.022 | 0.089 | 0.432 |
| electra | 0.203 | 0.014 | 0.109 | 0.595 |
| abstractive | 0.158 | 0.030 | 0.079 | 0.650 |

#### podcast
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.263 | 0.060 | 0.102 | 0.623 |
| lsa | 0.272 | 0.060 | 0.112 | 0.505 |
| electra | 0.232 | 0.055 | 0.110 | 0.548 |
| abstractive | 0.206 | 0.069 | 0.100 | 0.603 |

#### parlimen
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.317 | 0.063 | 0.151 | 0.759 |
| lsa | 0.374 | 0.119 | 0.136 | 0.704 |
| electra | 0.334 | 0.112 | 0.148 | 0.553 |
| abstractive | 0.210 | 0.042 | 0.115 | 0.799 |

#### sembang_pengalaman
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.225 | 0.051 | 0.115 | 0.831 |
| lsa | 0.212 | 0.036 | 0.079 | 0.687 |
| electra | 0.171 | 0.017 | 0.077 | 0.684 |
| abstractive | 0.161 | 0.056 | 0.118 | 0.683 |

### Reference: gemini
#### berita
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.262 | 0.069 | 0.187 | 0.542 |
| lsa | 0.268 | 0.081 | 0.113 | 0.260 |
| electra | 0.225 | 0.032 | 0.105 | 0.514 |
| abstractive | 0.206 | 0.080 | 0.103 | 0.406 |

#### culture_shock
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.253 | 0.036 | 0.134 | 0.743 |
| lsa | 0.183 | 0.022 | 0.085 | 0.785 |
| electra | 0.138 | 0.014 | 0.090 | 0.673 |
| abstractive | 0.174 | 0.016 | 0.103 | 0.631 |

#### debat_perdana
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.381 | 0.154 | 0.238 | 0.848 |
| lsa | 0.346 | 0.112 | 0.247 | 0.797 |
| electra | 0.133 | 0.009 | 0.071 | 0.621 |
| abstractive | 0.253 | 0.090 | 0.156 | 0.769 |

#### iv_menteri_kewangan
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.380 | 0.094 | 0.179 | 0.628 |
| lsa | 0.362 | 0.112 | 0.115 | 0.538 |
| electra | 0.400 | 0.134 | 0.179 | 0.591 |
| abstractive | 0.265 | 0.075 | 0.148 | 0.577 |

#### MeetingSample
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.362 | 0.114 | 0.226 | 0.697 |
| lsa | 0.366 | 0.153 | 0.220 | 0.688 |
| electra | 0.351 | 0.098 | 0.216 | 0.685 |
| abstractive | 0.080 | 0.016 | 0.048 | 0.518 |

#### mesyuarat_hari_sukan
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.306 | 0.103 | 0.136 | 0.532 |
| lsa | 0.248 | 0.058 | 0.115 | 0.566 |
| electra | 0.199 | 0.017 | 0.091 | 0.474 |
| abstractive | 0.235 | 0.027 | 0.136 | 0.623 |

#### pengalaman_camping
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.194 | 0.015 | 0.111 | 0.650 |
| lsa | 0.137 | 0.013 | 0.073 | 0.462 |
| electra | 0.154 | 0.006 | 0.109 | 0.628 |
| abstractive | 0.179 | 0.024 | 0.093 | 0.662 |

#### podcast
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.223 | 0.042 | 0.119 | 0.673 |
| lsa | 0.217 | 0.040 | 0.111 | 0.446 |
| electra | 0.223 | 0.042 | 0.107 | 0.408 |
| abstractive | 0.161 | 0.043 | 0.085 | 0.471 |

#### parlimen
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.251 | 0.051 | 0.135 | 0.733 |
| lsa | 0.239 | 0.043 | 0.108 | 0.539 |
| electra | 0.295 | 0.065 | 0.134 | 0.572 |
| abstractive | 0.208 | 0.030 | 0.126 | 0.747 |

#### sembang_pengalaman
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.191 | 0.012 | 0.064 | 0.770 |
| lsa | 0.200 | 0.038 | 0.076 | 0.628 |
| electra | 0.176 | 0.037 | 0.103 | 0.611 |
| abstractive | 0.147 | 0.009 | 0.069 | 0.613 |

### Reference: chatgpt
#### berita
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.313 | 0.133 | 0.205 | 0.467 |
| lsa | 0.325 | 0.144 | 0.134 | 0.362 |
| electra | 0.285 | 0.081 | 0.119 | 0.473 |
| abstractive | 0.188 | 0.083 | 0.088 | 0.405 |

#### culture_shock
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.228 | 0.028 | 0.139 | 0.761 |
| lsa | 0.220 | 0.048 | 0.124 | 0.825 |
| electra | 0.208 | 0.039 | 0.146 | 0.677 |
| abstractive | 0.188 | 0.018 | 0.152 | 0.705 |

#### debat_perdana
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.344 | 0.102 | 0.169 | 0.547 |
| lsa | 0.286 | 0.087 | 0.186 | 0.600 |
| electra | 0.128 | 0.000 | 0.069 | 0.506 |
| abstractive | 0.236 | 0.057 | 0.114 | 0.583 |

#### iv_menteri_kewangan
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.334 | 0.101 | 0.217 | 0.776 |
| lsa | 0.296 | 0.129 | 0.118 | 0.653 |
| electra | 0.389 | 0.156 | 0.206 | 0.519 |
| abstractive | 0.381 | 0.145 | 0.254 | 0.505 |

#### MeetingSample
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.323 | 0.143 | 0.263 | 0.745 |
| lsa | 0.453 | 0.207 | 0.327 | 0.793 |
| electra | 0.367 | 0.118 | 0.215 | 0.774 |
| abstractive | 0.055 | 0.028 | 0.055 | 0.621 |

#### mesyuarat_hari_sukan
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.297 | 0.083 | 0.164 | 0.629 |
| lsa | 0.270 | 0.065 | 0.146 | 0.670 |
| electra | 0.227 | 0.015 | 0.098 | 0.552 |
| abstractive | 0.205 | 0.058 | 0.131 | 0.662 |

#### pengalaman_camping
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.154 | 0.022 | 0.094 | 0.581 |
| lsa | 0.125 | 0.014 | 0.067 | 0.500 |
| electra | 0.153 | 0.007 | 0.080 | 0.602 |
| abstractive | 0.213 | 0.049 | 0.135 | 0.604 |

#### podcast
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.233 | 0.057 | 0.135 | 0.678 |
| lsa | 0.204 | 0.036 | 0.098 | 0.459 |
| electra | 0.223 | 0.058 | 0.116 | 0.400 |
| abstractive | 0.153 | 0.051 | 0.076 | 0.464 |

#### parlimen
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.300 | 0.077 | 0.176 | 0.775 |
| lsa | 0.250 | 0.077 | 0.117 | 0.700 |
| electra | 0.320 | 0.072 | 0.138 | 0.553 |
| abstractive | 0.263 | 0.054 | 0.162 | 0.807 |

#### sembang_pengalaman
| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Semantic |
|--------|---------|---------|---------|----------|
| textrank | 0.249 | 0.018 | 0.116 | 0.809 |
| lsa | 0.238 | 0.055 | 0.079 | 0.622 |
| electra | 0.211 | 0.016 | 0.086 | 0.630 |
| abstractive | 0.167 | 0.009 | 0.074 | 0.662 |

