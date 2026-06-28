# LLM Reference Summaries — How to Generate

These files are the **reference ("silver-standard") summaries** used by
`evaluate_summarization.py` to score the system's own summarizers with ROUGE
and semantic similarity.

You are using **three LLMs — Claude, Gemini, and ChatGPT — as separate
reference sets.** Each LLM's summaries live in its own subfolder, and your
methods are scored against each LLM's references independently. This shows
whether your method ranking depends on which LLM you trusted.

The references are **LLM-generated, not human-written** — a deliberate,
documented choice under time constraints. Record everything below so the
evaluation is reproducible and defensible in the report.

---

## Folder layout

```
reference_summaries/
  claude/<name>.txt     ← Claude's summary for each transcript
  gemini/<name>.txt     ← Gemini's summary
  chatgpt/<name>.txt    ← ChatGPT's summary
  PROMPT.md             ← this file
```

The 10 file stems: `berita, culture_shock, debat_perdana, iv_menteri_kewangan,
MeetingSample, mesyuarat_hari_sukan, pengalaman_camping, podcast, parlimen,
sembang_pengalaman`.

So there are 3 × 10 = **30 reference files** to fill in.

---

## Procedure

For **each transcript** and **each LLM**:

1. Open a **new / fresh chat** in the LLM (do not reuse one chat across files —
   a fresh session keeps every summary independent and reproducible).
2. Paste the prompt below + the full transcript from
   `stt_transcription_trimmed/<name>.txt`.
3. Save the LLM's summary into `reference_summaries/<llm>/<name>.txt`,
   **replacing the entire `<PASTE …>` placeholder line.**

Use the **same prompt, unchanged, for every file and every LLM** — that is what
makes the three reference sets comparable.

Files still holding the placeholder are skipped automatically, so you can fill
them in incrementally and re-run `python evaluate_summarization.py` any time.

---

## The exact prompt (identical for all files and all 3 LLMs)

> Anda diberi transkrip pertuturan dalam Bahasa Melayu (hasil pengecaman
> suara, jadi mungkin ada sedikit ralat ejaan/nama). Ringkaskan kandungan
> utama transkrip ini dalam Bahasa Melayu.
>
> Syarat:
> - Hanya gunakan maklumat yang ada dalam transkrip. JANGAN tambah fakta baru.
> - Fokus pada isi/keputusan/maklumat penting, bukan basa-basi.
> - 4–6 ayat ringkas (atau 4–6 poin), padat dan jelas.
> - Kekalkan nama, nombor dan istilah seperti dalam transkrip.
>
> Transkrip:
> """
> <PASTE THE FULL TRANSCRIPT HERE>
> """

---

## Record for reproducibility (REQUIRED — fill this in)

| LLM | Model name + version | Date generated | Refs spot-checked |
|-----|----------------------|----------------|-------------------|
| Claude  | ____________________ | __________ | ___ / 10 |
| Gemini  | ____________________ | __________ | ___ / 10 |
| ChatGPT | ____________________ | __________ | ___ / 10 |

- **Prompt:** the block above, used unchanged for all files and all LLMs.
- **Sessions:** each summary generated in a fresh/independent chat.

Put this table in your thesis methodology/appendix. It pre-empts the obvious
examiner question: *"where did your ground truth come from?"*

---

## Why this is acceptable (and how to frame it)

- Call them **silver-standard / LLM-generated references**, never "ground truth".
- Using **three LLMs** reduces dependence on any single model's style/bias —
  report results against each, and note whether the method ranking is stable.
- The ROUGE/semantic scores are a **relative** comparison across *your* methods
  against a common reference — not an absolute measure of correctness.
- ROUGE is lexical and penalises paraphrasing, so extractive methods (which copy
  transcript wording) and the abstractive model (which rewrites) are not directly
  comparable on ROUGE alone — read the **semantic** column alongside it.
- Spot-checking a few references by hand lets you state they are faithful.
