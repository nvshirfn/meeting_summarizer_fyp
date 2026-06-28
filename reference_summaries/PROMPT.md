# LLM Reference Summaries — How to Generate

These files are the **reference ("silver-standard") summaries** used by
`evaluate_summarization.py` to score the system's own summarizers with ROUGE
and semantic similarity.

They are **LLM-generated, not human-written.** This is a deliberate, documented
choice made under time constraints. Record everything below so the evaluation
is reproducible and defensible in the report.

---

## Procedure (do this for each of the 10 files)

1. Open the matching transcript in `stt_transcription_trimmed/<name>.txt`
   (e.g. `berita.txt`).
2. Paste the prompt below + the full transcript into the LLM.
3. Copy the LLM's summary into `reference_summaries/<name>.txt`,
   **replacing the entire `<PASTE …>` placeholder line.**
4. Re-run `python evaluate_summarization.py` — files still holding the
   placeholder are skipped automatically, so you can fill them in incrementally.

The 10 file stems: `berita, culture_shock, debat_perdana, iv_menteri_kewangan,
MeetingSample, mesyuarat_hari_sukan, pengalaman_camping, podcast, parlimen,
sembang_pengalaman`.

---

## The exact prompt (use the SAME prompt for every file)

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

- **LLM used (name + version):** ____________________
- **Date generated:** ____________________
- **Prompt:** the block above, used unchanged for all 10 files
- **Manual faithfulness spot-check:** verified ___ of 10 references by hand
  (note any that needed correction): ____________________

Put this table in your thesis methodology/appendix. It pre-empts the obvious
examiner question: *"where did your ground truth come from?"*

---

## Why this is acceptable (and how to frame it)

- Call them **silver-standard / LLM-generated references**, never "ground truth".
- The ROUGE/semantic scores are a **relative** comparison across *your* methods
  against a common reference — not an absolute measure of correctness.
- ROUGE is lexical and penalises paraphrasing, so extractive methods (which copy
  transcript wording) and the abstractive model (which rewrites) are not directly
  comparable on ROUGE alone — read the **semantic** column alongside it.
- Spot-checking a few references by hand lets you state they are faithful.
