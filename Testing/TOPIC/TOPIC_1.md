# Topic Modeling Evaluation — Coherence and Diversity
Files: 10  |  Methods: lda, nmf, bertopic
Preprocessing: meeting mode, **hybrid** normalization (matches web app default)

> **Coherence (c_v):** measures how semantically related the top keywords within each topic are — higher is better. Computed via gensim CoherenceModel using each transcript as the reference corpus.
> **Diversity:** proportion of unique words across all detected topics — higher means less repetition between topics (max 1.0).

---
## Aggregate Results (mean across all files)
| Model | Avg Coherence (c_v) | Avg Diversity | Avg Topics Detected |
|-------|---------------------|---------------|---------------------|
| **lda** | 0.524 | 0.907 | 3.0 |
| **nmf** | 0.512 | 0.993 | 3.0 |
| **bertopic** | 0.426 | 0.772 | 7.2 |

---
## Key Findings
- **Highest coherence:** lda (0.524) — produces the most semantically related keyword groups per topic.
- **Highest diversity:** nmf (0.993) — least repetition of keywords across topics.
- Coherence is computed against each individual transcript (not a large external corpus), so scores reflect within-document co-occurrence. Use these numbers for relative comparison across models, not as absolute quality thresholds.
- Topic detection is an unsupervised task with no ground truth. Coherence and diversity are standard intrinsic metrics used when human-annotated topic labels are unavailable.

---
## Per-File Detail
### berita
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.534 | 0.933 | 3 | membaca, quran, terbabit, pengajian, pencipta  /  negara, pemain, bola, undang, sepak  /  tafsir, undang, mangsa, urusan, tindakan |
| nmf | 0.615 | 1.000 | 3 | undang, pemain, negara, sepak, bola  /  membaca, quran, pencipta, manusia, nama  /  calon, menandatangani, parti, fadil, fahmi |
| bertopic | 0.404 | 1.000 | 3 | siasatan, mahkamah, difilkan, mangsa, dokumentasi  /  abdihil, muhammad, masjid, islam, umat  /  tindakan, persekutuan, negara, bangsa, zahid |

### culture_shock
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.317 | 0.933 | 3 | sarawak, cakap, roti, pasal, belajar  /  sarawak, kampus, universiti, kena, lorat  /  suara, nyanyi, sirap, merdu, kawan |
| nmf | 0.319 | 0.933 | 3 | sarawak, cakap, roti, gadinia, pasal  /  belajar, kehidupan, nyanyi, neutral, tengok  /  sirap, seringgit, jual, beli, roti |
| bertopic | 0.337 | 0.800 | 4 | kuala, merdu, ramai, kembang, kelab  /  borneo, kuala, budaya, teraihat, pantai  /  mall, binaan, barat, tengok, kuliah  /  seringgit, pantai, , ,  |

### debat_perdana
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.753 | 1.000 | 3 | kerugiannya, forex, kerajaan, rasuah, anwar  /  politik, negara, sprm, menteri, perdana  /  bilion, ambil, forensik, puluhan, hilang |
| nmf | 0.426 | 1.000 | 3 | forensik, ringgit, jatuh, laporan, justice  /  bilion, puluhan, hilang, kerugiannya, dollar  /  negara, diyakini, politik, ambil, rasuah |
| bertopic | 0.654 | 0.400 | 4 | menghalang, , , ,   /  soalan, pendirian, undang, siasatan, ahli  /  , , , ,   /  kerugiannya, , , ,  |

### iv_menteri_kewangan
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.595 | 0.933 | 3 | sektor, langkah, fasa, ekonomi, pelan  /  bilion, ringgit, vaksinasi, program, peruntukan  /  bantuan, langkah, kerajaan, rakyat, dilaksanakan |
| nmf | 0.637 | 1.000 | 3 | bilion, ringgit, peruntukan, disalurkan, setakat  /  langkah, fasa, mengkaji, dilaksanakan, bantuan  /  rakyat, perniagaan, vaksinasi, terkesan, menteri |
| bertopic | 0.424 | 0.960 | 5 | tempoh, dijangka, hala, bertujuan, mengkaji  /  memanfaatkan, tambahan, vaksinasi, terutamanya, terima  /  tahap, sehinggalah, setahun, mulakan, kadar  /  ramai, impak, think, fahami, terima  /  strategi, strategy, contoh, sektor, pekerjaan |

### MeetingSample
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.718 | 0.733 | 3 | mencadangkan, okey, mesyuarat, kasih, terima  /  pengurusi, mesyuarat, okey, terima, kasih  /  syarikat, lumpur, kuala, cawangan, ampang |
| nmf | 0.739 | 1.000 | 3 | syarikat, lumpur, kuala, penubuhan, cawangan  /  terima, kasih, pengurusi, majlis, hadir  /  okey, mesyuarat, pengusaha, serahkan, mencadangkan |
| bertopic | 0.514 | 1.000 | 2 | lancar, selamat, memohon, okey, jumpa  /  berikutan, bergabung, teruskan, peruntukan, mesyuarat |

### mesyuarat_hari_sukan
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.759 | 0.800 | 3 | peserta, pelajar, berkaitan, program, sukan  /  program, sukan, bajet, permainan, aktiviti  /  mesyuarat, tarikh, minit, sukan, mencadangkan |
| nmf | 0.395 | 1.000 | 3 | program, sukan, bajet, perlukan, tarikh  /  mesyuarat, minit, mengedarkan, sahkan, sedia  /  baiklah, okey, berbangkit, mesyuaranya, kompleks |
| bertopic | 0.337 | 0.900 | 6 | baiklah, faham, pendapat, semualah, okey  /  mishela, permohonan, cakap, neka, sagu  /  mesyuarat, hadir, maklum, minit, sahkan  /  tarikh, sesuai, dibincangkan, inisiatif, maklum  /  sesuai, perlukan, baiklah, aktiviti, kelengkapan  /  memulakan, fakulti, laluan, dijalankan, mencadangkan |

### pengalaman_camping
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.382 | 0.933 | 3 | pergi, benda, okey, contoh, pengalaman  /  this, makan, masak, camping, ramai  /  kena, anak, fikir, benda, genset |
| nmf | 0.483 | 1.000 | 3 | pergi, camping, beritahu, supposedly, enjoy  /  benda, just, luxury, share, cerita  /  kena, masak, okey, contoh, siakap |
| bertopic | 0.330 | 0.370 | 27 | kisah, aduh, salah, fikir, contoh  /  sekonok, letak, melepaskan, cakap, suka  /  cakap, kena, , ,   /  siakap, kena, okey, habis,   /  selesa, cakap, pandai, fikir, berpikir  /  cakap, letak, habis, ,   /  kena, perangai, mangkak, benda, mencelaka  /  macik, kasih, nurul, terima,   /  suka, beritahu, benci, mulakan, nampak  /  , , , ,   /  selidik, nampak, berjaga, ,   /  selesa, kemping, bertenang, tidur, tahulah  /  , , , ,   /  cakap, benda, keret, okey,   /  kacau, malang, aduh, benda, pelik  /  like, benda, , ,   /  siakap, keruh, , ,   /  , , , ,   /  benda, kipas, pakai, penggunaan, genset  /  , , , ,   /  contoh, kena, , ,   /  benda, , , ,   /  silap, pergi, , ,   /  pergi, apalah, , ,   /  disangkal, , , ,   /  , , , ,   /  , , , ,  |

### podcast
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.362 | 1.000 | 3 | politik, kena, ambil, rakyat, tipu  /  okey, menyumbang, campaign, manager, jumpa  /  parti, benda, pemilihan, pilihan_raya, panik |
| nmf | 0.542 | 1.000 | 3 | parti, pemilihan, labur, raya, politik  /  ambil, rakyat, pandai, deny, sekejap  /  manager, campaign, kena, explain, aduh |
| bertopic | 0.489 | 0.760 | 5 | sentimen, cakap, fikir, intimidate, tipu  /  sekian, fikirkan, selagi, mahukan, pelan  /  faham, cakap, letak, untungkan, kejadian  /  benda, lihat, , ,   /  berbeza, kegagalan, , ,  |

### parlimen
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.275 | 0.933 | 3 | terima, kasih, kawasan, rakyat, kerajaan  /  arahkan, mental, hidupan, permohonan, liar  /  kawasan, kera, selesai, kementerian, melaksanakan |
| nmf | 0.479 | 1.000 | 3 | terima, kasih, ucapkan, speaker, kenal  /  kawasan, selesai, kera, parlimen, tajuk  /  arahkan, mental, disorder, bipolar, tarali |
| bertopic | 0.332 | 0.560 | 10 | populasi, komuniti, malaysia, akibat, pertimbangan  /  ringgit, ekonomi, inisiatif, peruntukan, tempoh  /  terima, arahkan, bersetuju, kenal, kasih  /  ayat, , , ,   /  encik, terima, kasih, ucapkan, kementerian  /  arahkan, sokonglah, , ,   /  , , , ,   /  dijalankan, kenal, , ,   /  bekerjasama, nama, negara, rakyat, kawasan  /  berhormat, , , ,  |

### sembang_pengalaman
| Model | Coherence | Diversity | Topics | Keywords |
|-------|-----------|-----------|--------|----------|
| lda | 0.546 | 0.867 | 3 | iran, berjumpa, suni, masuk, pengalaman  /  azan, cari, dengar, masuk, pusing  /  masjid, bawa, pergi, cari, gambar |
| nmf | 0.484 | 1.000 | 3 | azan, habis, dengar, terkejutlah, arah  /  cari, kemping, bermalam, taman, pusing  /  masjid, pergi, bawa, iran, suni |
| bertopic | 0.443 | 0.967 | 6 | hala, azan, kemah, habis, terbengkalai  /  traveler, sembayang, ramai, berjumpa, terpinga  /  taman, bandar, lokal, perjalanan, cari  /  kena, masuk, berbaki, haiblan,   /  polis, bawa, ikutkan, pergi, cerita  /  masuk, hantar, menyekat, cakap, alhamdulillah |

