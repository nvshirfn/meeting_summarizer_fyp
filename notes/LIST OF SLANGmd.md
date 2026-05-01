#### **LIST OF SLANG/FILLER/INFORMAL**



Preprocessing technique used:



1. Lowercasing (Added Lowercasing as an Argument = python preprocess.py --input transcript.txt --lowercase)
2. Filler words removal (preprocess.py)
3. Slang / Informal / English Word Replacement (preprocess.py)
4. Repeated Word Removal (preprocess.py)
5. Remove puncuation/special charachter
6. Multiple punctuation cleanup (preprocess.py)
7. Tokenization (extractive.py)



DON'T HAVE

1. Stop Words Removal
2. Stemming
3. Lemmatization
4. Remove numbers
5. Remove extra spaces





TOPIC DETECTION (preprocess)

1. Stemming (remove imbuhan daripada perkataan to display root word)
* keep stemming for accuracy
* then map root words back to the most common original word
* display nicer words in output
* contoh: perbelanjaan, membelanja, perbelanjaan = belanja, belanja, belanja
* belanja ada byk so dipilih as topic
* but, extra logic checks: Which original word appeared most?
* then original word yg appeared the most akan display in topic detection bukan 'belanja'
* So final topic becomes | Topic: perbelanjaan, perbincangan, pengurusan
* Instead of: | Topic: belanja, bincang, urus



2\. POS Filtering

* Keeping only Nouns and Verbs
* The best topics are almost always described by Nouns (e.g., "syarikat", "bajet") and Verbs (e.g., "bincang", "kurang"). Adjectives or adverbs (e.g., "secepat", "sangat", "optimum") usually just add noise to topic clusters.
* The fix: Before passing the text to NMF/LDA, we can run a Part-Of-Speech (POS) tagger (like malaya.pos) and literally delete any word that isn't a Noun or a Core Verb. This acts as the ultimate stopword filter and guarantees that your extracted topics are always "Things" or "Actions".

menyediakan = sedia



3\. Multi-Word Expression (MWE)

* A simple preprocessing step using standard regex or Malaya entities can tie common phrases together with an underscore (sumber\_manusia). This way, algorithms like NMF treat it as a single token, which dramatically improves readability.
* "sumber manusia" (human resources) is currently treated as two independent words: "sumber" and "manusia".





##### **REPLACE**



1. ambik - ambil
2. adoi - aduh
3. akak - kak
4. ***asal - kenapa (MULTIPLE MEANING)***
5. apasal - kenapa
6. aje - sahaja
7. ***bantai - hentam (MULTIPLE MEANING)***
8. bagitahu - beritahu
9. bagitau - beritahu
10. bagitahulah - beritahulah
11. bagitaulah - beritahulah
12. camne - macam mana
13. camni - macam ini
14. camtu - macam itu
15. cenggitu - macam itu
16. cenggini - macam ini
17. cite - cerita
18. citer - cerita
19. ***cokia - biasa-biasa (MULTIPLE MEANING)***
20. diorang - mereka
21. dia orang - mereka
22. dah - sudah (SF)
23. duk - duduk (SF)
24. dulu - dahulu (SF)
25. gi - pergi
26. gak - juga
27. gostan - undur
28. je - sahaja
29. jom - mari
30. jomlah - marilah
31. jap - sekejap
32. kejap - sekejap
33. kat - dekat
34. karok - karaoke
35. keuangan - kewangan
36. korang - kamu semua
37. kitorang - kami
38. kitaorang - kami
39. kita orang - kami
40. kecek - cakap
41. kot - agaknya
42. ko - kau
43. kasi - beri
44. kesehatan - kesihatan
45. last-last - akhirnya
46. last last - akhirnya
47. lu - dahulu
48. laki - lelaki
49. leklok - elok-elok
50. macam contoh - contohnya
51. ***nego - runding (ORIGINAL WORD: NEGOTIATE, SAMA MAKSUD DENGAN ORIGINAL ENGLISH WORD CUMA SHORTENED)***
52. ni - ini
53. ***nak - mahu (MULTIPLE MEANING)***
54. naklah - mahu
55. no hal - tiada masalah
56. ***ngam - sesuai (MULTIPLE MEANING)***
57. ***ngam-ngam - tepat-tepat (MULTIPLE MEANING)***
58. nak-nak - lebih-lebih lagi
59. ok - okey
60. okay - okey
61. otomatik - automatik
62. pas - selepas
63. pastu - selepas itu
64. pi - pergi
65. payah - susah
66. pape - apa-apa
67. ***rilek - bertenang (MULTIPLE MEANING)***
68. sikit - sedikit
69. skang - sekarang
70. sat - sekejap
71. saja - sahaja
72. saje - sahaja
73. stylo - bergaya
74. sehat - sihat
75. setel - selesai
76. tu - itu
77. tak - tidak
78. takkan - tidak mungkin
79. takkanlah - tidak mungkin
80. takde - tidak ada
81. takpe - tidak mengapa
82. takyah - tidak perlu
83. tak payah - tidak perlu
84. tak payahlah - tidak perlulah
85. tapi - tetapi (SF)
86. tau - tahu
87. terkejut beruk - sangat terkejut
88. terer - hebat
89. usya - tengok
90. usya-usya - tengok-tengok
91. uang - wang

##### 

##### **ENGLISH**



1. anyways - walau bagaimanapun
2. and - dan
3. alright - baiklah
4. best - seronok (DIFF MEANING THAN ORIGINAL ENGLISH WORD)
5. blur - keliru (DIFF MEANING THAN ORIGINAL ENGLISH WORD)
6. boring - bosan
7. bimbo - wanita yang dianggap menarik secara fizikal tetapi kurang cerdik atau bodoh
8. chill - santai (DIFF MEANING THAN ORIGINAL ENGLISH WORD)
9. frust - kecewa (ORIGINAL WORD: FRUSTRATED, SAMA MAKSUD DENGAN ORIGINAL WORD CUMA SHORTENED)
10. for example - sebagai contoh
11. mostly - kebanyakan
12. member - kawan (DIFF MEANING THAN ORIGINAL ENGLISH WORD)
13. outstation - kerja di luar kawasan (DIFF MEANING THAN ORIGINAL WORD)
14. port - tempat (DIFF MEANING THAN ORIGINAL ENGLISH WORD)
15. power - hebat (DIFF MEANING THAN ORIGINAL ENGLISH WORD)
16. relax - bertenang (SAME MEANING BUT SOMETIMES USED IN DIFFERENT MEANING THAN THE ORIGINAL WORD)
17. so - jadi
18. still - masih
19. sound - marah (DIFF MEANING THAN ORIGINAL ENGLISH WORD)
20. serious - serius
21. settle - selesai
22. terror - hebat (DIFF MEANING THAN ORININAL ENGLISH WORD)
23. vibe - suasana (SAME MEANING BUT SOMETIMES USED IN DIFFERENT MEANING THAN THE ORIGINAL WORD)

##### 

##### **REMOVE (FILLERS/KATA PENGISI,INTERJECTIONS/KATA SERU, DISCOURSE MARKERS/PENANDA WACANA, VOCATIVES, PHATIC)**



1. ah
2. aa
3. ahh
4. **alah (MULTIPLE MEANING)**
5. aiyoh
6. aiyaya
7. bruh
8. beb
9. babe
10. ceh
11. ehh
12. err
13. emm
14. fuh
15. ha
16. haa
17. halah
18. hekeleh
19. hm
20. hmm
21. ***kan (MULTIPLE MEANING)***
22. lah
23. mat
24. ouh
25. peh
26. uhm
27. wuih





##### **MULTIPLE MEANING**



1. asal - kenapa
* asal tak cakap awal-awal weh
* buat kerja ni asal boleh je
* asal jadi cukup la
* kau asal mana



2\. bantai - memukul/hentam/asal boleh

* dia bantai tidur sampai petang
* pencuri tu kena bantai dengan orang kampung
* mari kita bantai makanan ni
* aku bantai je jawapan exam tadi



3\. bapak - sangat

* roti gardenia kat Sarawak ni bapak mahal
* bapak ah! (biar betul/kalau replace sangat pelik)
* bapak ah, kau ni (bole replace biar betul sbb kadang bapak ah bukan maksud hebat pon kadang cm shock or reaction biar betul)
* bapak dia tak kasi bawak kereta



4\. bagi - beri/untuk

* dia bagi roti tu dekat kawan dia
* bantuan ini adalah bagi golongan yang memerlukan
* bagi aku, dia tak sesuai pegang jawatan tu



5\. cam - macam

* dia cam tak suka je makanan tu
* cam mana nak gerak pagi-pagi ni
* aku macam cam muka dia, sebab selalu nampak



6\. cokia - biasa-biasa/lekeh/tidak berkualiti

* cuma kamera aku je lah, masa tu agak cokia sikit
* masa tu agak cokia sikit
* barang cokia macam ni pun kau nak beli
* aku ni cokia je bab kereta ni



**7.** gila/giler - sangat

* dekat kedai tadi ramai gila orang
* dia tu gila makan
* dengar cite, pakcik dia gila



8\. koyak - terasa hati

* baju dia koyak
* kertas cacatan meeting haritu koyak
* dia koyak la tu
* sikit-sikit nak koyak



9\. nak - hendak/mahu

* aku rindu gila nak pergi sarawak
* so, still tak nak jemput aku eh
* macam-macam kerajaan negeri buat nak meningkatkan liputan perkhidmatan pembentungan
* lagi senang lah aku nak travel



10\. ngam - sesuai/setuju (ADA DLM KAMUS DEWAN TAPI DIANGGAP BP - BAHASA PERCAKAPAN/TAK FORMAL)

* kasut ni ngam dengan kaki aku
* aku tengok dia kau dengan dia memang ngam la
* malam ni kita lepak mana? okay, ngam!



11\. ngam-ngam - tepat-tepat/cukup-cukup/nyaris (ADA DLM KAMUS DEWAN TAPI DIANGGAP BP - BAHASA PERCAKAPAN/TAK FORMAL)

* aku sampai kat stesen bas ngam-ngam bas tu nak gerak
* duit poket aku ada 10 ringgit je, ngam-ngam buat bayar makan tengah hari
* dia bawak kereta laju sangat, tadi ngam-ngam je nak langgar divider



12\. pasal - tentang

* saya tak tahu maklumat pasal ni
* dia selalu cari pasal dengan saya



13\. rilek - bertenang/berehat/bersahaja/santai-santai

* kau rilek dulu, jangan marah-marah kita bincang elok-elok
* hujung minggu ni aku nak rilek kat rumah je
* dia tu orangnya rilek je, kalau kena marah pon dia senyum
* rilek-rilek sudah, benda boleh settle buat apa nak gaduh







**FILLERS**



1. eh
* to be honest, eh. aku tak pernah stereotype dekat orang sarawak (boleh remove)
* pastu akak tu macam, eh! (kalau remove jadi pelik)
* eh, tapi serious (boleh remove)
* tapi satu hari ni aku ada terjumpa roti gardenia kat satu pasaraya kat sarawak ni, aku macam, eh, ada rupanya (kalau remove jadi pelik)



2\. peh

* tapi bila aku belek-belek harga dia, peh! (kalau remove ayat jadi tergantung and tkde sentiment value)



3\. kan - bukan @ (remove)

* tapi kan, walaupun aku tak pandai cakap Sarawak... (remove)
* cakap pasal nyanyi kan, aku baru tahu satu benda ni bila aku belajar kat Sarawak.. (remove)
* macam over lah pula kan? (remove)
* Sampai kan kalau kau naik flight nak ke Sarawak tu... (remove)
* cantik kan? (bukan)
* pertama sekali, sudah tentu, untuk ibu bapa saya, kan? (bukan)







##### **REMOVE WORDS FROM THE LIST**



1. kena -> perlu (kalau tuka meaning ayat jadi lain)
* dia kena denggi
* kita kena mulakan meeting
* Alhamdulillah nak kena panggil YB ke sepanjang 2 jam ni



2\. bagi -> beri/untuk

* dia bagi roti tu dekat kawan dia
* bantuan ini adalah bagi golongan yang memerlukan
* bagi aku, dia tak sesuai pegang jawatan tu



3\. bapak -> sangat

* bapak mahal sia
* bapak dia tak balik lagi (kalau tukar bapak -> sangat jadi pelik ayat)



4\. cam -> macam

* dia cam tak suka je makanan tu
* cam mana nak gerak pagi-pagi ni
* aku macam cam muka dia, sebab selalu nampak (kalau tukar jadi pelik ayat



5\. gila/giler -> sangat (kalau replace ayat yg betul2 meaning gila jadi pelik)



6\. koyak -> terasa hati (kalau replace ayat yg betul2 meaning koyak jadi pelik)

* kertas cacatan meeting haritu koyak



7\. pasal -> tentang (kalau replace ayat yg betul2 meaning pasal jadi pelik)

* dia selalu cari pasal dengan saya



8\. lepas -> selepas (lepas ngn selepas ada different meaning, kadang lepas tu bukan singkatan selepas tapi maksud yang lain)





###### FILLERS



1. pun (remove stand alone pun je kalau perkataan tu mmg ada pun jgn remove eg: walaupun, sekalipun - questionable gak nak remove sbb kadang ayat tu kalau remove bunyi pelik)
2. aduh (remove from the list sbb perkataan ni diiktiraf sebagai tatabahasa yg sah and also brings sentiment)
3. alamak (ada dalam kamus)
4. apa ke
5. cis (ada dalam kamus)
6. eh (MULTIPLE MEANING - ada dalam kamus)
7. oh (ada dalam kamus)
8. weh (ada dalam kamus)
9. wah (ada dalam kamus)
10. kan (most ayat kalau remove jadi pelik ayat)







NORMALIZATION - PERKATAAN SINGKATAN/EJAAN DAH DIMODIFIED

