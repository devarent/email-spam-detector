# TUGAS PRAKTIKUM: PERCOBAAN MENDETEKSI EMAIL SPAM
- **Mata Kuliah:** Keamanan Berbasis Sistem Cerdas
- **Kelas:** TIFM
- **Dosen Pengajar:** HERIANTO, S.Pd., M.T.
- **Nama Mahasiswa:** Muhammad Arya Fatthurahman
- **NIM:** 2023230006

---
### Referensi Sumber Materi:
1. **Modul Praktikum Utama:** *Spam Mail Prediction using Machine Learning with Python* (TF-IDF & Logistic Regression).
2. **Materi Tambahan Keamanan Siber:** *AI dalam Cyber Security* (Onno W. Purbo) — Studi Kasus Klasifikasi Email Spam, Algoritma Naïve Bayes, Random Forest, Analisis Isu Kritis (*False Positives*, *Data Imbalance*, dan *Obfuscation Attacks*).

---
### Daftar File dalam Arsip:
1. `SpamMailPrediction.ipynb` - Jupyter Notebook komprehensif yang telah di-RUN lengkap dengan visualisasi EDA, komparasi 3 model (Logistic Regression, Multinomial Naive Bayes, Random Forest), confusion matrix per model, dan uji inferensi (termasuk *obfuscation attack*).
2. `SpamMailPrediction.html` - Halaman web mandiri hasil ekspor notebook yang dapat langsung dibuka pada browser untuk memeriksa hasil tanpa perlu membuka Jupyter.
3. `SpamMailPrediction.py` - Kode sumber Python mandiri (standalone CLI).
4. `mail_data.csv` - Dataset pesan email spam dan ham (5.572 baris data).
5. `README.md` - Dokumentasi identitas mahasiswa, metodologi, dan analisis komparasi model.

---
### Ringkasan Hasil Eksperimen & Komparasi Model:

| Model Algoritma | Akurasi Data Latih | Akurasi Data Uji | Precision (Ham) | Recall (Ham) | F1-Score (Ham) | Catatan Keamanan Siber (Onno W. Purbo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 96.77% | 96.59% | 96.38% | 99.90% | 98.11% | Sesuai modul utama; seimbang dan konvergen cepat pada teks berdimensi tinggi. |
| **Multinomial Naïve Bayes** | 97.42% | 97.04% | 96.89% | 99.79% | 98.32% | Algoritma klasik filter spam; sangat efisien, cepat, dan presisi tinggi. |
| **Random Forest** | 100.00% | 97.49% | 97.38% | 99.79% | 98.57% | Algoritma ensemble; akurasi uji tertinggi, tangguh terhadap variasi pola teks. |

---
### Pembahasan Isu Kritis (Cybersecurity Context):
1. **False Positives:** Meminimalkan *false positive* adalah prioritas utama dalam email security agar email sah dari klien, pimpinan, atau transaksi finansial tidak diblokir.
2. **Data Imbalance:** Dataset memiliki rasio 86.6% Ham vs 13.4% Spam. Pembobotan TF-IDF dan metrik F1-score memastikan model tidak bias pada kelas mayoritas.
3. **Evasion / Obfuscation:** Penyerang sering menyamarkan kata kunci (`FR33`, `W1N`, `cl!ck`). Solusi modern memerlukan penggabungan filter ML dengan analisis heuristik (*rule-based*) dan reputasi domain email.
