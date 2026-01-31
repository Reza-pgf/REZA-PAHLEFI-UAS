# UAS Natural Language Processing - Klasifikasi Sentimen

**Nama**: Reza Pahlefi  
**NIM**: 24146087  
**Program Studi**: Sistem Informasi

## Deskripsi Project

Project ini dikerjakan untuk memenuhi tugas Ujian Akhir Semester mata kuliah NLP. Sistem ini mampu mengklasifikasikan teks komentar dari dataset `dataset_sentimen (2).csv` ke dalam kategori **Positif**, **Negatif**, dan **Netral** menggunakan algoritma **Multinomial Naive Bayes**.

## Tahapan Preprocessing (Sesuai Instruksi UAS)

Program menjalankan 4 tahap utama pembersihan teks:

1. **Case Folding**: Mengubah teks menjadi huruf kecil dan menghapus simbol/angka.
2. **Tokenizing**: Memecah kalimat menjadi satuan kata (token).
3. **Stopword Removal**: Menghilangkan kata-kata umum yang tidak memiliki makna sentimen.
4. **Stemming**: Mengubah setiap kata ke bentuk kata dasarnya dengan library Sastrawi.

## Pembagian Dataset

Menggunakan fungsi `train_test_split` dari library `sklearn` dengan rasio:

- **Data Training**: 80% (1.756 sampel)
- **Data Testing**: 20% (439 sampel)

## Hasil Evaluasi Model

Model Naive Bayes yang dibangun memberikan performa yang sangat baik:

- **Accuracy**: 1.00 (100%)
- **F1-Score**: 1.00 untuk semua kategori

## Output File

Hasil pemrosesan disimpan otomatis di folder `file/`:

- `dataset_hasil_bersih.csv` (Gabungan)
- `hasil_positif.csv`
- `hasil_negatif.csv`
- `hasil_netral.csv`
