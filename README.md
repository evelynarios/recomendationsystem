# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Project Overview

Sistem rekomendasi adalah teknologi yang digunakan untuk memberikan saran produk atau konten kepada pengguna berdasarkan preferensi mereka. Dalam proyek ini, kami membangun dua sistem rekomendasi film dengan pendekatan **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)** menggunakan data MovieLens 100K.

Sistem ini penting untuk meningkatkan kepuasan pengguna dalam platform streaming dan e-commerce, karena mampu mempersonalisasi pengalaman pengguna.

Referensi:
- Ricci, F., Rokach, L., & Shapira, B. (2011). *Recommender Systems Handbook*. Springer.
- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.

## Business Understanding

### Problem Statements
1. Bagaimana memberikan rekomendasi film yang relevan kepada pengguna berdasarkan konten film yang mereka sukai?
2. Bagaimana memberikan rekomendasi yang akurat meskipun pengguna belum memberikan banyak rating?
3. Algoritma mana yang lebih baik dalam memberikan rekomendasi personal antara CBF dan CF?

### Goals
1. Membangun sistem CBF yang dapat merekomendasikan film berdasarkan genre film yang pernah disukai pengguna.
2. Membangun sistem CF yang dapat mempelajari pola dari interaksi antar pengguna dan film.
3. Mengevaluasi dan membandingkan performa kedua pendekatan.

### Solution Statements
Terdapat dua pendekatan solusi:
- **Content-Based Filtering** menggunakan cosine similarity berdasarkan genre film.
- **Collaborative Filtering** menggunakan model neural network (Embedding Layer) untuk memprediksi rating film oleh pengguna.

## Data Understanding

### Dataset
- Jumlah data: 100.000 rating oleh 943 pengguna terhadap 1.682 film.
- Sumber data: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)

### Fitur
- `userId`: ID pengguna
- `movieId`: ID film
- `rating`: Rating (0.5 - 5.0)
- `title`: Judul film
- `genres`: Genre film (bisa lebih dari satu genre)

### Exploratory Data Analysis (EDA)
- Genre terbanyak: Drama dan Komedi.
- Sebagian besar rating berada di antara 3 dan 4.
- Film populer seperti *Toy Story* memiliki rating yang tinggi.

## Data Preparation

### Teknik yang digunakan:
1. **One-hot encoding** pada genre untuk CBF.
2. **Encoding ID** (user dan movie) menggunakan `LabelEncoder` untuk CF.
3. **Normalisasi rating** menjadi 0-1 (untuk model neural network).
4. **Split data** ke training dan validation set.

### Alasan:
- Genre perlu dikonversi ke vektor numerik agar dapat dihitung kemiripannya.
- Model neural network membutuhkan input numerik.
- Normalisasi membantu konvergensi model.
- Validation set digunakan untuk mengevaluasi generalisasi model.

## Modeling

### Content-Based Filtering (CBF)

CBF merekomendasikan film berdasarkan kemiripan konten, khususnya genre. Kami menggunakan **cosine similarity** pada representasi genre yang telah diubah ke bentuk one-hot encoding.

#### Cara Kerja:
1. Genre dikonversi menjadi vektor biner (multi-hot).
2. Cosine similarity dihitung antara film input dan semua film lainnya.
3. Film dengan skor similarity tertinggi direkomendasikan.

#### Parameter Penting:
- `top_n`: Jumlah film teratas yang direkomendasikan.

### Collaborative Filtering (CF) - Neural Network Based

Model CF berbasis neural network menggunakan embedding untuk `userId` dan `movieId`. Model mempelajari pola interaksi antara pengguna dan film untuk memprediksi rating.

#### Arsitektur Model:
```python
Embedding(userId, 50)
Embedding(movieId, 50)
Dense(128, activation='relu')
Dropout(0.2)
Dense(1, activation='sigmoid')
```

#### Detail Pelatihan:
- Optimizer: Adam
- Loss Function: Binary Crossentropy / MSE
- Epochs: 15
- Batch Size: 128

#### Cara Kerja:
1. `userId` dan `movieId` diubah menjadi embedding berdimensi 50.
2. Embedding digabung, diproses oleh dense layer.
3. Output berupa prediksi rating ter-normalisasi (0-1).

## Evaluation

### Content-Based Filtering
- **Precision@5**: 0.1640
- **MAP@5**: 0.1002

### Collaborative Filtering
- **RMSE**: 1.0075

### Interpretasi Hasil:
- CBF cukup efektif untuk cold-start item (film baru), tapi tidak untuk user baru.
- CF lebih akurat secara prediktif, tapi membutuhkan data interaksi yang cukup.
- Pendekatan hybrid dapat menjadi solusi masa depan.

## Conclusion

Proyek ini mengimplementasikan dua pendekatan sistem rekomendasi, yaitu Content-Based Filtering (CBF) dan Collaborative Filtering (CF), menggunakan data MovieLens 100K.

* CBF memberikan rekomendasi berdasarkan kesamaan konten, khususnya genre film. 
Pendekatan ini efektif untuk cold-start item (film baru) dan menghasilkan rekomendasi yang sangat mirip dengan film acuan.

* CF menggunakan model neural network dengan embedding layer untuk mempelajari pola rating antar pengguna dan item. Hasilnya menunjukkan kemampuan dalam menangkap preferensi personal pengguna secara lebih fleksibel, bahkan ketika film tidak memiliki genre yang serupa.

* Evaluasi metrik menunjukkan bahwa CBF memiliki keunggulan dalam relevansi genre, sementara CF unggul dalam memahami pola interaksi pengguna.

Secara keseluruhan, CBF dan CF saling melengkapi, dan menggabungkan keduanya dalam sistem hybrid akan sangat bermanfaat untuk meningkatkan kualitas rekomendasi. Model hybrid akan mengatasi kelemahan masing-masing pendekatan, menghasilkan sistem yang lebih akurat dan adaptif terhadap kebutuhan pengguna.
