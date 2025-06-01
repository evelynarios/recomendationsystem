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
- Sumber data: [[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)](https://www.kaggle.com/datasets/fuzzywizard/movielens-100k-small-dataset/data)
- Terdapat dua file, yaitu movies.csv dan ratings.csv, dengan informasi masing-masing:
  ![image](https://github.com/user-attachments/assets/2aa959dc-eaf8-4a5b-b3f9-8e6a842d7e24)<br>
  Dataset movies.csv terdiri dari 9.742 baris dan 3 kolom yaitu movieId, title, dan genres. Fungsi head() menampilkan 5 baris pertama yang menunjukkan bahwa setiap film memiliki ID unik, judul yang mencakup tahun rilis, serta genre yang dituliskan dalam format teks dengan pemisah tanda |. Fungsi info() menunjukkan bahwa tidak terdapat nilai kosong pada ketiga kolom dan tipe data yang digunakan mencakup integer untuk ID film serta objek (string) untuk judul dan genre.<br>
  ![image](https://github.com/user-attachments/assets/19706fd0-5345-4161-85a6-f873ab4520bb)<br>
  Dataset ratings.csv berisi 100.836 baris dan 4 kolom, yaitu userId, movieId, rating, dan timestamp. Data ini merekam aktivitas pengguna dalam memberikan penilaian terhadap film tertentu, di mana rating memiliki tipe data float dan mencerminkan skor yang diberikan, sementara timestamp menunjukkan waktu penilaian dalam format Unix time. Hasil dari head() memperlihatkan bahwa satu pengguna dapat memberikan rating untuk beberapa film, dan info() mengonfirmasi bahwa tidak ada data yang hilang di seluruh kolom.


### Fitur
#### movies.csv:
- `movieId`: ID film
- `title`: Judul film
- `genres`: Genre film (bisa lebih dari satu genre)
#### ratings.csv:
- `movieId`: ID film
- `rating`: Rating (0.5 - 5.0)
- `userId`: ID pengguna
- `timestamp`: waktu

### Cek Missing Values
![image](https://github.com/user-attachments/assets/527d2f6e-421e-454d-9869-3d8a667e2ca0)<br>
Pada data baik di movies.csv maupun di ratings.csv, tidak ditemukan missing values.

## Exploratory Data Analysis (EDA)
![image](https://github.com/user-attachments/assets/a817df7f-7865-4578-ab91-f552e0b9efbd)

Grafik di atas menunjukkan distribusi rating dari seluruh pengguna terhadap film dalam dataset MovieLens 100K. Terlihat bahwa sebagian besar rating berada pada rentang 3.0 hingga 4.0, dengan puncak distribusi pada rating 4.0. Hal ini menunjukkan bahwa pengguna cenderung memberikan penilaian yang cukup positif terhadap film yang mereka tonton. Sementara itu, rating rendah seperti 0.5 hingga 1.5 relatif jarang diberikan. Distribusi ini penting untuk diperhatikan karena dapat memengaruhi akurasi model dalam membedakan film yang benar-benar disukai dan yang kurang diminati.<br>

![image](https://github.com/user-attachments/assets/9c82ec68-e52b-465c-89f6-48cb973fc2f6)
Grafik di atas menggambarkan distribusi jumlah film berdasarkan genre yang ada di dalam dataset. Genre Drama dan Comedy merupakan dua genre paling dominan, masing-masing memiliki lebih dari 4.000 dan 3.700 film. Hal ini menunjukkan bahwa film dengan genre tersebut lebih banyak diproduksi atau lebih sering dikategorikan dalam database. Di sisi lain, genre seperti Film-Noir, IMAX, dan Western termasuk kategori dengan jumlah film paling sedikit. Informasi ini penting untuk diperhatikan karena ketidakseimbangan distribusi genre dapat memengaruhi performa sistem rekomendasi, terutama jika genre minoritas kurang terwakili dalam proses pelatihan model.<br>

![image](https://github.com/user-attachments/assets/9fd31685-e51d-4642-ab0a-6024c9a5c268)
Grafik di atas menunjukkan daftar 10 film yang paling banyak menerima rating dari pengguna. Film Forrest Gump (1994) menempati posisi pertama dengan jumlah rating tertinggi, diikuti oleh The Shawshank Redemption (1994) dan Pulp Fiction (1994). Mayoritas film dalam daftar ini berasal dari era 1990-an, yang menunjukkan bahwa film-film klasik dari dekade tersebut memiliki daya tarik dan popularitas tinggi yang bertahan lama. Jumlah rating yang tinggi dapat mencerminkan seberapa dikenal dan disukai sebuah film.


## Data Preparation

Tahapan *data preparation* dilakukan untuk memastikan data dapat digunakan secara optimal dalam proses pemodelan sistem rekomendasi. Proses ini mencakup pembersihan, transformasi, dan encoding data agar sesuai dengan format masukan model.

Berikut adalah tahapan-tahapan data preparation yang dilakukan secara berurutan:

### 1. Menggabungkan Dataset
Dua dataset yang digunakan—`ratings` dan `movies`—digabungkan menggunakan kolom `movieId` sebagai kunci penghubung. Hal ini dilakukan untuk menyatukan informasi rating dari pengguna dengan metadata film seperti judul dan genre.

```python
df = pd.merge(ratings, movies, on='movieId')
```

**Alasan:**  
Menggabungkan data diperlukan agar setiap baris memiliki informasi lengkap tentang rating yang diberikan oleh pengguna pada film tertentu, beserta informasi genre dan judul film tersebut.

### 2. Pra-pemrosesan Kolom Genre
Kolom `genres` berisi genre film yang dipisahkan dengan tanda pipe (`|`). Kolom ini ditransformasikan ke format teks yang bisa diproses oleh model content-based filtering, yaitu dengan mengganti `|` menjadi spasi dan menyimpannya di kolom baru `genres_processed`.

```python
df['genres_processed'] = df['genres'].str.replace('|', ' ')
```

**Alasan:**  
Transformasi ini diperlukan untuk memungkinkan penggunaan genre sebagai representasi teks (*text feature*) yang akan digunakan dalam pemodelan berbasis konten.

### 3. Encoding ID Pengguna dan Film
Nilai `userId` dan `movieId` diubah menjadi indeks numerik berturut-turut (mulai dari 0) yang dapat digunakan sebagai input embedding dalam model.

```python
user_ids = df['userId'].unique().tolist()
movie_ids = df['movieId'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
df['user'] = df['userId'].map(user2user_encoded)
df['movie'] = df['movieId'].map(movie2movie_encoded)
```

**Alasan:**  
Model machine learning tidak dapat bekerja langsung dengan ID kategorikal dalam bentuk string atau angka acak. Oleh karena itu, ID perlu di-*encode* ke dalam indeks integer untuk memungkinkan pembuatan embedding dan pelatihan model.

### 4. Normalisasi Nilai Rating
Nilai rating yang awalnya berada pada skala 0.5 hingga 5 diubah ke dalam skala 0 hingga 1 dengan rumus:

```python
df['rating_norm'] = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
```

**Alasan:**  
Normalisasi nilai rating diperlukan agar model lebih stabil saat melakukan prediksi, terutama ketika menggunakan fungsi aktivasi seperti sigmoid atau relu. Skala yang konsisten (0–1) juga membantu mempercepat konvergensi model.

### 5. Split Data Train dan Validasi
Data diacak (*shuffled*) dan dibagi menjadi dua bagian: 90% untuk data pelatihan (*training*) dan 10% untuk data validasi (*validation*).

```python
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(0.9 * len(df))
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]
```

**Alasan:**  
Pemecahan data menjadi train dan validasi penting untuk mengevaluasi kinerja model pada data yang tidak dilihat sebelumnya. Proses shuffle memastikan distribusi data acak dan tidak bias terhadap urutan aslinya.

### 6. Mempersiapkan Input dan Target
Variabel input (`x_train`, `x_val`) terdiri dari pasangan `user` dan `movie`, sementara target (`y_train`, `y_val`) adalah `rating_norm`.

```python
x_train = train_df[['user', 'movie']].values
y_train = train_df['rating_norm'].values
x_val = val_df[['user', 'movie']].values
y_val = val_df['rating_norm'].values
```

**Alasan:**  
Persiapan input dan target ini penting agar data siap digunakan dalam proses pelatihan model pembelajaran mesin yang akan mempelajari hubungan antara pengguna, film, dan nilai rating.

#### Keseluruhan:
- Genre perlu dikonversi ke vektor numerik agar dapat dihitung kemiripannya.
- Model neural network membutuhkan input numerik.
- Normalisasi membantu konvergensi model.
- Validation set digunakan untuk mengevaluasi generalisasi model.

## Modeling

### 1. Content-Based Filtering (CBF)

#### Penjelasan Metode

Content-Based Filtering merekomendasikan item yang mirip dengan item yang disukai pengguna berdasarkan informasi konten. Dalam proyek ini, kami menggunakan informasi dari fitur `genres_processed` yang telah diproses sebelumnya dari data genre film.

Cara Kerja:

1. **Ambil Data Unik Film**  
   Mengambil data `movieId`, `title`, dan `genres_processed`, serta menghapus duplikat.

2. **Ekstraksi Fitur dengan TF-IDF**  
   Menggunakan `TfidfVectorizer` untuk mengubah teks genre menjadi representasi vektor berbobot. Stop words dalam bahasa Inggris diabaikan.

3. **Perhitungan Kemiripan Cosine**  
   Menghitung kesamaan antar film menggunakan cosine similarity dari vektor TF-IDF.

4. **Mapping Judul ke Indeks**  
   Membuat mapping dari judul film ke indeks baris untuk akses cepat.

5. **Fungsi Rekomendasi**  
   Fungsi `recommend_movies_cbf()` menerima input judul film dan menghasilkan rekomendasi *top-N* film berdasarkan kemiripan konten.

#### Fungsi `recommend_movies_cbf`

```python
def recommend_movies_cbf(title, top_n=10):
    if title not in indices:
        return f"Film '{title}' tidak ditemukan."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # exclude itself
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres_processed']]
```

##### Penjelasan Parameter:

- `title` : Judul film input dari pengguna untuk dijadikan dasar rekomendasi.
- `top_n` : Jumlah film yang direkomendasikan (default = 10).
- `cosine_sim` : Matriks kemiripan cosine antar film berdasarkan TF-IDF genre.
- `indices` : Mapping judul film ke indeks baris dalam dataset `movies`.

#### Kelebihan CBF

- Tidak membutuhkan data pengguna lain (cocok untuk sistem baru atau pengguna baru).
- Rekomendasi konsisten dengan preferensi sebelumnya.
- Mudah dijelaskan karena berbasis fitur konten.

#### Kekurangan CBF

- Cenderung merekomendasikan item yang terlalu mirip (kurang variasi).
- Tidak dapat menangkap *trend* atau film populer dari komunitas.
- Kualitas rekomendasi terbatas pada informasi konten yang tersedia (hanya berdasarkan genre).

---

## 2. Collaborative Filtering (CF)

### Penjelasan Metode

Collaborative Filtering adalah metode yang merekomendasikan item berdasarkan interaksi pengguna lain yang memiliki pola mirip. Dalam proyek ini, pendekatan yang digunakan adalah *Neural Collaborative Filtering* berbasis embedding menggunakan TensorFlow/Keras.

Cara kerja:

1. **Encoding User dan Movie**
   - Setiap `userId` dan `movieId` diubah ke indeks integer agar bisa digunakan dalam layer embedding.
   - Mapping disimpan dalam `user2user_encoded` dan `movie2movie_encoded`.

2. **Normalisasi Rating**
   - Rating dinormalisasi ke rentang [0, 1] agar sesuai dengan output sigmoid model.
   - Rumus:  
     \[
     rating\_norm = \frac{rating - min\_rating}{max\_rating - min\_rating}
     \]

3. **Split Data**
   - Dataset diacak dan dibagi menjadi 90% data pelatihan dan 10% data validasi.

4. **Model Neural Collaborative Filtering**

```python
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        ...
        self.user_embedding = layers.Embedding(...)
        self.movie_embedding = layers.Embedding(...)
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        ...
        dot = tf.reduce_sum(user_vector * movie_vector, axis=1)
        return tf.squeeze(tf.clip_by_value(dot + user_bias + movie_bias, 0, 1), axis=1)
```

Model ini mempelajari representasi vektor (embedding) dari pengguna dan film, serta bias-nya, untuk memprediksi seberapa besar kemungkinan seorang user menyukai sebuah film.

5. **Training**

Model dilatih dengan konfigurasi berikut:
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning_rate = 0.001)
- **Epochs**: 15
- **Batch size**: 64

```python
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=15,
    verbose=2
)
```

---

#### Penjelasan Parameter:

- `EMBEDDING_SIZE = 50`: Dimensi vektor embedding untuk user dan film.
- `loss='mse'`: Digunakan karena prediksi berupa rating kontinu (0-1).
- `clip_by_value`: Output dibatasi agar tetap dalam rentang valid [0, 1].
- `Adam`: Optimizer adaptif yang efisien untuk pembelajaran berbasis mini-batch.

---
### Fungsi `recommend_movies_cf(user_id, top_n=10)`

```python
def recommend_movies_cf(user_id, top_n=10):
    if user_id not in user2user_encoded:
        return f"User ID {user_id} tidak ditemukan."

    user_enc = user2user_encoded[user_id]
    watched_movie_ids = df[df['userId'] == user_id]['movieId'].tolist()

    movies_not_watched = [m for m in movie_ids if m not in watched_movie_ids]
    movies_not_watched_encoded = [movie2movie_encoded[m] for m in movies_not_watched]

    user_array = np.array([user_enc] * len(movies_not_watched_encoded))
    movie_array = np.array(movies_not_watched_encoded)
    input_array = np.stack([user_array, movie_array], axis=1)

    preds = model.predict(input_array, batch_size=128).flatten()
    top_indices = preds.argsort()[-top_n:][::-1]
    recommended_movie_ids = [movies_not_watched[i] for i in top_indices]

    return movies[movies['movieId'].isin(recommended_movie_ids)][['title', 'genres_processed']]
```

#### Parameter
- `user_id` (`int`): ID pengguna asli (bukan ID yang sudah di-encode). Ini digunakan untuk mencocokkan dan mencarikan rekomendasi untuk pengguna tersebut.
- `top_n` (`int`, default = 10): Jumlah film teratas yang ingin direkomendasikan berdasarkan skor prediksi tertinggi.

### Kelebihan CF

- Mampu menangkap pola kompleks berdasarkan perilaku pengguna lain.
- Tidak tergantung pada fitur konten (genre, deskripsi, dll).
- Potensial untuk memberikan rekomendasi yang lebih bervariasi dan mengejutkan.

### Kekurangan CF

- Membutuhkan data interaksi historis yang cukup (tidak cocok untuk pengguna atau item baru — cold start).
- Kualitas rekomendasi bergantung pada kepadatan matriks interaksi.
- Lebih kompleks dalam pelatihan dan tuning dibanding CBF.

---

## Evaluation of Recommendation Models

### Evaluation: Content-Based Filtering (CBF)

#### Metrik Evaluasi
Untuk mengevaluasi performa model Content-Based Filtering (CBF), digunakan dua metrik utama:
- **Precision@K**: Mengukur proporsi item relevan dari top-K item yang direkomendasikan.
- **Mean Average Precision@K (MAP@K)**: Mengukur rata-rata presisi dari setiap rekomendasi yang relevan terhadap urutan rekomendasi.

##### Formula
- Precision@K:

$$
\text{Precision@K} = \frac{\text{Jumlah item relevan di top-K rekomendasi}}{K}
$$

- Average Precision (AP):

$$
\text{AP@K} = \sum_{i=1}^{K} \frac{\text{Jumlah item relevan hingga posisi } i}{i}
$$

- Mean Average Precision (MAP@K): rata-rata AP dari seluruh pengguna.

#### Hasil Evaluasi
Evaluasi dilakukan pada 50 pengguna acak. Hasilnya adalah:
- **Precision@5**: 0.1640
- **MAP@5**: 0.1002

Model CBF cukup baik dalam memberikan rekomendasi yang relevan untuk sejumlah kecil item teratas, namun masih bisa ditingkatkan dengan mempertimbangkan fitur lain atau metadata tambahan.


### Evaluation: Collaborative Filtering (CF)

#### Metrik Evaluasi
Model Collaborative Filtering (CF) dinilai menggunakan:
- **Root Mean Squared Error (RMSE)**: Mengukur perbedaan antara rating aktual dan rating hasil prediksi.

##### Formula
$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}
$$

#### Hasil Evaluasi
- **RMSE**: 1.0075

Nilai RMSE yang rendah menunjukkan bahwa model cukup akurat dalam memprediksi rating pengguna. Namun, nilai ini juga mengindikasikan bahwa masih ada ruang untuk perbaikan, terutama dalam menangani cold-start problem atau data yang sparse.


### Visualisasi Perbandingan CBF dan CF

Untuk memberikan pemahaman lebih lanjut terhadap perbedaan pendekatan CBF dan CF, ditampilkan dua grafik:

- **CBF**: Menampilkan top-10 film yang paling mirip dengan film query (`Toy Story (1995)`) berdasarkan cosine similarity.
- **CF**: Menampilkan top-10 film yang diprediksi paling disukai oleh pengguna tertentu (`User 387`) berdasarkan skor prediksi model.

Dapat disimpulkan:
- CBF fokus pada **kemiripan konten** antar film.
- CF fokus pada **preferensi pengguna** yang dipelajari dari data historis.

Pendekatan gabungan atau hybrid dapat menjadi solusi yang baik untuk menutupi kekurangan masing-masing metode.


## Conclusion

Proyek ini mengimplementasikan dua pendekatan sistem rekomendasi, yaitu Content-Based Filtering (CBF) dan Collaborative Filtering (CF), menggunakan data MovieLens 100K.

* CBF memberikan rekomendasi berdasarkan kesamaan konten, khususnya genre film. 
Pendekatan ini efektif untuk cold-start item (film baru) dan menghasilkan rekomendasi yang sangat mirip dengan film acuan.

* CF menggunakan model neural network dengan embedding layer untuk mempelajari pola rating antar pengguna dan item. Hasilnya menunjukkan kemampuan dalam menangkap preferensi personal pengguna secara lebih fleksibel, bahkan ketika film tidak memiliki genre yang serupa.


Secara keseluruhan, CBF dan CF saling melengkapi, dan menggabungkan keduanya dalam sistem hybrid akan sangat bermanfaat untuk meningkatkan kualitas rekomendasi. Model hybrid akan mengatasi kelemahan masing-masing pendekatan, menghasilkan sistem yang lebih akurat dan adaptif terhadap kebutuhan pengguna.
