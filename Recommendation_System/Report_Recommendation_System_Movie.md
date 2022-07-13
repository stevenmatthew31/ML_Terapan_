# Laporan Proyek Machine Learning - Steven Matthew Gondowijoyo
## Project Overview - ***Movie Recommendation System***
<p align="Justify">
Industri film saat ini sudah menjadi industri yang terbilang besar. terdapat 4,734,693 judul, yang
diantaranya terdapat judul TV series, film pendek, documenter dan sebagainya. Perkembangan industi film juga berkembang pesat. Seiring dengan berkembangnya teknologi yang digunakan dalam pembuatan film. Tidak hanya dari segi plot cerita, film saat ini harus memiliki sisi visual yang membuat konsumen terkagum saat menontonnya. Saat ini menonton film merupakan suatu hiburan alternatif dalam mengusir kebosanan. Tidak sedikit juga seseorang menonton film karena memang hobi. Hal ini menuntut industri film untuk menghadapi persaingan ketat dalam menciptakan terobosan baru guna memenuhi kebutuhan konsumen yang semakin beragam.
<p align="Justify">
Selera setiap orang pasti berbeda. Seseorang bisa menyukai film berdasarkan genre, aktor atau rumah
produksi. Hal ini yang menjadi permasalahan seseorang dalam menentukan film yang sesuai dengan ekspektasi. Mengingat jumlah film yang begitu banyak dan beragam jenisnya, seseorang tentu tidak memiliki cukup waktu untuk memeriksa sinopsis atau trailer satu per satu. Belum lagi jika ada film baru yang belum diketahui judulnya. Maka dari itu harapan seseorang adalah menginginkan rekomendasi film yang sesuai harapan dari berbagai aspek dengan efektifitas waktu yang maksimal. 
  
  Referensi :
  -  [Press Room | IMDb](https://www.imdb.com/pressroom/about/) 
  -  [Movie Recommendation System using *Cosine Similarity* dan *KNN*](https://drive.google.com/file/d/1WwQhc75AhwDuU75sjStxkgJjNHeX0QjN/view?usp=sharing)
  -  [Sistem Rekomendasi Film menggunakan Metode *User Based Collaborative Filtering*](https://drive.google.com/file/d/1L29sdWJCSzZm8E1nfjoAKa8j2KdSE_v6/view?usp=sharing)

## Business Understanding
### Problem Statements

Berikut yang menjadi permasalahan pada proyek ini adalah:
- Bagaimana membuat sistem rekomendasi film?
- Apa teknik/algoritma yang digunakan dalam proyek ini?
- Bagaimana jika subjek ingin melihat berdasarkan rating?
- Bagaimana jika subjek ingin melihat berdasarkan genre?

### Goals

Berikut yang menjadi tujuan dari adanya proyek ini :
- Membuat sistem rekomendasi film yang *reliable* dan akurat.
- Menentukan teknik/algoritma yang baik sesuai pengguna.
- Membuat sistem rekomendasi film berdasarkan rating dari pengguna lain.
- Membuat sistem rekomendasi film berdasarkan genre.

    ### Solution statements
    - Menggunakan *Content Based Filtering* berdasarkan acuan yang digunakan sebagai referensi dapat dari *genre* movie sesuai dengan ketersediaannya pada dataset.
    - Menggunakan *Collaborative Filtering* berdasarkan acuan yang digunakan sebagai referensi dapat dari rating pengguna lain dari movie tersebut.

## Data Understanding
Dataset yang digunakan dalam proyek ini diambil dari Kaggle, dimana terdiri dari 4 data yaitu *links*, *movies*, *ratings*, dan *tags* yang semuanya memiliki format csv. Pada datasets ini juga masih banyak data yang *missing*. Oleh karena itu, diperlukan sebuah proses untuk mem*filter* data ini dan menghilangkan data yang *missing*. Datasets yang dipilih dapat diakses pada [*Movie Recommendation* Datasets](https://www.kaggle.com/datasets/bandikarthik/movie-recommendation-system).

Variabel-variabel pada *Movie Recommendation* dataset adalah sebagai berikut:
- link.csv
  - movieId : ID Movie
  - imdbId : ID Movie berdasarkan IMDB
  - tmdbId : ID Movie berdasarkan TMDB
- rating.csv
  - userId : ID User
  - movieId : ID Movie
  - rating : Rating Movie
  - timestamp : Timestamp yang digunakan pada Movie tersebut
- movie.csv
  - movieId : ID Movie
  - title : Judul berdasarkan ID Movie
  - genres : Genres berdasarkan judul
- tags.csv
  - userId : ID User
  - movieId : ID Movie
  - tag : Kata Kunci pada Genres Movie
  - timestamp : Timestamp yang digunakan pada Movie tersebut
- ``movie_info`` : Hasil *merge* pada link dan movie berdasarkan movieId
- ``rate_group`` : Hasil groupby yaitu mean() untuk rating berdasarkan movieId
- ``movies`` : Hasil *merge* pada ``movie_info`` dan ``rate_group``
- ``movie_clean`` : Data yang sudah dibersihkan dari *missing variable*
- ``movie_fix`` : Data yang sudah dibersihkan dari genres yang tidak ada
- ``movie_new`` : Datasets yang dipakai untuk *Content Based Filtering*
- ``user_to_user_encoded`` : Melakukan encoding userID
- ``user_encoded_to_user`` : Melakukan proses encoding angka ke ke userID
- ``movie_to_movie_encoded`` : Melakukan proses encoding movieId
- ``movie_encoded_to_movie`` : Melakukan proses encoding angka ke movieId

Untuk memahami korelasi antara variabel diatas sebagai berikut:
- |links.csv|ratings.csv|movies.csv|tags.csv|
  |:-:|:-:|:-:|:-:|
  |![link](https://drive.google.com/uc?export=view&id=1o7S0Z_fBT3AilmfKwJ8eq9QRRmGwiO49)|![rating](https://drive.google.com/uc?export=view&id=13WaCalsteAFVU8eJV8vDR2UWBVT4iZfy)|![movie](https://drive.google.com/uc?export=view&id=1rf7hmTywB6Jd66bYyyN8XRsyFVqpPpqu)|![tag](https://drive.google.com/uc?export=view&id=1Sw9UdZVjkMx611DnjvKK9POP_l7pGPyT)|
- Melakukan EDA dengan fungsi info()
  |links.csv|ratings.csv|movies.csv|tags.csv|
  |:-:|:-:|:-:|:-:|
  |![link_info](https://drive.google.com/uc?export=view&id=1sgA1NYmx5QOq4ebSz36k5UQUYWZqMkNu)|![rating_info](https://drive.google.com/uc?export=view&id=1RLMRyLx_AMnduJB_iwx7j-lRiii7qYsZ)|![movie_info](https://drive.google.com/uc?export=view&id=1M-k6Uj0rolMzYMWT8J4YqIHbP0AJkM9g)|![tag_info](https://drive.google.com/uc?export=view&id=1uHHqnHqnaLAkjGKC0coiSHKTOJEkPGB5)|
- Untuk data hasil pembersihan *missing variable*
  ![movie_clean](https://drive.google.com/uc?export=view&id=1_MYJ8X39VkzqVxQfxG2nJ8l7csllfPPm)
- Untuk data hasil pembersihan dari genres yang tidak ada
  ![movie_fix](https://drive.google.com/uc?export=view&id=15UJrwkw_MHWZMmsjj-_N5NDa8qpN6WIo)
- Untuk data fix yang digunakan dalam *Content Based Filtering*
  ![movie_clean](https://drive.google.com/uc?export=view&id=1TJoUs-jaAUXZjVDlHJkcp6Jqd7hk5iss)
- Kemudian, untuk ``user_to_user_encoded`` dan ``user_encoded_to_user`` berikut visualisasinya
  ![user](https://drive.google.com/uc?export=view&id=1KxcMifE_HVwEfCnSTVDhlYUvpQPihrhH)
- Kemudian, untuk ``movie_to_movie_encoded`` dan ``movie_encoded_to_movie`` berikut visualisasinya
  ![movie_enc](https://drive.google.com/uc?export=view&id=1nsT0u5u0v-4wZElyCj7vJwotrP0a62xt)

## Data Preparation
Pertama, karena datasets yang digunakan memiliki banyak data dan tidak cukup bersih. Sehingga, diperlukan penyesuaian sesuai dengan masing-masing _Content-based_ dan _Collaborative-based_.
- _Content-based_
  Mengambil data dari links.csv, ratings.csv dan movies.csv yang menjadi datasets yang digunakan. Namun, sebelum itu perlu dilakukan tahap *preprocessing* dimana dilakukan 
  - Penggabungan semua data sehingga didapatkan ``movie_fix``
  - Drop *Missing Variable*
  - Drop genres yang tidak ada(*no genres listed*)
  - Drop rating dibawah 2.5
- _Collaborative-based_
  - Untuk _Collaborative-based_ dilakukan proses encoding dari userId dan movieId untuk mendapatkan ``num_users`` dan ``num_movie``
  - Serta dilakukan proses fraction dengan ``frac = 0.1`` untuk mendapatkan membagi data dengan 1000. 
  ![movie_enc](https://drive.google.com/uc?export=view&id=1q5M3OSf3QXj8cGwq6hPZkyVTE57zF9W0)
  - Berikut kesimpulan dari dilakukan *Data Preparation*:
    - Memahami data rating yang kita miliki.
    - Menyandikan (encode) fitur ‘user’ dan ‘movieId’ ke dalam indeks integer. 
    - Memetakan ‘userId’ dan ‘movieId’ ke dataframe yang berkaitan
    - Mengecek beberapa hal dalam data seperti jumlah user, jumlah resto, kemudian mengubah nilai rating menjadi float.
  
## Modeling
Pada proyek ini digunakan 2 jenis algoritma, yaitu:
- **Content-Based Filtering**

    Seperti yang diketahui, pengunaan _Content-based filtering_ adalah menyeleksi items berdasarkan kesamaan fitur, seperti ``genre`` maupun ``title`` seperti pada proyek ini. Berhubung pada algoritma ini mengabaikan nilai yang diberikan pengguna, maka hal itu akan diabaikan. Namun algoritma ini cukup mudah untuk di-implementasikan mengingat tidak diperlukannya perhatian pada sisi pengguna atau user. Dengan begitu, kriteria data yang digunakan cukup mudah dan sederhana. Untuk proses filteringnya:

    1. Menggunakan TF-IDF untuk _feature extraction_ dengan ``fit`` sekaligus ``transform`` dengan acuan ``genres``
    2. Melakukan fungsi ``todense()`` untuk mengembalikan representasi matriks padat dari matriks ini.
    3. Menggunakan _cosine similarity_ yang tersedia pada _framework_ Scikit-Learn untuk menentukan jarak antar nilai pada matriks TD-IDF.
    4. Membuat sebuah "Dataframe" dari hasil _cosine similarity_ dengan masing-masing sumbu berisikan judul movie atau ``title``.
    5. Membuat fungsi untuk melakukan proses pencarian dari "Dataframe" tersebut. Dimana fungsi tersebut mencari 10 `title` dengan jarak terdekat (``k=10``) dan membuang/_drop_ judul yang menjadi parameter pencarian.
    ![Fungsi_rekomendasi](https://drive.google.com/uc?export=view&id=16hcoq-vqOvZ7pr86MyW7Pcv2ZjVJHCXf)
    Berikut merupakan hasil rekomendasi sesuai dengan title dari salah satu movie yang tentunya terdapat pada _dataset_ dimana yang memiliki genres yang sama dengan How to Train Your Dragon 2 (2014):
    
    ![Hasil Content-based](https://drive.google.com/uc?export=view&id=1X2yW5RQHwBsWz_zHa0yg3ySc7WYn0k_y)

- **Collaborative Filtering dengan Deep learning**
    
    Metode ini memperhatikan penilaian dari pengguna dan menggunakan dapat data yang sama. Kelebihan dari metode ini adalah, jika dilakukan _tuning_ parameter yang tepat dan pengunaan model yang baik, hasil yang diberikan akan sangat memuaskan. Namun kekurangan utamanya adalah sama seperti sebelumnya dimana ukuran data yang besar sangat memakan banyak biaya komputasi dalam pembuatan matriks dan sangat tidak direkomendasikan ketika pada keadaan _cold start_. Algoritma ini juga sangat sulit diterapkan mengingat kita harus menggunakan parameter dan model yang tepat. Jika tidak hasil yang diberikan akan sangat tidak relevan. Untuk tahapan dari proses ini sebagai berikut:
    1. Melakukan _encoding_ terhadap nilai pada ID User dan ID Movie. Hal ini bertujuan untuk menghindari kesalahan pada saat proses _training_ data karena data yang sebenarnya memiliki distribusi yang acak.
    2. Memasukan data hasil _encoding_ kedalam Dataframe dan diacak serta dilakukan proses fraction dengan ``frac = 0.1`` untuk mendapatkan membagi data dengan 1000. .
      ![movie_enc](https://drive.google.com/uc?export=view&id=1q5M3OSf3QXj8cGwq6hPZkyVTE57zF9W0)
    3. Membagi data menjadi X dan Y, dimana X adalah nilai hasil _encoding_ ID User dan ID Movie, sedangkan Y adalah _rating_ yang di normalisasi dengan _layer lambda_.
    4. Membagi data X dan Y menjadi _train set_ dan _validation set_ dengan rasio 80% banding 20%
    5. Membuat class model dengan tf.Keras.Model kemudian menggunakan 4 buah _embedding layer_ untuk masing-masing data ID User beserta _bias_ dan ID Movie beserta _bias_. Hasil output model adalah hasil jumlah dari _sum of product_ kedua vector ditambah beserta kedua biasnya yang dilewatkan sebuah fungsi aktivasi "Sigmoid". Pada saat pemanggilan fungsi, digunakan jumlah _embedding_ sama dengan 50.
    ![k=50](https://drive.google.com/uc?export=view&id=1c-KkG__IqEAGVUTOuH0yDQsafBHyDRhy)
    1. _Compile_ model dengan loss "Binary Crossentropy", optimizer "Adam" dan menggunakan metrics "RMSE"
    2. Lakukan proses _training_ dengan menggunakan 25 _epochs_ dan _batch size_ sebanyak 8, begitu pula dengan bagian validasinya.
    3. Kemudian untuk mencoba hasil prediksi, pertama perlu mengambil salah satu user secara acak, lalu melihat movie mana saja yang pernah ia lihat. Setelah itu, bandingkan dengan semua data dan ambil data yang tidak pernah ia lihat dan encoding semua nilai ID Movienya. ID User dan ID Movie yang belum pernah dilihat dijadikan satu seperti format X sebelumnya dengan menggunakan "np.hstack". Lalukan prediksi menggunakan data tersebut dan ambil nilai tertingginya untuk hasil rekomendasi.
    ![no8](https://drive.google.com/uc?export=view&id=1jk0nREbrPKtZ__pAgSpI3pYJkf7zi1Gh)
    Kemudian untuk hasil rekomendasi yang diberikan adalah dari salah satu user acak adalah:

    ![Hasil Colla](https://drive.google.com/uc?export=view&id=1c2QlvxbJi25iZhFnaCZhU8hfS8l7CceY)

## Evaluation
Setiap metode/algoritma yang digunakan dalam proyek ini memiliki metrik evaluasi yang berbeda-beda, berikut merupakan algoritma beserta dengan metrik evaluasinya:

- **Content-Based Filtering**
    Pada algoritma ini, digunakan evaluasi terhadap berapa persen rekomendasi yang tepat berdasarkan item yang digunakan. Pada proyek ini, kita menggunakan penulis buku sebagai acuan untuk sistem rekomendasi, maka untuk meng-evaluasinya kita perlu melihat berapa persen sistem merekomendasi penulis buku yang sesuai. Cara kerjanya sendiri sangatlah sederhana, dimana kita hanya perlu membandingkan jumlah rekomendasi yang tepat sesuai dengan penulis dari buku yang disebutkan dengan seluruh rekomendasi yang menjadi keluaran sistem rekomendasi.

- **Model-Based Collaborative Filtering**

    Pada _Model-based_ sendiri menggunakan metrik evaluasi berupa _Root Mean Squarred Error_ (RMSE) :
    ![model](https://drive.google.com/uc?export=view&id=1c-KkG__IqEAGVUTOuH0yDQsafBHyDRhy)
    ![train](https://drive.google.com/uc?export=view&id=1Scj2fCyanilfKP29xFYqQHXPY2hgmBy-)
    ![train_hasil](https://drive.google.com/uc?export=view&id=1XXrV_ucJI4CRqyxIXGJH3Y3gHm5hK_Eq)

    ![train_graph](https://drive.google.com/uc?export=view&id=1dQOmdUBlZ_eoI5KXPnoiGcOf392OjcGy)

    Dapat dilihat bahwa nilai RMSE berada di sekiat 0.238


- Maka, secara keseluruhan sistem rekomendasi film dapat dibuat dengan akurat
- Teknik/algoritma yang baik sesuai pengguna atau users terdapat 2 jenis yaitu 
  - *Content Based Filtering*
    Dimana hasil output berdasarkan genres movie yang ia sukai
  - *Collaborative Filtering*
    Dimana hasil output berdasarkan rating movie yang user lain sukai/berikan



