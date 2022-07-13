# Laporan Proyek Machine Learning -Steven Matthew Gondowijoyo

## Project Overview - *Stroke Detection*
<p align="Justify">
Saat ini gangguan neurologis sangat mempengaruhi kehidupan masyarakat pada tingkat epidemi. Penyakit stroke merupakan penyakit Neurodegeneratif yang paling sering diderita oleh pasien yang berusia diatas 60 tahun. Secara khusus, stroke adalah penyakit kronis gangguan neurologis yang berhubungan dengan hemiplegia, kurangnya keseimbangan dan gaya berjalan yang abnormal. Sebanyak 83-90% penderita mengalami gejala tersebut. [1]. Meskipun stroke merupakan penyakit yang tidak menular, namun stroke merupakan penyebab kedua kematian di dunia menurut [2]. Terlebih lagi, usia adalah faktor utama yang paling penting dalam mendeteksi stroke dan fakta bahwa populasi semakin tua, angka-angka ini dapat meningkat lebih lanjut dalam waktu yang tidak terlalu lama. Umumnya, masyarakat sekitar sering menganggap sepele gejala stroke yang dapat berakibat fatal. 
<p align="Justify">
Stroke merupakan penyakit paling umum di dunia. Terapi endovascular terhadap penderita stroke ischemic akut telah memberikan hasil selama dua decade terakhir. Namun stroke tetap menjadi penyebab paling umum dari penyakit jangka panjang dan mengkhawatirkan masyarakat [3]. Pasien stroke biasanya memiliki cacat sisa. Umumnya penderita stroke mengalami ketidakmampuan berjalan dan mengakibatkan keterbatasan pergerakan sehari-hari. Stroke iskemik akut dapat dicegah dengan pendeteksian yang lebih awal. Pendeteksian lebih awal lebih mudah untuk dilakukan terapi, penyembuhan dan pencegahan terhadap serangan stroke yang mendadak.
</p>

- <p align="Justify"> Permasalahan tersebut harus diatasi karena seperti yang disebutkan diawal jika stroke terlambat diatasi maka dapat berakibat fatal. Maka dengan hal ini, tujuan dari adanya Proyek ini adalah untuk proses pendeteksian yang singkat, terjangkau, dan nyaman namun tetap memiliki tingkat keakuratan yang tinggi, dan memiliki kontribusi yaitu membantu pencegahan penyakit stroke akut dan dapat melakukan aktivitas dengan normal serta untuk meningkatkan kesejahteraan masyarakat dengan cara melakukan pengecekan secara mandiri oleh masing-masing individu.

Referensi : [1] [A wearable inertial measurement system with complementary filter for gait analysis of patients with stroke or Parkinson’s disease](https://drive.google.com/file/d/1L0YFE8jarNO5q1hJJQhjdQkTKM-BpKM9/view?usp=sharing)
[2] [AI-based stroke
disease prediction system using real-time electromyography signals](https://drive.google.com/file/d/1CwPoM2UYVzeuFHtu67z8I97cQ4YH4BfR/view?usp=sharing)
[3] [Optimal Foot Location for Placing Wearable
IMU Sensors and Automatic Feature Extraction for Gait Analysis](https://drive.google.com/file/d/1jGhashpF9J9pAmvYF-zbo8PAkM-cQku_/view?usp=sharing)

## Business Understanding
### Problem Statements
- Bagaimana mendeteksi stroke secara dini?
- Apakah Stroke dapat diatasi?

### Goals
- Stroke dapat dideteksi secara dini dengan menggunakan bantuan dari Machine Learning dimana dengan menginputkan beberapa data yang diperlukan
- Stroke dapat diatasi apabila dapat dideteksi secara dini sehingga dapat dilakukan terapi

## Data Understanding

Seperti yang dijelaskan sebelumnya, alasan proyek ini yaitu deteksi stroke maka diambil sebuah dataset yaitu data sekunder yang berfungsi untuk *``testing``* dan *``training``*. Dataset yang dipilih dapat diakses di [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Serta pada podul ini, kita akan menggunakan proporsi pembagian sebesar 90:10 (``test_size=0.1``) dengan fungsi ``train_test_split`` dari sklearn seperti pada gambar 
![Train_Test_Split](https://drive.google.com/uc?export=view&id=1kath8iol9iyFHzFIkpQpZuKNuPr2Y728)

### Variabel-variabel pada Stroke dataset adalah sebagai berikut:
- stroke_clean : merupakan variabel yang diambil secara langsung dari data yang diupload.
- stroke_data : merupakan variabel setelah menghilangkan missing variable
- dimension : merupakan variabel pengganti dari paramater/column yang memiliki korelasi tinggi
- id : Identifikasi Unik
- gender	: "Laki-laki", "Perempuan" atau "Lainnya"
- age	: Umur dari Pasien
- hypertension	: Menjelaskan bahwa "0" jika Pasien tidak mempunyai hipertensi dan "1" jika pasien memiliki hipertensi
- heart_disease	: Menjelaskan bahwa "0" jika Pasien tidak mempunyai penyakit jantung dan "1" jika pasien memiliki penyakit jantung
- ever_married	:  Menjelaskan bahwa Pasien sudah menikah atau belum dengan memberikan respon "No" atau "Yes"
- work_type	: Menjelaskan bahwa pekerjaan mereka dengan memberikan respon
  - children : anak-anak, 
  - Govt_jov : Pegawai Pemerintah
  -  Never_worked : Tidak bekerja
  - Private : Privat
  - Self-employed : Wiraswasta
- Residence_type	:  Menjelaskan bahwa Pasien tinggal dilingkungan dengan memberikan respon
  - Rural :  Wilayah Perdesaan 
  - Urban :  Wilayah Perkotaan
- avg_glucose_level : Kadar Glukosa dalam darah
- bmi	: Indeks Bodi Massa
- smoking_status	:Menjelaskan status apakah mereka merokok atau tidak dengan memberikan respon
  - formerly smoked : Orang yang pernah menjadi perokok aktif namun telah berhenti
  - ever smoked : Orang yang pernah merokok atau menjadi perokok pasif
  - smokes : Orang yang masih menjadi perokok aktif
  - Unknown : Tidak diketahui
- stroke	:  Menjelaskan bahwa "1" jika pasien mempunyai stroke dan "0" jika tidak
![Variabel](https://drive.google.com/uc?export=view&id=1u54CYkaUDZ8H7d3lXUtwn4d_oQYinK4D)


Untuk memahami korelasi antara 15 variabel diatas, digunakan beberapa proses EDA (_Exploratory Data Analysis_)
- Data Analysis yang telah dilakukan adalah sebagai berikut: 
  - Deskripsi Variabel
    Seperti info() dan describe()
    
    **info()**            |  **describe()**
    :-------------------------:|:-------------------------:
    ![info](https://drive.google.com/uc?export=view&id=1S1G2k2aNWOcx0vziv5wAd2GTvZKzuRv6)  |  ![describe](https://drive.google.com/uc?export=view&id=1rRmCRvyWu1H3F-3aiqbfb4e4x5cz5t6k)

      
  - Missing Variabel
    Dalam proyek ini nilai yang dihilangkan adalah data BMI
    ![MissingBMI](https://drive.google.com/uc?export=view&id=1amGzaYyiZLfST4l0uVCySulN65wd2w1J)
    
    Sehingga, data awal yang berupa 5110 menjadi 4909 seusai data BMI yang "*missing*" dihilangkan
  - Univariate Analysis
    Melakukan Numerical Features dan Categorical Features serta 
    ![kategori](https://drive.google.com/uc?export=view&id=1qUNVPtnXraefz5WAHVfPBRzTn2t6F32v)
    ``gender``             |  ``ever_married``           | ``work_type``    | ``Residence_type``    | ``smoking_status``
    :-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
    ![gender](https://drive.google.com/uc?export=view&id=12lloG38gI-KA9f8Os6Jg_KAdCibC8JiI)  |  ![married](https://drive.google.com/uc?export=view&id=1zNO0YEGEP2vMmlfsiNI7AsFtN7c5TNBU) | ![work](https://drive.google.com/uc?export=view&id=1d7p6K2tCOrh9OsXSleZiHO-diNpFnGfR) | ![residence](https://drive.google.com/uc?export=view&id=1RWt9kflJ9y-pQWPkbnk5Y50iGuod_s53) | ![smoking](https://drive.google.com/uc?export=view&id=1mYidggXTNpau69IMDKNxQNikjti0P91b)
  - Multivariate Analysis
    Melakukan Numerical Features dan Categorical Features serta memvisualisasikan yaitu dengan mengamati hubungan antar fitur numerik dengan fungsi pairplot()
    ![pairplot](https://drive.google.com/uc?export=view&id=1zjRaymBSKLh60RDXF4WPW5_1Z6LMJ_B1)
  

## Data Preparation
Pada proyek ini terdapat data preparation sebagai berikut:
- Encoding fitur kategori.

  Dilakukan encoding dengan one hot encoder dengan tujuan untuk mengubah data atau variabel kategori menjadi data atau variabel ___numerik___
  ![OneHot](https://drive.google.com/uc?export=view&id=1opOLg3vXC0aiEchvZHCZSAW1IIL0e06F)
  Sehingga, variabel yang dilakukan proses encoding adalah 
  - ``gender``
  - ``ever_married``
  - ``work_type``
  - ``Residence_type``
  - ``smoking_status``
- Reduksi dimensi dengan Principal Component Analysis (PCA).
  Reduksi dengan PCA dilakukan untuk mengurangi sejumlah fitur agar menjadi 3 komponen PC. Fitur yang dikurangi adalah sebagai berikut
  ![PCA](https://drive.google.com/uc?export=view&id=1PBttmt7wK7alzEg2i5B713hO0QaN8Y6f)
  Sehingga, variabel yang dilakukan pengurangan oleh proses PCA adalah 
  - ``age``
  - ``hypertension``
  - ``heart_disease``
  - ``avg_glucose_level``
 
  Dan digantikan oleh variabel bernama ``Dimension``
- Pembagian dataset dengan fungsi ``train_test_split`` dari library sklearn
  Tujuan dari pembagian dataset agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. Hal ini berguna untuk mengevaluasi kinerja dari dataset kita. _train set_ akan digunakan untuk melatih model _machine learning_ kita, sedangkan untuk _test set_ adalah bagian dataset yang belu pernah "dilihat" oleh model dan digunakan untuk mengevaluasi kinerja dari model kita. Hal ini diperlukan dikarenakan adanya kemungkinan model machine learning kita mengalami _overfitting_, yaitu kondisi dimana model yang dikembangkan bekerja sangat baik di _train set_ namun tidak demikian pada _test set_ atau bahkan ketika pengaplikasian. Proses pembagian ini dapat dipermudah dengan penggunaan framework _Scikit-learn_ dan modul _train_test_split_.
  **Code ``train_test_split``**            |  **Hasil ``train_test_split``**
    :-------------------------:|:-------------------------:
  ![train_test](https://drive.google.com/uc?export=view&id=1T_LAUaKEDDlwuI5rTxyMXuEki9NXCe7D) |   ![hasil_traintest](https://drive.google.com/uc?export=view&id=1Ilx4ys1jJzQ-aqi5lg8nnvR2_fcYzTuz)
- Standarisasi.

  ![hasil_traintest](https://drive.google.com/uc?export=view&id=1Qih2Gl6SnkVkDNnh8DNgyEPX3t_b9aBQ)

  Standarisasi untuk mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1 agar siap dilatih
  

## Modeling
Pada tahap ini, saya mengembangkan model machine learning dengan tiga algoritma yaitu 
- K-Nearest Neighbor
  Kita menggunakan k = 10 tetangga dan metric Euclidean untuk mengukur jarak antara titik. Pada tahap ini kita hanya melatih data training dan menyimpan data testing untuk tahap evaluasi yang akan dibahas di Modul Evaluasi Model.
- Random Forest
  Mengimpor RandomForestRegressor dari library scikit-learn. Anda juga mengimpor mean_squared_error sebagai metrik untuk mengevaluasi performa model. Selanjutnya, Anda membuat variabel RF dan memanggil RandomForestRegressor dengan beberapa nilai parameter. Berikut adalah parameter-parameter yang digunakan:
  - n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=50.
  - max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
  - random_state: digunakan untuk mengontrol random number generator yang digunakan. 
  - n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.
- Boosting Algorithm
  Algoritma boosting terdiri dari dua metode yaitu Adaptive boosting dan Gradient boosting
  
  Parameter-parameter yang digunakan pada kode proyek saya adalah
  - learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting
  - random_state: digunakan untuk mengontrol random number generator yang digunakan.

--
- Kekurangan dari KNN adalah jika dihadapkan pada jumlah fitur atau dimensi yang besar
- Kelebihan dari Random Forest adalah algoritma yang cukup sederhana tetapi memiliki stabilitas yang mumpuni. 
- Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest.
- Kemudian, jika dilihat dalam proyek saya juga terdapat perbandingan dari MSE pada 3 algoritma yang digunakan. Dimana yang memiliki model terbaik adalah **_Random Forest_**

## Evaluation
Pada tahap ini metrik evaluasi yang digunakan dari ketiga model tersebut adalah MSE (*Mean Squared Error*)

Sebelumnya juga dilakukan proses scaling fitur agar skala antara data latih dan data uji sama dan kita bisa melakukan evaluasi.

Dihadirkan juga pengujian pada model yang telah digunakan dalam proyek ini sebagai berikut
| | train | test |
| ----------- | :---------: | ----------: |
| **KNN** | 0.000036 | 0.000051 |
| **RF** | 0.000008 | 0.00005 |
| **Boosting** | 0.000043 | 0.000048 |

Serta, untuk pengvisualisasian adalah sebagai berikut
  
 ![MSE](https://drive.google.com/uc?export=view&id=1EvwptvV5xRwdzVOj30qwtgd1G0XR_cZP)

Sehingga, dari tabel dan gambar diatas yang memiliki model terbaik adalah ___**Random Forest**___

---

- MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.
- Proyek ini juga bermanfaat bagi pasien stroke karena dapat menyediakan deteksi stroke secara dini yang lebih efektif.
- Dalam sebuah prediksi di dunia kedokteran, nilai akurasi perlu sangat diperhatikan karena berhubungan langsung dengan nyawa seseorang. Sehingga, dengan adanya proyek ini yang memiliki akurasi tinggi dengan algoritma *Random Forest* diharapkan dapat membantu meningkatkan kesejahteraan masyarakat.
