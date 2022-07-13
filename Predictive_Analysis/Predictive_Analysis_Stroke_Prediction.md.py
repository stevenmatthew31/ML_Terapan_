# -*- coding: utf-8 -*-
"""Predictive Analysis

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z0X74yPTzui6F6IguhxBFQljYEaFuRFN
"""

import pandas as pd

"""### Data Loading"""

from google.colab import files
uploaded = files.upload()

stroke_clean = pd.read_csv('/content/healthcare-dataset-stroke-data.csv')
# Menghilangkan ID pada dataset
stroke_clean

"""### Deskripsi Variabel"""

stroke_clean.info()

"""### Menangani Missing Varible"""

bmi = (stroke_clean.bmi.isnull()).sum()
 
print("Nilai 0 di kolom BMI ada: ", bmi)

stroke_clean.loc[(stroke_clean['bmi'].isnull())]

print("Banyaknya Data BMI yang kosong adalah {}".format(len(stroke_clean.loc[(stroke_clean['bmi'].isnull())])))

# Drop baris dengan nilai BMI yang null
stroke_data = stroke_clean.loc[(stroke_clean[['bmi']].notnull()).all(axis=1)]
 
# Cek ukuran data untuk memastikan baris sudah di-drop
stroke_data.shape
stroke_data

stroke_data.describe()

"""### Outliers"""

import seaborn as sns

sns.boxplot(x=stroke_data['avg_glucose_level'])

sns.boxplot(x=stroke_data['bmi'])

stroke_data.describe()

"""### Univariate Analysis"""

numerical_features = ['stroke', 'hypertension', 'heart_disease', 'age', 'avg_glucose_level', 'bmi', 'id']
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

"""#### Categorical"""

feature = categorical_features[0]
count = stroke_data[feature].value_counts()
percent = 100*stroke_data[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[1]
count = stroke_data[feature].value_counts()
percent = 100*stroke_data[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[2]
count = stroke_data[feature].value_counts()
percent = 100*stroke_data[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[3]
count = stroke_data[feature].value_counts()
percent = 100*stroke_data[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

feature = categorical_features[4]
count = stroke_data[feature].value_counts()
percent = 100*stroke_data[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""#### Numerical"""

import matplotlib.pyplot as plt
import plotly.express as px

stroke_data.hist(bins=50, figsize=(20,15))
plt.show()

"""### Multivariate Analysis"""

stroke_data['stroke'].value_counts()

for i in stroke_data.drop(['stroke'], axis=1).columns:
    fig = px.histogram(stroke_data, x=i, color='stroke')
    fig.show()

# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(stroke_data, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = stroke_data.corr().round(2)
 
# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='RdBu', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""Fitur 'id' memiliki korelasi yang sangat kecil. Sehingga, fitur tersebut dapat di-drop."""

stroke_data.drop(['id'], inplace=True, axis=1)
stroke_data.head()

from sklearn.preprocessing import  OneHotEncoder
stroke_data = pd.concat([stroke_data, pd.get_dummies(stroke_data['gender'], prefix='gender')],axis=1)
stroke_data = pd.concat([stroke_data, pd.get_dummies(stroke_data['ever_married'], prefix='ever_married')],axis=1)
stroke_data = pd.concat([stroke_data, pd.get_dummies(stroke_data['work_type'], prefix='work_type')],axis=1)
stroke_data = pd.concat([stroke_data, pd.get_dummies(stroke_data['Residence_type'], prefix='Residence_type')],axis=1)
stroke_data = pd.concat([stroke_data, pd.get_dummies(stroke_data['smoking_status'], prefix='smoking_status')],axis=1)
stroke_data.drop(['gender','ever_married','work_type', 'Residence_type', 'smoking_status'], axis=1, inplace=True)
stroke_data.head()

sns.pairplot(stroke_data[['age','hypertension','heart_disease', 'avg_glucose_level']], plot_kws={"s": 4});

from sklearn.decomposition import PCA
 
pca = PCA(n_components=3, random_state=123)
pca.fit(stroke_data[['age','hypertension','heart_disease', 'avg_glucose_level']])
princ_comp = pca.transform(stroke_data[['age','hypertension','heart_disease', 'avg_glucose_level']])

pca.explained_variance_ratio_.round(3)

from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=123)
pca.fit(stroke_data[['age','hypertension','heart_disease', 'avg_glucose_level']])
stroke_data['dimension'] = pca.transform(stroke_data.loc[:, ('age','hypertension','heart_disease', 'avg_glucose_level')]).flatten()
stroke_data.drop(['age','hypertension','heart_disease', 'avg_glucose_level'], axis=1, inplace=True)
stroke_data

"""#### Train Test Split"""

from sklearn.model_selection import train_test_split
 
X = stroke_data.drop(["stroke"],axis =1)
y = stroke_data["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""#### Standarisasi"""

from sklearn.preprocessing import StandardScaler
 
numerical_features = ['bmi', 'dimension']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

"""## Model Development

### Model Development dengan KNN
"""

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
 
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""### Model Development dengan Random Forest"""

# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor
 
# buat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""### Model Development dengan Boosting Algorithm"""

from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""## Evaluasi Model"""

# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])
 
# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}
 
# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
 
# Panggil mse
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)