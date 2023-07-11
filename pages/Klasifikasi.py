import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import st_pages
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from st_pages import Page, Section, show_pages, add_page_title
from imblearn.under_sampling import RandomUnderSampler

class Klasifikasi:
    def __init__(self):
        self.conn = sqlite3.connect('sentiment.db')
    
    def run(self):
        add_page_title()

        st.write(
            """Pada halaman akan melakukan klasifikasi dengan menggunakan algoritma
            Multinomial Naive Bayes dan melihat hasil akurasinya."""
        )

        # Membaca data dari tabel menggunakan pandas
        df = pd.read_sql_query("SELECT * FROM hasilLabelling", self.conn)

        #mengubah tampilan dataframe
        df = pd.DataFrame(df[['tweet_clean', 'label']])

        # Menampilkan tabel menggunakan Streamlit
        st.dataframe(df)

        # tweet df
        df_train_test = df
        df_kfold = df

        train_X_kfold = df_kfold['tweet_clean']
        train_y_kfold = df_kfold['label']

        # train test split
        train_data, test_data = train_test_split(df_train_test, test_size = 0.2, random_state=42)

        train_X = train_data['tweet_clean']
        train_y = train_data['label'] 
        test_X = test_data['tweet_clean']
        test_y = test_data['label']

        # ubah kedalam bentuk vektor
        tf_vectorizer = TfidfVectorizer()
        X_train_tf = tf_vectorizer.fit_transform(train_X)
            
        tf_vectorizer_kfold = TfidfVectorizer()
        x_kfold_tf = tf_vectorizer_kfold.fit_transform(train_X_kfold)

        X_test_tf = tf_vectorizer.transform(test_X)

        # melakukan teknik imbalance dataset dengan undersampling data mayoritas dan melakukan oversampling data minoritas
        train_counts = train_y.value_counts()
        train_counts2 = train_y_kfold.value_counts()
        #st.write(train_counts)

        # undersampling data yang di split 70:30
        rus = RandomUnderSampler(sampling_strategy={'Neutral':train_counts['Positive'], 'Positive':train_counts['Positive'], 'Negative':train_counts['Negative']} , random_state=42)

        # undersampling data yang tdk di split
        rus_kfold = RandomUnderSampler(sampling_strategy={'Neutral':train_counts2['Positive'], 'Positive':train_counts2['Positive'], 'Negative':train_counts2['Negative']} , random_state=42)

        X_undersample, y_undersample = rus.fit_resample(X_train_tf, train_y)

        X_undersample_kfold, y_undersample_kfold = rus_kfold.fit_resample(x_kfold_tf, train_y_kfold) # untuk kfold

        # menampilkan data setelah di undersampling
        resample_counts = y_undersample.value_counts()
        resample_counts2 = y_undersample_kfold.value_counts()
        #st.write(resample_counts)

        # oversampling dengan SMOTE
        smote = SMOTE(random_state = 42)
        X_resampled, y_resampled = smote.fit_resample(X_undersample, y_undersample)

        X_resampled_kfold, y_resampled_kfold = smote.fit_resample(X_undersample_kfold, y_undersample_kfold) #untuk k-fold

        # menampilkan hasil setelah di oversampling dgn SMOTE
        smote_counts = y_resampled.value_counts()
        smote_counts2 = y_resampled_kfold.value_counts()
        #st.write(smote_counts) 

        #memanggil model algoritma MNB
        model = MultinomialNB()

        # membuat 3 kolom utk menampilkan dataset
        col, col2, col3 = st.columns(3)

        with col:
            st.write('Tampilan Dataset awal:')
            st.write(train_counts)
            st.write(train_counts2)
        with col2:
            st.write('Tampilan setelah undersampling:')
            st.write(resample_counts)
            st.write(resample_counts2)

        with col3:
            st.write('Tampilan setelah oversampling:')
            st.write(smote_counts)
            st.write(smote_counts2)


        st.divider()

        model.fit(X_resampled, y_resampled)
        # Predict
        y_pred = model.predict(X_test_tf)
        # Hasil Kinerja Tanpa K-Fold dan SMOTE
        st.write("Hasil Akurasi Train Test Split dan SMOTE")
        #score_accuracy = metrics.accuracy_score(test_y, y_pred)
        #st.write("Accuracy:   %0.2f" % score_accuracy)
        accuracy_with_SMOTE = metrics.accuracy_score(test_y, y_pred)*100
        st.write('MNB Accuracy Score = ',accuracy_with_SMOTE.round(3))

        st.divider()

        # konfigurasi cv
        kfold_withsmote = KFold(n_splits=10, shuffle=True, random_state=42)
        # hitung cv score
        kfold_scores_withsmote = cross_val_score(model, X_resampled_kfold, y_resampled_kfold, cv=kfold_withsmote, scoring='accuracy')

        fold_scores = []
        for fold, score in enumerate(kfold_scores_withsmote):
            fold_scores.append(score)
        # Menggambar grafik akurasi tiap lipatan
        plt.plot(range(1, len(fold_scores)+1), fold_scores, marker='o')
        plt.xlabel('Lipatan')
        plt.ylabel('Akurasi')
        plt.title('Akurasi Tiap Lipatan dengan K-Fold Cross Validation dan SMOTE')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.write('Akurasi:')
        result = ''
        for fold, score in enumerate(kfold_scores_withsmote):
            result += f"Fold {fold+1} = {score.round(2)}" + ', '
        #st.write(result)
        mean_accuracy = sum(kfold_scores_withsmote) / len(kfold_scores_withsmote)
        st.write("Rata-rata akurasi dengan SMOTE = ", mean_accuracy.round(2))

        self.conn.close()
if __name__ == "__main__":
    app = Klasifikasi()
    app.run()