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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from st_pages import Page, Section, show_pages, add_page_title
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay

class Evaluasi:
    def __init__(self):
        self.conn = sqlite3.connect('sentiment.db')

    def under_sampling(self, X_train_tf, train_y):
        # hitung jumlah label
        train_counts = train_y.value_counts()
        
        # undersampling data yang di split 80:20
        rus = RandomUnderSampler(
            sampling_strategy={
                'Neutral':train_counts['Positive'], 
                'Positive':train_counts['Positive'], 
                'Negative':train_counts['Negative']
                }, 
                random_state=42)
        X_undersample, y_undersample = rus.fit_resample(X_train_tf, train_y)

        return X_undersample, y_undersample

    def over_sampling(self, X_undersample, y_undersample):
        # oversampling dengan SMOTE
        smote = SMOTE(random_state = 42)
        X_resampled, y_resampled = smote.fit_resample(X_undersample, y_undersample)

        # menampilkan hasil setelah di oversampling dgn SMOTE
        smote_counts = y_resampled.value_counts()

        return X_resampled, y_resampled

    def train_evaluasi(self, X_resampled, y_resampled, X_test_tf, test_y):
        model = MultinomialNB()
        #latih data
        model.fit(X_resampled, y_resampled)
        #prediksi data test
        y_pred = model.predict(X_test_tf)

        acc_score = metrics.accuracy_score(test_y, y_pred)
        st.write("Accuracy: ", acc_score.round(3)*100)

        recall_score = metrics.recall_score(test_y, y_pred, average='weighted')
        st.write("Recall: ", recall_score.round(3)*100)

        precision_score = metrics.precision_score(test_y, y_pred, average='weighted', zero_division=1)
        st.write("Precision: ", precision_score.round(3)*100)

        # Menghitung metrik F1 score
        f1_score = metrics.f1_score(test_y, y_pred, average='weighted', zero_division=1)
        st.write("F1-Score: ", f1_score.round(3)*100)

        cm = metrics.confusion_matrix(test_y, y_pred)

        st.subheader("Confusion Matrix")
        #plot_confusion_matrix(model, X_test_tf, test_y)
        ConfusionMatrixDisplay.from_predictions(test_y, y_pred)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def run(self):
        add_page_title()

        st.write(
            """Pada halaman ini akan menampilkan hasil kinerja dan evaluasi dari algoritma Multinomial Naive Bayes dengan Confusion Matrix."""
        )

        # Membaca data dari tabel menggunakan pandas
        df = pd.read_sql_query("SELECT * FROM hasilLabelling", self.conn)

        train_data, test_data = train_test_split(df, test_size = 0.2, random_state=42)

        train_X = train_data['tweet_clean']
        train_y = train_data['label'] 
        test_X = test_data['tweet_clean']
        test_y = test_data['label']

        # hitung TF-IDF
        tf_vectorizer = TfidfVectorizer()    
        X_train_tf = tf_vectorizer.fit_transform(train_X)

        X_test_tf = tf_vectorizer.transform(test_X)

        X_undersample, y_undersample = self.under_sampling(X_train_tf, train_y)
        
        X_resampled, y_resampled = self.over_sampling(X_undersample, y_undersample)

        self.train_evaluasi(X_resampled, y_resampled, X_test_tf, test_y)
        # melakukan teknik imbalance dataset dengan undersampling data mayoritas dan melakukan oversampling data minoritas
        
if __name__ == "__main__":
    app = Evaluasi()
    app.run()
