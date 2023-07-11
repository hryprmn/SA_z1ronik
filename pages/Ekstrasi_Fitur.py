import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import googletrans

from st_pages import Page, Section, show_pages, add_page_title
from sklearn.feature_extraction.text import TfidfVectorizer

class EkstrasiFitur:
    def __init__(self):
        self.conn = sqlite3.connect('sentiment.db')

    def transform_ekstrasi_fitur(self,df):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['tweet_clean'])

        # Mendapatkan daftar kata kunci yang digunakan sebagai kolom matriks
        feature_names = vectorizer.get_feature_names_out()

        # Membuat dataframe dari matriks TF-IDF
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        # Menampilkan matriks transpose
        df_tfidf_transposed = df_tfidf.transpose()

        st.subheader("Berikut ini merupakan tampilan dataset dalam bentuk vektor setelah TF-IDF")
        st.dataframe(df_tfidf_transposed)

    def run(self):
        add_page_title()

        st.write(
            """Pada halaman ini akan dilakukan ekstrasi fitur dengan menggunakan metode TF-IDF. Ini akan merubah representasi dari data teks menjadi kedalam bentuk numerik (vektor)."""
        )

        # Membaca data dari tabel menggunakan pandas
        df = pd.read_sql_query("SELECT * FROM hasilLabelling", self.conn)

        #mengubah tampilan dataframe
        df = pd.DataFrame(df[['tweet_clean', 'label']])

        # Menampilkan tabel menggunakan Streamlit
        st.dataframe(df)
        st.divider()

        self.transform_ekstrasi_fitur(df)

if __name__ == "__main__":
    app = EkstrasiFitur()
    app.run()
    # Menghitung frekuensi kata-kata
    #word_frequencies = tfidf_matrix.sum(axis=0)

    # Membuat dataframe frekuensi kata-kata
    #df_word_frequencies = pd.DataFrame(
        #word_frequencies.T.A, columns=['Frequency'], index=feature_names
    #)

    # Menampilkan kata-kata yang paling sering muncul
    #most_frequent_words = df_word_frequencies.nlargest(10, 'Frequency')
    #st.write("Kata-kata yang paling sering muncul:")
    #st.dataframe(most_frequent_words)
