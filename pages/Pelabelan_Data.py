import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import googletrans

from st_pages import Page, Section, show_pages, add_page_title
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class LabelingApp:
    def __init__(self):
        # Membuat koneksi ke database SQLite
        self.conn = sqlite3.connect('sentiment.db')

    def vader_analysis(self, compound):
        if compound >= 0.5:
            return 'Positive'
        elif compound <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    def label_data(self, df):
        translator = Translator()
        df["translated"] = df.apply(lambda _: ' ', axis=1)
        
        translated_word = []
        
        for element in df['tweet_clean']:
            translated_word.append(translator.translate(element).text)
        
        df['translated'] = translated_word
        
        analyzer = SentimentIntensityAnalyzer()
        
        for index, row in df.iterrows():
            text = row['translated']
            sentiment = analyzer.polarity_scores(text)
            df.at[index, 'neu'] = sentiment['neu']
            df.at[index, 'pos'] = sentiment['pos']
            df.at[index, 'neg'] = sentiment['neg']
            df.at[index, 'compound'] = sentiment['compound']
        
        df['label'] = df['compound'].apply(self.vader_analysis)
        
        st.write(
            """Berikut merupakan tampilan data setelah dilakukan pelabelan data:"""
        )
        st.dataframe(df)
        
        vader_counts = df['label'].value_counts()
        st.write(vader_counts)
        
        csv = df.to_csv().encode('utf-8')
        
        st.download_button(
            label=":page_facing_up: :green[Download data as CSV]",
            data=csv,
            file_name='tweet_labeled.csv',
            mime='text/csv',
        )

    def save_labeled_data(self, df):
        df = pd.DataFrame(df[['id', 'tweet_clean', 'translated', 'compound', 'label']])
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS hasilLabelling;")
        
        create_table_query = "CREATE TABLE IF NOT EXISTS hasilLabelling (id int, tweet_clean text, translated text, compound float, label text)"
        
        cursor.execute(create_table_query)
        
        df.to_sql('hasilLabelling', self.conn, if_exists='append', index=False)
        
        select_query = "SELECT * FROM hasilLabelling"
        cursor.execute(select_query)
        
        results = cursor.fetchall()
        
        self.conn.commit()

    def run(self):    
        add_page_title()

        # Membaca data dari tabel menggunakan pandas
        df = pd.read_sql_query("SELECT * FROM hasilpreprocess", self.conn)

        st.write(
            """Berikut merupakan tampilan data sebelum dilakukan tahapan pelabelan data, yang dimana tweet yang telah dicleaning
            akan ditranslate ke dalam bahasa Inggris untuk dilakukan pelabelan data dengan VaderSentiment yang akan memberikan polaritas
            sentimen dengan 3 kelas yakni positif, netral, dan negatif."""
        )

        jum = df['tweet_clean'].count()
        st.write('Jumlah data: ',jum,' tweets')

        # Menampilkan tabel menggunakan Streamlit
        st.dataframe(df)

        if st.button("Pelabelan _Lexicon_ VADER"):
            self.label_data(df)
            self.save_labeled_data(df)
            
            self.conn.close()

if __name__ == "__main__":
    app = LabelingApp()
    app.run()