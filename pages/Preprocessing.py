import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk 
import string
import re

from st_pages import Page, Section, show_pages, add_page_title
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class PreprocessingApp:
    def __init__(self):
        self.conn = sqlite3.connect('sentiment.db')

    # Proses Cleansing Data
    def cleansing(self, tweet):
        # hapus mentions
        tweet = re.sub(r'\@([\w]+)',' ', tweet)

        # hapus hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        
        # hapus duplikasi tiga karakter beruntun atau lebih (contoh. yukkk)
        #tweet = re.sub(r'([a-zA-Z])\1\1','\\1', tweet)
        #tweet = re.sub(r'(\w)\1', r'\1', tweet)
        
        #remove whitespace leading & trailing/ spasi
        tweet = tweet.strip()

        #remove multiple whitespace into single whitespace
        tweet = re.sub('\s+',' ',tweet)

        # hapus newline
        tweet = tweet.replace('\n','')

        #hapus angka
        tweet = re.sub(r'[0-9]+',' ', tweet)

        # hapus simbol simbol
        tweet = re.sub(r"[^\w\s]",' ',tweet)

        #hapus spasi di awal dan akhir kalimat
        tweet = re.sub(r'^[ ]|[ ]$','', tweet)
        return tweet

    # Menghapus emoji
    def remove_emojis(self, data):
        emoj = re.compile("["
            u"\U00002700-\U000027BF"  # Dingbats
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U00002600-\U000026FF"  # Miscellaneous Symbols
            u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                        "]+", re.UNICODE)
        return re.sub(emoj, '', data)

    # Mengganti kata-kata slang/gawl
    def replace_slang_word(self, words):
        indo_slang_words = pd.read_sql_query("SELECT * FROM kamusalay", self.conn)
        for index in  range(0,len(words)-1):
            index_slang = indo_slang_words.slang==words[index]
            formal = list(set(indo_slang_words[index_slang].formal))
            if len(formal):
                words[index]=formal[0]
        return words

    #Tokenizing
    #nltk.download('punkt')
    def tokenize_tweet(self, tweet):
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(tweet)
        return tokens

    # Proses stopwords
    def stopwords_removal(self, words):
        #nltk.download('stopwords')
        stopwords_indonesia = stopwords.words('indonesian')
        return [word for word in words if word not in stopwords_indonesia]

    def stem_text(self, tokens):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return [stemmer.stem(token) for token in tokens]

    # Proses pengembalian Token menjadi kata utuh
    def list_to_text(self, token):
        text = " "
        return text.join(token)

    # untuk menyimpan df ke dalam csv

    # def convert_df(df):
    #     # IMPORTANT: Cache the conversion to prevent computation on every rerun
    #     return df.to_csv().encode('utf-8')
        # convert df to csv
    @st.cache_data
    def convert_df_to_csv(self, df):
       return df.to_csv().encode('utf-8')


    # menyimpan data yang telah dibersihkan kedalam database
    def save_preprocessing_data(self, df):
        df = pd.DataFrame(df[['id','tweet_clean']]) #mengubah dataframe
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS hasilpreprocess;")

        create_table_query = "CREATE TABLE IF NOT EXISTS hasilpreprocess (id int, tweet_clean text)"

        cursor.execute(create_table_query)

        # Menggunakan metode "to_sql" dari pandas untuk memasukkan data ke tabel
        df.to_sql('hasilpreprocess', self.conn, if_exists='append', index=False)

        select_query = "SELECT * FROM hasilpreprocess"
        cursor.execute(select_query)

        results = cursor.fetchall()

        self.conn.commit()

    def run(self):
        add_page_title()
        st.write(
            """Berikut merupakan tampilan data sebelum dilakukan tahapan pre-processing, Pada pre-processing ini data akan melalui beberapa
            proses seperti: _case folding_, _cleansing_, _tokenizing_, _stopwords removal_, dan _stemming_."""
        )

        # Membaca data dari tabel menggunakan pandas
        anies_df = pd.read_sql_query("SELECT * FROM anies", self.conn)

        prabowo_df = pd.read_sql_query("SELECT * FROM prabowo", self.conn)

        # Menampilkan tabel menggunakan Streamlit
        option1 = st.selectbox('Tampilkan Dataset:',('Tweet Anies Baswedan', 'Tweet Prabowo Subianto'),key=1)

        if option1 == 'Tweet Anies Baswedan':
            st.dataframe(anies_df)
            df = anies_df
            jum = df['tweet'].count()
            st.write('Jumlah data: ',jum,' tweets')

        elif option1 == 'Tweet Prabowo Subianto':
            st.dataframe(prabowo_df)
            df = prabowo_df
            jum = df['tweet'].count()
            st.write('Jumlah data: ',jum,' tweets')
            # Generate pandas profiling report

        # Mengubah dataframe

        tweet_df = pd.DataFrame(df[['id', 'tweet']])#mengubah data frame

        if st.button("Pre-processing Data"):
            # drop table
            # nama tabel: hasilpreprocess
            # CREATE TABLE IF NOT EXISTS hasilpreprocess ('tweet string')
            # cursor execute
            tweet_df = pd.DataFrame(df[['id','tweet']]) #mengubah dataframe

            tweet_df['tweet_lower']= tweet_df['tweet'].str.lower() #Proses case folding yg mengubah setiap kata menjadi lowercase

            tweet_df['tweet_cleansing'] = tweet_df['tweet_lower'].apply(self.cleansing) #Proses cleansing tweets

            tweet_df['tweet_cleansing']= tweet_df['tweet_cleansing'].apply(self.remove_emojis) #menghilangkan emoji pada tweets

            tweet_df.drop_duplicates(subset ="tweet_cleansing", keep = 'first', inplace = True) #menghapus data duplikasi pada tweet dan menyimpan yang 'pertama'
            
            tweet_df['tweet_token'] = tweet_df['tweet_cleansing'].apply(self.tokenize_tweet) #Proses Tokenizing yang memisahkan kata pada suatu tweets dng menggunakan regexp

            tweet_df['tweet_slang'] = tweet_df['tweet_token'].apply(self.replace_slang_word) # Proses mengubah kata slang/gawl yg terdapat pada tweets kedalam kata asalnya

            tweet_df['tweet_slang'] = tweet_df['tweet_slang'].apply(self.list_to_text) #mengembalikan terlebih dahulu token ke text karena terdapat kata-kata tidak di token setelah mengubah slang words

            tweet_df['tweet_slang'] = tweet_df['tweet_slang'].apply(self.cleansing)

            tweet_df['tweet_slang'] = tweet_df['tweet_slang'].apply(self.tokenize_tweet)

            tweet_df['tweet_stopwords'] = tweet_df['tweet_slang'].apply(self.stopwords_removal) #Proses menghapus stopwords berbahasa Indonesia

            tweet_df['tweet_stem'] = tweet_df['tweet_stopwords'].apply(self.stem_text) #Proses Stemming

            tweet_df['tweet_clean'] = tweet_df['tweet_stem'].apply(self.list_to_text) #Pengembalian Token ke kata utuh

            tweet_df.sort_values('tweet_clean', ascending=True)

            # menampilkan df yg telah di pre-processing
            st.write(
                """Berikut merupakan tampilan data setelah dilakukan pre-processing:"""
            )
            st.dataframe(tweet_df)
            # hitung jumlah data
            jum = tweet_df['tweet_clean'].count()
            st.write('Jumlah data: ',jum,' tweets')

            tweet_df_fix = pd.DataFrame(tweet_df[['id', 'tweet_clean']])#mengubah data frame
            self.save_preprocessing_data(tweet_df_fix)

            csv_df = tweet_df.to_csv().encode('utf-8')
            st.download_button(
                label=":page_facing_up: :green[Download data as CSV]",
                data=csv_df,
                file_name='tweet_hasilpreprocessing.csv',
                mime='text/csv',
            )
    
if __name__ == "__main__":
    app = PreprocessingApp()
    app.run()
