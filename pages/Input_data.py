import streamlit as st
import pandas as pd
import sqlite3

from io import StringIO
from st_pages import Page, Section, show_pages, add_page_title

add_page_title()

st.write(
    """Drag dan drop data yang ingin dijadikan data latih"""
)

uploaded_file = st.file_uploader("Choose a file", type='csv')

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)

    # Membuat koneksi ke database SQLite
    conn = sqlite3.connect('sentiment.db')
    # Membuat objek cursor
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS data_tweet;")
    # Membuat tabel dalam database (jika belum ada)
    create_table_query = "CREATE TABLE IF NOT EXISTS data_tweet(id integer, date date, tweet text)"

    cursor.execute(create_table_query)

    # Menggunakan metode "to_sql" dari pandas untuk memasukkan data ke tabel
    df.to_sql('data_tweet', conn, if_exists='append', index=False)

    select_query = "SELECT * FROM data_tweet"
    cursor.execute(select_query)

    results = cursor.fetchall()

    # Membuat dataframe dari hasil query menggunakan pandas
    df = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])

    # Menampilkan dataframe
    st.write("Berikut merupakan tampilan data yang diunggah dalam bentuk dataframe:")
    st.dataframe(df)
    st.write(":green[File berhasil diunggah kedalam database!]")
    # Melakukan commit dan menutup koneksi
    conn.commit()
    conn.close()


    