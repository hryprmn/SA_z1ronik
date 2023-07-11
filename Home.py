import streamlit as st
import sqlite3
import pandas as pd
from st_pages import Page, Section, show_pages, add_page_title

class Home:
    def __init__(self):
        self.pages = [
            Page("Home.py", "Home", "🏠"),
            Page("pages/Input_data.py", "Import Data", ":inbox_tray:"),
            Page("pages/Preprocessing.py", "Pre-processing", ":gear:"),
            Page("pages/Pelabelan_Data.py", "Pelabelan Data", ":label:"),
            Page("pages/Ekstrasi_Fitur.py", "Ekstrasi Fitur", ":broccoli:"),
            Page("pages/Klasifikasi.py", "Klasifikasi", ":hammer_and_wrench:"),
            Page("pages/Evaluasi.py", "Evaluasi", ":thinking_face:")
        ]
    
    def run(self):
        show_pages(self.pages)
        
        st.header(':blue[Analisis Sentimen Terhadap Bakal Calon Presiden 2024 Menggunakan Algoritma Multinomial Naive Bayes (MNB)]')
        st.divider()
        col, col2 = st.columns(2)

        with col:
            st.subheader("Pembimbing 1: Yulison H. Chrisnanto, S.T., M.T.")

        with col2:
            st.subheader("Pembimbing 2: Herdi Ashaury, S.Kom., M.T.")
        
        st.divider()
        st.subheader('Dibuat oleh: Hary Permana - 3411191021')

if __name__ == "__main__":
    app = Home()
    app.run()

