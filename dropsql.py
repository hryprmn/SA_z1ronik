import sqlite3
import pandas as pd

# Membuat koneksi ke database SQLite
conn = sqlite3.connect('sentiment.db')

# Membuat objek cursor
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS anies;")


# Melakukan commit dan menutup koneksi
conn.commit()
conn.close()

