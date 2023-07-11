import sqlite3
import pandas as pd

# Membaca data dari file CSV menggunakan pandas
df = pd.read_csv('colloquial-indonesian-lexicon2.csv')

# Membuat koneksi ke database SQLite
conn = sqlite3.connect('sentiment.db')

# Membuat objek cursor
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS kamusalay;")
# Membuat tabel dalam database (jika belum ada)
create_table_query = "CREATE TABLE IF NOT EXISTS kamusalay (slang text, formal text, dictionary text, context text, category1 text, category2 text, category3 text)"

cursor.execute(create_table_query)

# Menggunakan metode "to_sql" dari pandas untuk memasukkan data ke tabel
df.to_sql('kamusalay', conn, if_exists='append', index=False)

select_query = "SELECT * FROM kamusalay"
cursor.execute(select_query)

results = cursor.fetchall()

# Membuat dataframe dari hasil query menggunakan pandas
df = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])

# Menampilkan dataframe
print(df)

# Melakukan commit dan menutup koneksi
conn.commit()
conn.close()

