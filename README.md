# 🥖 Bread Basket Analysis - Association Rules with Apriori

Aplikasi web interaktif berbasis **Streamlit** untuk menganalisis pola pembelian konsumen pada transaksi toko roti (*Bakery*) menggunakan **Market Basket Analysis (Algoritma Apriori)**. Aplikasi ini membantu pelaku bisnis memahami asosiasi antarproduk (*cross-selling*) berdasarkan waktu transaksi, hari, dan bulan.

---

## 📌 Daftar Isi
- [Tentang Proyek](#-tentang-proyek)
- [Fitur Utama](#-fitur-utama)
- [Struktur Direktori](#-struktur-direktori)
- [Dataset](#-dataset)
- [Memahami Metrik Apriori](#-memahami-metrik-apriori)
- [Panduan Instalasi & Menjalankan](#-panduan-instalasi--menjalankan)
- [Panduan Penggunaan](#-panduan-penggunaan)
- [Troubleshooting](#-troubleshooting)

---

## 📖 Tentang Proyek

**Market Basket Analysis** adalah teknik data mining yang digunakan untuk menemukan pola hubungan atau asosiasi antar-item yang sering dibeli secara bersamaan dalam sebuah transaksi. 

Proyek ini menerapkan algoritma **Apriori** dari pustaka `mlxtend` pada dataset transaksi riil bakery (*Bread Basket Dataset*). Melalui dashboard interaktif Streamlit, pengguna dapat:
1. Memfilter transaksi berdasarkan parameter waktu (pagi, siang, malam), jenis hari (hari kerja atau akhir pekan), bulan, dan hari spesifik.
2. Memilih item acuan (*antecedent*).
3. Mendapatkan rekomendasi produk pasangan (*consequent*) dengan tingkat keyakinan (*confidence*) dan lift rasio tertinggi.

---

## ✨ Fitur Utama

- 🔍 **Filter Data Dinamis**:
  - Pilihan waktu belanja: `Morning`, `Afternoon`, `Evening`.
  - Pilihan kategori hari: `Weekday`, `Weekend`.
  - Slider pilihan bulan: `January` s/d `December`.
  - Slider pilihan hari: `Monday` s/d `Sunday`.
- 📊 **Model Apriori Real-time**: Mengolah data transaksi yang difilter menjadi pivot matrix biner (*one-hot encoded*), menghitung *frequent itemsets* dengan ambang batas `min_support = 0.01`, serta aturan asosiasi berdasarkan `lift >= 1`.
- 💡 **Rekomendasi Bundling/Cross-Selling**: Menampilkan rekomendasi produk pendamping jika seorang konsumen membeli item tertentu.
- 📓 **Jupyter Notebook Analisis Eksploratif**: Menyertakan notebook `bread_apriori.ipynb` untuk eksplorasi data mendalam dan visualisasi EDA (Exploratory Data Analysis).

---

## 📂 Struktur Direktori

```text
streamlit-BreadApriori-main/
│
├── app.py                 # File utama antarmuka web Streamlit
├── bread basket.csv       # Dataset transaksi kasir toko roti
├── bread_apriori.ipynb    # Jupyter Notebook untuk eksplorasi dan pemodelan Apriori
├── requirements.txt       # Daftar pustaka & dependensi Python
└── README.md              # Dokumentasi lengkap proyek
```

---

## 📊 Dataset

Dataset yang digunakan adalah `bread basket.csv` yang berisi riwayat transaksi pembelian produk roti dan minuman dengan kolom-kolom berikut:

| Kolom | Deskripsi | Contoh Nilai |
| :--- | :--- | :--- |
| `Transaction` | ID unik transaksi | `1`, `2`, `3`, ... |
| `Item` | Nama item/produk yang dibeli | `Bread`, `Coffee`, `Pastry`, `Tea` |
| `date_time` | Waktu dan tanggal transaksi | `30-10-2016 09:58` |
| `period_day` | Periode waktu pembelian | `morning`, `afternoon`, `evening`, `night` |
| `weekday_weekend` | Tipe hari transaksi | `weekday`, `weekend` |

---

## 🧠 Memahami Metrik Apriori

Algoritma Apriori menghasilkan aturan asosiasi berbentuk:
$$\text{Antecedent (Jika membeli A)} \longrightarrow \text{Consequent (Maka membeli B)}$$

Tiga metrik utama yang digunakan dalam aplikasi ini:
1. **Support**: Seberapa sering kombinasi item $A$ dan $B$ muncul dalam seluruh transaksi.
   $$\text{Support}(A \cup B) = \frac{\text{Jumlah transaksi berisi } A \text{ dan } B}{\text{Total seluruh transaksi}}$$
   *(Pada aplikasi disetel `min_support = 0.01` atau minimal 1% dari transaksi).*

2. **Confidence**: Seberapa sering item $B$ dibeli ketika item $A$ dibeli.
   $$\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
   *(Aturan diurutkan berdasarkan confidence tertinggi).*

3. **Lift Ratio**: Mengukur kekuatan aturan dibandingkan jika kedua item dibeli secara independen.
   $$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$
   - **Lift > 1**: Menunjukkan hubungan asosiasi positif (kedua produk saling mendorong penjualan).
   - **Lift = 1**: Kedua produk independen (tidak berhubungan).
   - **Lift < 1**: Asosiasi negatif (pembelian satu item menurunkan kemungkinan pembelian item lain).

---

## 🚀 Panduan Instalasi & Menjalankan

### 1. Prasyarat Sistem
- Python versi **3.8** ke atas (direkomendasikan Python 3.9 - 3.11).
- Git (opsional).

### 2. Clone atau Buka Folder Proyek
Buka terminal / PowerShell di direktori proyek ini:
```bash
cd /path/to/streamlit-BreadApriori-main
```

### 3. Buat Virtual Environment (Disarankan)
Sangat dianjurkan menggunakan virtual environment agar dependensi tidak berbenturan dengan sistem utama:

- **Windows (PowerShell/CMD):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Instal Dependensi
Jalankan perintah berikut untuk menginstal seluruh pustaka yang dibutuhkan:
```bash
pip install -r requirements.txt
```

> **Catatan Dependensi:** Pastikan paket `mlxtend` terinstal dengan baik karena digunakan langsung untuk perhitungan algoritma Apriori.

### 5. Jalankan Aplikasi Streamlit
Jalankan server aplikasi lokal:
```bash
streamlit run app.py
```

Setelah perintah dijalankan, antarmuka web akan otomatis terbuka di browser Anda (atau akses melalui alamat: `http://localhost:8501`).

---

## 🖥️ Panduan Penggunaan

1. Buka aplikasi di browser (`http://localhost:8501`).
2. Pada panel kontrol:
   - **Pilih Item**: Tentukan item utama yang ingin dicari produk pendampingnya (contoh: *Bread*, *Coffee*, *Cookies*).
   - **Pilih Waktu**: Tentukan periode transaksi (`Morning`, `Afternoon`, `Evening`).
   - **Pilih Hari**: Tentukan kategori hari (`Weekday` atau `Weekend`).
   - **Pilih Bulan & Hari**: Sesuaikan slider bulan dan hari.
3. Aplikasi akan langsung menghitung aturan asosiasi berdasarkan filter yang dipilih:
   - Jika ditemukan pola asosiasi, kotak hijau akan menampilkan:
     > *"Jika konsumen membeli **[Item Pilihan]**, maka membeli **[Item Rekomendasi]** secara bersamaan"*
   - Jika tidak ditemukan pola asosiasi pada data transaksi yang difilter, aplikasi akan menampilkan pesan peringatan atau tidak ada rekomendasi.

---

## 🛠️ Troubleshooting

- **Error: `ModuleNotFoundError: No module named 'mlxtend'`**  
  Jalankan:
  ```bash
  pip install mlxtend
  ```
- **Error versi pandas / numpy dengan `mlxtend`**:  
  Pastikan menggunakan versi scikit-learn dan scipy yang kompatibel jika terjadi peringatan deprecation.
- **Port 8501 sudah digunakan**:  
  Jalankan streamlit dengan port alternatif:
  ```bash
  streamlit run app.py --server.port 8502
  ```

---

## 🤝 Kontribusi & Pengembangan Lanjutan

Ide pengembangan yang dapat ditambahkan di masa mendatang:
- [ ] Penambahan visualisasi grafik asosiasi (Network Graph antar item).
- [ ] Slider interaktif untuk mengatur nilai `min_support` dan `min_threshold` langsung dari antarmuka web.
- [ ] Tabel interaktif yang menampilkan daftar lengkap seluruh *association rules* (Support, Confidence, Lift).
