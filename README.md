# 🥖 Bread Basket Analysis

Aplikasi web interaktif berbasis Streamlit untuk menganalisis pola pembelian konsumen pada transaksi toko roti menggunakan Market Basket Analysis dengan algoritma Apriori.

Project ini membantu pelaku usaha memahami asosiasi antarproduk berdasarkan waktu transaksi, hari, dan bulan agar dapat meningkatkan strategi cross-selling dan bundling produk.

---

## 📌 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur Utama](#-fitur-utama)
- [Struktur Direktori](#-struktur-direktori)
- [Dataset](#-dataset)
- [Metrik Apriori](#-metrik-apriori)
- [Instalasi](#-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Panduan Penggunaan](#-panduan-penggunaan)
- [Troubleshooting](#-troubleshooting)
- [Roadmap Pengembangan](#-roadmap-pengembangan)

---

## 📖 Tentang Proyek

Market Basket Analysis adalah teknik data mining yang digunakan untuk menemukan pola hubungan atau asosiasi antar-item yang sering dibeli dalam satu transaksi.

Pada proyek ini, data transaksi bakery diproses menggunakan algoritma Apriori dari pustaka `mlxtend`. Hasilnya ditampilkan dalam dashboard Streamlit yang memungkinkan pengguna untuk:

1. Memfilter transaksi berdasarkan periode waktu, jenis hari, bulan, dan hari spesifik.
2. Menentukan item utama sebagai antecedent.
3. Mendapatkan rekomendasi item pendamping (consequent) dengan nilai support, confidence, dan lift yang relevan.

---

## ✨ Fitur Utama

- 🔍 Filter data dinamis:
  - Waktu belanja: `Morning`, `Afternoon`, `Evening`
  - Jenis hari: `Weekday`, `Weekend`
  - Bulan: `January` sampai `December`
  - Hari: `Monday` sampai `Sunday`
- 📊 Analisis Apriori secara real-time:
  - Mengubah data transaksi menjadi matriks biner (`one-hot encoding`)
  - Menghitung frequent itemsets dengan `min_support = 0.01`
  - Menghasilkan association rules berdasarkan `lift >= 1`
- 💡 Rekomendasi cross-selling:
  - Menampilkan produk yang sering dibeli bersama dengan item tertentu
- 📓 Notebook eksplorasi data:
  - `bread_apriori.ipynb` untuk analisis EDA dan eksperimen model

---

## 📂 Struktur Direktori

```text
streamlit-BreadApriori-main/
├── app.py                 # Antarmuka utama aplikasi Streamlit
├── bread basket.csv       # Dataset transaksi toko roti
├── bread_apriori.ipynb    # Notebook eksplorasi dan analisis Apriori
├── requirements.txt       # Daftar dependensi Python
├── README.md              # Dokumentasi proyek
└── .gitignore             # File konfigurasi Git
```

---

## 📊 Dataset

Dataset yang digunakan adalah `bread basket.csv`, berisi riwayat transaksi pembelian produk roti dan minuman. Kolom utama pada dataset adalah:

| Kolom | Deskripsi | Contoh Nilai |
| :--- | :--- | :--- |
| `Transaction` | ID unik transaksi | `1`, `2`, `3` |
| `Item` | Nama produk yang dibeli | `Bread`, `Coffee`, `Pastry`, `Tea` |
| `date_time` | Tanggal dan waktu transaksi | `30-10-2016 09:58` |
| `period_day` | Periode transaksi | `morning`, `afternoon`, `evening`, `night` |
| `weekday_weekend` | Jenis hari | `weekday`, `weekend` |

---

## 🧠 Metrik Apriori

Algoritma Apriori menghasilkan aturan asosiasi dengan format:

A → B

Keterangan:
- A = antecedent (jika membeli A)
- B = consequent (maka membeli B)

Tiga metrik utama yang digunakan dalam aplikasi ini adalah:

1. Support
   
   Menunjukkan seberapa sering kombinasi item A dan B muncul dalam seluruh transaksi.

   Formula:
   
   Support(A ∪ B) = (Jumlah transaksi berisi A dan B) / (Total transaksi)

   Dalam aplikasi ini, nilai minimum support diatur pada `0.01` (1%).

2. Confidence
   
   Menunjukkan seberapa sering item B dibeli ketika item A dibeli.

   Formula:
   
   Confidence(A → B) = Support(A ∪ B) / Support(A)

3. Lift Ratio
   
   Mengukur apakah hubungan antara A dan B lebih kuat dari hubungan acak.

   Formula:
   
   Lift(A → B) = Confidence(A → B) / Support(B)

   Interpretasi:
   - `Lift > 1` : asosiasi positif
   - `Lift = 1` : independen
   - `Lift < 1` : asosiasi negatif

---

## 🚀 Instalasi

### Prasyarat

- Python 3.8 atau lebih tinggi (disarankan 3.9 - 3.11)
- Git (opsional)

### Langkah-langkah

1. Buka terminal atau PowerShell di folder proyek.

```bash
cd /path/to/streamlit-BreadApriori-main
```

2. Buat virtual environment.

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependensi.

```bash
pip install -r requirements.txt
```

> Catatan: Pastikan paket `mlxtend` terpasang dengan benar karena digunakan untuk proses perhitungan Apriori.

---

## ▶️ Cara Menjalankan

Jalankan aplikasi Streamlit dengan perintah berikut:

```bash
streamlit run app.py
```

Setelah server aktif, buka browser Anda dan akses:

```text
http://localhost:8501
```

---

## 🖥️ Panduan Penggunaan

1. Buka aplikasi di browser pada `http://localhost:8501`.
2. Pilih item utama yang ingin dianalisis, misalnya `Bread`, `Coffee`, atau `Cookies`.
3. Atur filter waktu, hari, bulan, dan hari tertentu sesuai kebutuhan.
4. Aplikasi akan menghitung aturan asosiasi secara otomatis.
5. Jika pola asosiasi ditemukan, hasil akan menampilkan rekomendasi produk yang sering dibeli bersamaan.
6. Jika tidak ada pola yang cocok, aplikasi akan menampilkan pesan bahwa tidak ada rekomendasi yang relevan.

---

## 🛠️ Troubleshooting

### 1. `ModuleNotFoundError: No module named 'mlxtend'`

Jalankan:

```bash
pip install mlxtend
```

### 2. Konflik versi pandas / numpy / scipy

Pastikan versi dependency yang digunakan kompatibel. Jika terjadi warning atau error terkait kompatibilitas, upgrade atau sesuaikan package sesuai kebutuhan proyek.

### 3. Port 8501 sudah digunakan

Jalankan Streamlit pada port lain:

```bash
streamlit run app.py --server.port 8502
```

---

## 🤝 Roadmap Pengembangan

Beberapa ide pengembangan yang bisa ditambahkan di masa depan:

- [ ] Visualisasi grafik asosiasi antar item
- [ ] Slider interaktif untuk mengatur `min_support` dan `min_threshold` dari UI
- [ ] Tabel interaktif yang menampilkan daftar lengkap association rules
- [ ] Export hasil rekomendasi ke CSV atau Excel

---

## 📎 Catatan

Proyek ini cocok digunakan untuk studi kasus analisis pola pembelian, terutama pada sektor retail, supermarket, atau toko roti yang ingin mengoptimalkan strategi penjualan berbasis data.
