# Quadtree BMP Image Compression

Project ini dibuat untuk tugas praktikum Sistem Multimedia. Program ini melakukan kompresi gambar `.bmp` menggunakan algoritma **Quadtree Image Compression** dengan dua mode kompresi, yaitu **lossless** dan **lossy**.

## Deskripsi Singkat

Quadtree Image Compression bekerja dengan membagi gambar menjadi blok-blok kecil. Jika suatu blok memiliki warna yang seragam atau cukup mirip, blok tersebut disimpan sebagai satu warna. Jika belum seragam, blok akan dibagi lagi menjadi empat bagian.

Pada project ini:

- **Lossless**: blok hanya digabung jika semua piksel benar-benar identik.
- **Lossy**: blok boleh digabung jika perbedaan warna masih berada di bawah nilai threshold tertentu.

## Dataset

Dataset yang digunakan berupa gambar dengan format `.bmp`.

Ketentuan dataset:

- Jumlah gambar: 28 gambar
- Format file: `.bmp`
- Lokasi dataset: `data/raw_bmp/`

Contoh struktur dataset:

```text
data/
└── raw_bmp/
    ├── img_01.bmp
    ├── img_02.bmp
    ├── img_03.bmp
    └── ...
````

## Struktur Project

```text
quadtree-bmp-compression/
├── data/
│   └── raw_bmp/
│
├── output/
│   ├── compressed_lossless/
│   ├── compressed_lossy/
│   ├── reconstructed_lossless/
│   └── reconstructed_lossy/
│
├── results/
│   ├── tables/
│   └── charts/
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── utils.py
│   ├── quadtree.py
│   ├── compressor.py
│   ├── decompressor.py
│   ├── metrics.py
│   ├── charts.py
│   └── clean_outputs.py
│
├── report/
├── requirements.txt
├── README.md
└── .gitignore
```

## Fungsi Folder

| Folder                           | Fungsi                                     |
| -------------------------------- | ------------------------------------------ |
| `data/raw_bmp/`                  | Menyimpan gambar BMP asli sebagai data uji |
| `output/compressed_lossless/`    | Menyimpan hasil kompresi lossless `.qtree` |
| `output/compressed_lossy/`       | Menyimpan hasil kompresi lossy `.qtree`    |
| `output/reconstructed_lossless/` | Menyimpan hasil dekompresi lossless        |
| `output/reconstructed_lossy/`    | Menyimpan hasil dekompresi lossy           |
| `results/tables/`                | Menyimpan hasil pengujian dalam format CSV |
| `results/charts/`                | Menyimpan grafik hasil pengujian           |
| `src/`                           | Menyimpan seluruh source code program      |

## Instalasi

Pastikan Python sudah terinstall, lalu install dependency:

```bash
pip install -r requirements.txt
```

Isi `requirements.txt`:

```text
pillow
numpy
pandas
matplotlib
opencv-python
```

## Cara Menjalankan Program

Jalankan program utama dari root folder project:

```bash
python src/main.py
```

Program akan menampilkan menu interaktif:

```text
Mau proses gambar apa?
1. Proses semua gambar
2. Proses beberapa gambar tertentu
3. Proses rentang gambar tertentu
```

Setelah itu, user dapat memilih mode kompresi:

```text
1. Lossless dan Lossy
2. Lossless saja
3. Lossy saja
```

## Contoh Penggunaan

### Proses semua gambar

Pilih:

```text
1. Proses semua gambar
1. Lossless dan Lossy
```

### Proses beberapa gambar tertentu

Contoh input:

```text
1 5 10
```

Artinya program hanya memproses gambar nomor 1, 5, dan 10.

### Proses rentang gambar

Contoh:

```text
Dari nomor gambar: 5
Sampai nomor gambar: 10
```

Artinya program memproses gambar nomor 5 sampai 10.

## Output Program

Program menghasilkan beberapa output:

### 1. File hasil kompresi

```text
output/compressed_lossless/
output/compressed_lossy/
```

File hasil kompresi menggunakan ekstensi custom:

```text
.qtree
```

Contoh:

```text
img_01_lossless.qtree
img_01_lossy.qtree
```

### 2. File hasil dekompresi

```text
output/reconstructed_lossless/
output/reconstructed_lossy/
```

File ini berupa gambar `.bmp` hasil rekonstruksi dari file `.qtree`.

### 3. Tabel hasil pengujian

```text
results/tables/hasil_pengujian.csv
results/tables/ringkasan_hasil.csv
```

Metrik yang dihitung:

* ukuran file asli
* ukuran file hasil kompresi
* compression ratio
* saving percentage
* MSE
* RMSE
* MAE
* PSNR
* waktu kompresi
* waktu dekompresi
* jumlah node Quadtree
* jumlah leaf node
* kedalaman tree

### 4. Grafik hasil pengujian

```text
results/charts/
```

Grafik yang dibuat:

* perbandingan ukuran file
* compression ratio
* saving percentage
* PSNR
* waktu proses kompresi dan dekompresi

## Membersihkan Output

Untuk menghapus hasil output lama secara aman, jalankan:

```bash
python src/clean_outputs.py
```

Script ini hanya membersihkan folder:

```text
output/
results/tables/
results/charts/
```

Dataset di `data/raw_bmp/` tidak akan dihapus.

## Algoritma yang Digunakan

Algoritma yang digunakan adalah **Quadtree Image Compression**.

Tahapan umum algoritma:

1. Gambar dibaca sebagai array RGB.
2. Gambar dibagi menjadi blok.
3. Setiap blok dicek tingkat keseragaman warnanya.
4. Jika blok seragam, blok disimpan sebagai satu warna.
5. Jika belum seragam, blok dipecah menjadi empat bagian.
6. Proses dilakukan berulang sampai blok memenuhi syarat.
7. Struktur Quadtree disimpan ke file `.qtree`.
8. File `.qtree` dapat didekompresi kembali menjadi gambar `.bmp`.

## Perbedaan Lossless dan Lossy

| Mode     | Penjelasan                                                                              |
| -------- | --------------------------------------------------------------------------------------- |
| Lossless | Gambar hasil dekompresi identik dengan gambar asli                                      |
| Lossy    | Gambar hasil dekompresi mirip dengan gambar asli, tetapi ada sedikit penurunan kualitas |

Pada mode lossless, nilai MSE seharusnya `0` dan PSNR bernilai `infinity`.

Pada mode lossy, nilai MSE biasanya lebih besar dari `0`, tetapi ukuran file hasil kompresi jauh lebih kecil.

## Kesimpulan Singkat

Project ini menunjukkan bahwa algoritma Quadtree dapat digunakan untuk kompresi gambar `.bmp` dalam mode lossless dan lossy. Mode lossless menjaga gambar tetap identik dengan gambar asli, sedangkan mode lossy menghasilkan ukuran file yang lebih kecil dengan konsekuensi penurunan kualitas gambar.

```