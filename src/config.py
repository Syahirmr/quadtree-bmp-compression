"""
config.py

File ini berisi semua konfigurasi utama project Quadtree BMP Compression.

Tujuan file ini:
1. Menyimpan path folder project secara terpusat.
2. Menyimpan pengaturan algoritma Quadtree.
3. Menyimpan pengaturan output dan hasil pengujian.
4. Membuat project lebih rapi, aman, dan mudah di-maintain.

Catatan:
- Jangan hardcode path seperti "D:/..." supaya project tetap bisa jalan
  di komputer lain.
- Semua path dibuat relatif dari root project.
"""

from pathlib import Path


# ============================================================
# ROOT PROJECT
# ============================================================

# Path ke folder root project.
# __file__ menunjuk ke file config.py.
# parents[1] artinya naik 2 level:
# src/config.py -> src/ -> quadtree-bmp-compression/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


# ============================================================
# DATASET CONFIG
# ============================================================

# Folder utama data.
DATA_DIR: Path = PROJECT_ROOT / "data"

# Folder gambar asli BMP yang akan dipakai sebagai data uji.
# Semua gambar input resmi untuk tugas kita harus berada di folder ini.
RAW_BMP_DIR: Path = DATA_DIR / "raw_bmp"

# Ekstensi gambar yang dipakai kelompok kita.
# Dibuat lowercase agar pengecekan file lebih konsisten.
IMAGE_EXTENSION: str = ".bmp"

# Minimal jumlah gambar sesuai ketentuan tugas.
MINIMUM_IMAGE_COUNT: int = 20


# ============================================================
# OUTPUT CONFIG
# ============================================================

# Folder utama untuk semua hasil output program.
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# Folder hasil kompresi lossless.
COMPRESSED_LOSSLESS_DIR: Path = OUTPUT_DIR / "compressed_lossless"

# Folder hasil kompresi lossy.
COMPRESSED_LOSSY_DIR: Path = OUTPUT_DIR / "compressed_lossy"

# Folder gambar hasil dekompresi lossless.
RECONSTRUCTED_LOSSLESS_DIR: Path = OUTPUT_DIR / "reconstructed_lossless"

# Folder gambar hasil dekompresi lossy.
RECONSTRUCTED_LOSSY_DIR: Path = OUTPUT_DIR / "reconstructed_lossy"


# ============================================================
# RESULTS CONFIG
# ============================================================

# Folder utama hasil analisis.
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Folder untuk tabel CSV hasil pengujian.
RESULTS_TABLES_DIR: Path = RESULTS_DIR / "tables"

# Folder untuk grafik hasil pengujian.
RESULTS_CHARTS_DIR: Path = RESULTS_DIR / "charts"

# File CSV utama untuk menyimpan hasil pengujian.
RESULTS_CSV_PATH: Path = RESULTS_TABLES_DIR / "hasil_pengujian.csv"


# ============================================================
# QUADTREE ALGORITHM CONFIG
# ============================================================

# Ukuran blok terkecil yang boleh dibuat oleh Quadtree.
# Nilai 1 berarti blok bisa dipecah sampai ukuran 1x1 piksel.
# Ini penting untuk lossless agar gambar bisa direkonstruksi dengan akurat.
MIN_BLOCK_SIZE: int = 1

# Threshold untuk mode lossy.
# Semakin besar nilainya:
# - kompresi semakin tinggi,
# - ukuran file semakin kecil,
# - kualitas gambar semakin menurun.
#
# Semakin kecil nilainya:
# - kualitas gambar semakin bagus,
# - ukuran file hasil kompresi cenderung lebih besar.
LOSSY_THRESHOLD: float = 18.0

# Threshold untuk mode lossless.
# Nilai 0 berarti tidak ada toleransi perbedaan warna.
# Jadi satu blok hanya bisa disimpan sebagai satu warna jika pikselnya benar-benar identik.
LOSSLESS_THRESHOLD: float = 0.0

# Format file hasil kompresi custom.
# Kita pakai .qtree supaya jelas bahwa ini adalah hasil kompresi Quadtree,
# bukan file gambar biasa.
COMPRESSED_FILE_EXTENSION: str = ".qtree"


# ============================================================
# IMAGE PROCESSING CONFIG
# ============================================================

# Mode warna gambar yang dipakai.
# RGB artinya setiap piksel punya 3 channel warna: Red, Green, Blue.
IMAGE_MODE: str = "RGB"

# Kualitas default untuk penyimpanan gambar hasil rekonstruksi jika diperlukan.
# Untuk BMP biasanya tidak terlalu berpengaruh karena BMP tidak memakai kompresi lossy JPEG.
DEFAULT_SAVE_QUALITY: int = 95


# ============================================================
# PERFORMANCE CONFIG
# ============================================================

# Jika True, program akan menampilkan progress saat memproses banyak gambar.
SHOW_PROGRESS: bool = True

# Jika True, program akan menimpa file output lama dengan hasil baru.
# Ini membantu supaya tidak ada hasil lama yang tercampur dengan hasil pengujian baru.
OVERWRITE_OUTPUT: bool = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_output_directories() -> list[Path]:
    """
    Mengembalikan daftar semua folder output yang dibutuhkan program.

    Fungsi ini dibuat agar pembuatan folder bisa dilakukan dari satu tempat,
    sehingga lebih mudah di-maintain.
    """
    return [
        COMPRESSED_LOSSLESS_DIR,
        COMPRESSED_LOSSY_DIR,
        RECONSTRUCTED_LOSSLESS_DIR,
        RECONSTRUCTED_LOSSY_DIR,
        RESULTS_TABLES_DIR,
        RESULTS_CHARTS_DIR,
    ]


def ensure_output_directories() -> None:
    """
    Membuat semua folder output jika belum ada.

    exist_ok=True berarti:
    - Kalau folder belum ada, folder akan dibuat.
    - Kalau folder sudah ada, program tidak error.
    """
    for directory in get_output_directories():
        directory.mkdir(parents=True, exist_ok=True)


def get_input_images() -> list[Path]:
    """
    Mengambil semua file gambar BMP dari folder data/raw_bmp.

    Return:
        list[Path]: Daftar path gambar BMP yang sudah diurutkan.

    Kenapa diurutkan?
    Supaya urutan pengujian konsisten, misalnya:
    img_01.bmp, img_02.bmp, img_03.bmp, dan seterusnya.
    """
    if not RAW_BMP_DIR.exists():
        return []

    return sorted(
        file_path
        for file_path in RAW_BMP_DIR.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == IMAGE_EXTENSION
    )


def validate_dataset() -> None:
    """
    Mengecek apakah dataset sudah memenuhi syarat dasar.

    Yang dicek:
    1. Folder data/raw_bmp harus ada.
    2. Jumlah gambar BMP minimal 20.
    3. Semua file input harus berekstensi .bmp.

    Jika ada masalah, fungsi ini akan melempar error yang jelas
    supaya mudah diperbaiki.
    """
    if not RAW_BMP_DIR.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {RAW_BMP_DIR}"
        )

    image_paths = get_input_images()

    if len(image_paths) < MINIMUM_IMAGE_COUNT:
        raise ValueError(
            f"Jumlah gambar BMP kurang. "
            f"Ditemukan {len(image_paths)} gambar, "
            f"minimal harus {MINIMUM_IMAGE_COUNT} gambar."
        )

    invalid_files = [
        file_path.name
        for file_path in RAW_BMP_DIR.iterdir()
        if file_path.is_file() and file_path.suffix.lower() != IMAGE_EXTENSION
    ]

    if invalid_files:
        raise ValueError(
            "Terdapat file selain .bmp di folder data/raw_bmp: "
            + ", ".join(invalid_files)
        )


def print_config_summary() -> None:
    """
    Menampilkan ringkasan konfigurasi project.

    Fungsi ini berguna untuk debugging awal, supaya kita tahu:
    - Project root terdeteksi benar.
    - Folder dataset benar.
    - Jumlah gambar terdeteksi benar.
    - Threshold algoritma benar.
    """
    image_paths = get_input_images()

    print("=== Quadtree BMP Compression Config ===")
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Dataset folder      : {RAW_BMP_DIR}")
    print(f"Jumlah gambar BMP   : {len(image_paths)}")
    print(f"Output folder       : {OUTPUT_DIR}")
    print(f"Results CSV         : {RESULTS_CSV_PATH}")
    print(f"Lossless threshold  : {LOSSLESS_THRESHOLD}")
    print(f"Lossy threshold     : {LOSSY_THRESHOLD}")
    print(f"Min block size      : {MIN_BLOCK_SIZE}")
    print("=======================================")