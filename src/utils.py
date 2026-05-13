"""
utils.py

File ini berisi fungsi-fungsi bantuan umum untuk project
Quadtree BMP Compression.

Isi file ini bukan inti algoritma Quadtree, tapi alat bantu untuk:
1. Membaca gambar BMP.
2. Menyimpan gambar BMP.
3. Mengambil daftar file gambar.
4. Mengecek ukuran file.
5. Membuat folder jika belum ada.
6. Membersihkan folder output dengan aman.

Dengan memisahkan fungsi bantu ke file ini, kode utama akan lebih:
- clean,
- maintainable,
- mudah dites,
- tidak banyak duplikasi.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import IMAGE_EXTENSION, IMAGE_MODE


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def ensure_directory(directory_path: Path) -> None:
    """
    Membuat folder jika folder belum ada.

    Args:
        directory_path (Path): Path folder yang ingin dibuat.

    Catatan:
        parents=True membuat parent folder ikut dibuat jika belum ada.
        exist_ok=True membuat program tidak error jika folder sudah ada.
    """
    directory_path.mkdir(parents=True, exist_ok=True)


def ensure_directories(directory_paths: Iterable[Path]) -> None:
    """
    Membuat banyak folder sekaligus.

    Args:
        directory_paths (Iterable[Path]): Kumpulan path folder.
    """
    for directory_path in directory_paths:
        ensure_directory(directory_path)


def clear_directory(directory_path: Path) -> None:
    """
    Menghapus semua isi folder output secara aman.

    Args:
        directory_path (Path): Folder yang ingin dibersihkan.

    Catatan keamanan:
        Fungsi ini hanya menghapus isi di dalam folder yang diberikan,
        bukan folder utamanya.

        Fungsi ini dipakai untuk membersihkan output lama agar hasil
        pengujian baru tidak tercampur dengan hasil sebelumnya.
    """
    if not directory_path.exists():
        ensure_directory(directory_path)
        return

    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path bukan folder: {directory_path}")

    for item_path in directory_path.iterdir():
        if item_path.is_file() or item_path.is_symlink():
            item_path.unlink()
        elif item_path.is_dir():
            shutil.rmtree(item_path)


# Import shutil diletakkan di bawah fungsi agar tetap jelas bahwa modul ini
# hanya dipakai untuk operasi folder tertentu.
import shutil


# ============================================================
# FILE HELPERS
# ============================================================

def get_file_size_bytes(file_path: Path) -> int:
    """
    Mengambil ukuran file dalam satuan bytes.

    Args:
        file_path (Path): Path file.

    Returns:
        int: Ukuran file dalam bytes.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path bukan file: {file_path}")

    return file_path.stat().st_size


def format_file_size(size_bytes: int) -> str:
    """
    Mengubah ukuran file dari bytes menjadi format yang mudah dibaca.

    Args:
        size_bytes (int): Ukuran file dalam bytes.

    Returns:
        str: Ukuran file dalam format B, KB, MB, atau GB.
    """
    if size_bytes < 0:
        raise ValueError("Ukuran file tidak boleh negatif.")

    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} GB"


def get_bmp_files(directory_path: Path) -> list[Path]:
    """
    Mengambil semua file BMP dari sebuah folder.

    Args:
        directory_path (Path): Folder tempat gambar BMP berada.

    Returns:
        list[Path]: Daftar file BMP yang sudah diurutkan.
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {directory_path}")

    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path bukan folder: {directory_path}")

    return sorted(
        file_path
        for file_path in directory_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == IMAGE_EXTENSION
    )


def build_output_path(
    input_path: Path,
    output_directory: Path,
    suffix: str,
    extension: str,
) -> Path:
    """
    Membuat path output berdasarkan nama file input.

    Contoh:
        input_path      = data/raw_bmp/img_01.bmp
        suffix          = "_lossy"
        extension       = ".qtree"

        hasil:
        output/img_01_lossy.qtree

    Args:
        input_path (Path): Path file input.
        output_directory (Path): Folder output.
        suffix (str): Tambahan nama file.
        extension (str): Ekstensi file output.

    Returns:
        Path: Path output final.
    """
    ensure_directory(output_directory)

    safe_extension = extension if extension.startswith(".") else f".{extension}"
    output_name = f"{input_path.stem}{suffix}{safe_extension}"

    return output_directory / output_name


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image_as_array(image_path: Path) -> np.ndarray:
    """
    Membaca gambar BMP dan mengubahnya menjadi array NumPy.

    Args:
        image_path (Path): Path gambar BMP.

    Returns:
        np.ndarray: Array gambar dengan bentuk (height, width, 3).

    Kenapa dikonversi ke RGB?
        Supaya semua gambar punya format channel warna yang konsisten.
        Jadi walaupun ada gambar yang terbaca sebagai mode lain, program
        tetap memprosesnya sebagai RGB.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    if image_path.suffix.lower() != IMAGE_EXTENSION:
        raise ValueError(f"File bukan BMP: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = image.convert(IMAGE_MODE)
            image_array = np.array(image, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ValueError(f"File tidak dikenali sebagai gambar valid: {image_path}") from exc

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            f"Format gambar tidak valid. Diharapkan RGB 3 channel: {image_path}"
        )

    return image_array


def save_array_as_bmp(image_array: np.ndarray, output_path: Path) -> None:
    """
    Menyimpan array NumPy sebagai gambar BMP.

    Args:
        image_array (np.ndarray): Array gambar dengan bentuk (height, width, 3).
        output_path (Path): Path tujuan file BMP.

    Catatan:
        Fungsi ini dipakai untuk menyimpan hasil rekonstruksi dari Quadtree.
    """
    validate_image_array(image_array)

    if output_path.suffix.lower() != IMAGE_EXTENSION:
        raise ValueError(f"Output harus berekstensi BMP: {output_path}")

    ensure_directory(output_path.parent)

    image = Image.fromarray(image_array.astype(np.uint8), mode=IMAGE_MODE)
    image.save(output_path)


def validate_image_array(image_array: np.ndarray) -> None:
    """
    Memastikan array gambar punya format yang benar.

    Args:
        image_array (np.ndarray): Array gambar yang ingin dicek.

    Format valid:
        - Bertipe numpy.ndarray.
        - Memiliki 3 dimensi.
        - Channel warna berjumlah 3.
        - Tipe data uint8.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array harus bertipe numpy.ndarray.")

    if image_array.ndim != 3:
        raise ValueError("image_array harus memiliki 3 dimensi: height, width, channel.")

    if image_array.shape[2] != 3:
        raise ValueError("image_array harus memiliki 3 channel warna RGB.")

    if image_array.dtype != np.uint8:
        raise ValueError("image_array harus bertipe uint8.")


def get_image_dimensions(image_array: np.ndarray) -> tuple[int, int]:
    """
    Mengambil ukuran gambar dari array.

    Args:
        image_array (np.ndarray): Array gambar.

    Returns:
        tuple[int, int]: Lebar dan tinggi gambar dalam format (width, height).
    """
    validate_image_array(image_array)

    height, width, _ = image_array.shape
    return width, height


# ============================================================
# TESTING HELPER
# ============================================================

def print_image_summary(image_path: Path) -> None:
    """
    Menampilkan ringkasan informasi satu gambar.

    Args:
        image_path (Path): Path gambar BMP.

    Fungsi ini berguna untuk memastikan gambar berhasil dibaca.
    """
    image_array = load_image_as_array(image_path)
    width, height = get_image_dimensions(image_array)
    file_size = get_file_size_bytes(image_path)

    print(f"Nama file     : {image_path.name}")
    print(f"Ukuran gambar : {width} x {height}")
    print(f"Ukuran file   : {format_file_size(file_size)}")