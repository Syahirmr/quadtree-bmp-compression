"""
compressor.py

File ini berisi proses kompresi gambar BMP menggunakan algoritma Quadtree.

Tugas utama file ini:
1. Membaca gambar BMP.
2. Membangun struktur Quadtree.
3. Menyimpan hasil Quadtree ke file kompresi custom berekstensi .qtree.
4. Menyediakan fungsi kompresi lossless dan lossy.

Catatan keamanan:
- File kompresi tidak disimpan memakai pickle.
- Pickle dihindari karena bisa berbahaya jika membuka file dari sumber tidak tepercaya.
- Project ini memakai format binary custom + gzip agar lebih aman dan lebih ringkas.
"""

from __future__ import annotations

import gzip
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from config import (
    COMPRESSED_FILE_EXTENSION,
    COMPRESSED_LOSSLESS_DIR,
    COMPRESSED_LOSSY_DIR,
    LOSSLESS_THRESHOLD,
    LOSSY_THRESHOLD,
    MIN_BLOCK_SIZE,
    OVERWRITE_OUTPUT,
)

from quadtree import (
    QuadTreeNode,
    build_quadtree,
    calculate_tree_depth,
    count_leaf_nodes,
    count_nodes,
)

from utils import (
    build_output_path,
    get_file_size_bytes,
    load_image_as_array,
)


# ============================================================
# FORMAT CONFIG
# ============================================================

# Magic bytes dipakai sebagai penanda bahwa file ini adalah file .qtree valid.
# Nanti saat dekompresi, file akan dicek apakah diawali dengan magic bytes ini.
MAGIC_BYTES: bytes = b"QTBMP1\n"

# Versi format file kompresi.
# Jika format file berubah di masa depan, versi ini bisa dinaikkan.
FORMAT_VERSION: int = 1

# Ukuran field header length.
# Kita pakai unsigned int 4 byte big-endian untuk menyimpan panjang header JSON.
HEADER_LENGTH_STRUCT: str = ">I"

# Penanda node internal.
# Node internal berarti blok masih punya anak/children.
NODE_INTERNAL: int = 0

# Penanda node leaf.
# Node leaf berarti blok akhir yang disimpan sebagai satu warna.
NODE_LEAF: int = 1


# ============================================================
# RESULT DATACLASS
# ============================================================

@dataclass(slots=True)
class CompressionResult:
    """
    Menyimpan ringkasan hasil kompresi satu gambar.

    Dataclass ini memudahkan kita membawa data hasil kompresi
    ke file lain seperti metrics.py atau main.py.
    """

    input_path: Path
    output_path: Path
    mode: str
    width: int
    height: int
    threshold: float
    original_size_bytes: int
    compressed_size_bytes: int
    total_nodes: int
    leaf_nodes: int
    tree_depth: int
    processing_time_seconds: float

    @property
    def compression_ratio(self) -> float:
        """
        Menghitung rasio kompresi.

        Formula:
            compression_ratio = ukuran_asli / ukuran_kompresi

        Nilai lebih besar berarti hasil kompresi lebih baik.
        """
        if self.compressed_size_bytes == 0:
            return 0.0

        return self.original_size_bytes / self.compressed_size_bytes

    @property
    def saving_percentage(self) -> float:
        """
        Menghitung persentase penghematan ukuran file.

        Formula:
            saving = (1 - ukuran_kompresi / ukuran_asli) * 100
        """
        if self.original_size_bytes == 0:
            return 0.0

        return (1 - (self.compressed_size_bytes / self.original_size_bytes)) * 100


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_compression_mode(mode: str) -> None:
    """
    Memastikan mode kompresi hanya lossless atau lossy.

    Args:
        mode (str): Mode kompresi.
    """
    valid_modes = {"lossless", "lossy"}

    if mode not in valid_modes:
        raise ValueError(
            f"Mode kompresi tidak valid: {mode}. "
            f"Gunakan salah satu dari: {', '.join(sorted(valid_modes))}"
        )


def validate_output_path(output_path: Path, overwrite: bool = OVERWRITE_OUTPUT) -> None:
    """
    Mengecek apakah file output aman untuk ditulis.

    Args:
        output_path (Path): Path file output.
        overwrite (bool): Jika True, file lama boleh ditimpa.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"File output sudah ada dan overwrite dimatikan: {output_path}"
        )

    if output_path.suffix.lower() != COMPRESSED_FILE_EXTENSION:
        raise ValueError(
            f"File hasil kompresi harus berekstensi {COMPRESSED_FILE_EXTENSION}: "
            f"{output_path}"
        )


def validate_color(color: tuple[int, int, int]) -> None:
    """
    Memastikan warna RGB valid.

    Args:
        color (tuple[int, int, int]): Warna RGB.
    """
    if len(color) != 3:
        raise ValueError("Warna harus berisi 3 nilai RGB.")

    for value in color:
        if not 0 <= int(value) <= 255:
            raise ValueError(f"Nilai warna tidak valid: {value}")


# ============================================================
# METADATA HELPERS
# ============================================================

def build_metadata(
    image_path: Path,
    mode: str,
    width: int,
    height: int,
    threshold: float,
    root: QuadTreeNode,
) -> dict:
    """
    Membuat metadata untuk disimpan di dalam file .qtree.

    Metadata ini penting supaya saat dekompresi nanti program tahu:
    - ukuran gambar asli,
    - mode kompresi,
    - threshold,
    - jumlah node,
    - kedalaman tree,
    - nama file asal.
    """
    return {
        "format": "QTBMP",
        "version": FORMAT_VERSION,
        "algorithm": "Quadtree Image Compression",
        "mode": mode,
        "original_filename": image_path.name,
        "width": width,
        "height": height,
        "channels": 3,
        "threshold": threshold,
        "min_block_size": MIN_BLOCK_SIZE,
        "total_nodes": count_nodes(root),
        "leaf_nodes": count_leaf_nodes(root),
        "tree_depth": calculate_tree_depth(root),
    }


# ============================================================
# BINARY SERIALIZATION
# ============================================================

def encode_node_preorder(node: QuadTreeNode, buffer: bytearray) -> None:
    """
    Mengubah node Quadtree menjadi data binary secara preorder.

    Preorder berarti:
    1. Simpan node saat ini.
    2. Simpan child pertama.
    3. Simpan child kedua.
    4. Dan seterusnya.

    Format node leaf:
        [NODE_LEAF][R][G][B]

    Format node internal:
        [NODE_INTERNAL][JUMLAH_CHILD][DATA_CHILD_1][DATA_CHILD_2]...

    Kenapa tidak menyimpan x, y, width, height setiap node?
        Karena posisi dan ukuran node bisa dihitung ulang saat dekompresi
        menggunakan aturan split Quadtree yang sama.
        Ini membuat file .qtree jauh lebih kecil.
    """
    if node.is_leaf:
        validate_color(node.color)

        buffer.append(NODE_LEAF)
        buffer.extend(bytes(node.color))
        return

    children = node.children or []

    if len(children) > 255:
        raise ValueError("Jumlah child terlalu banyak untuk format binary ini.")

    buffer.append(NODE_INTERNAL)
    buffer.append(len(children))

    for child in children:
        encode_node_preorder(child, buffer)


def build_binary_body(root: QuadTreeNode) -> bytes:
    """
    Membuat binary body dari struktur Quadtree.

    Args:
        root (QuadTreeNode): Root node Quadtree.

    Returns:
        bytes: Data binary node Quadtree.
    """
    buffer = bytearray()
    encode_node_preorder(root, buffer)

    return bytes(buffer)


def write_qtree_file(
    output_path: Path,
    metadata: dict,
    root: QuadTreeNode,
) -> None:
    """
    Menyimpan hasil kompresi Quadtree ke file .qtree.

    Struktur file:
        gzip(
            MAGIC_BYTES
            HEADER_LENGTH
            HEADER_JSON
            BINARY_TREE_BODY
        )

    Args:
        output_path (Path): Path tujuan file .qtree.
        metadata (dict): Metadata gambar dan kompresi.
        root (QuadTreeNode): Root node Quadtree.
    """
    validate_output_path(output_path)

    # Pastikan folder output ada.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON header dibuat compact agar tidak boros ukuran.
    header_bytes = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    # Panjang header disimpan dalam 4 byte.
    header_length = struct.pack(HEADER_LENGTH_STRUCT, len(header_bytes))

    # Body berisi struktur node Quadtree dalam format binary custom.
    body_bytes = build_binary_body(root)

    # Semua data dibungkus gzip agar lebih ringkas.
    with gzip.open(output_path, "wb", compresslevel=9) as file:
        write_binary_qtree_content(
            file=file,
            header_length=header_length,
            header_bytes=header_bytes,
            body_bytes=body_bytes,
        )


def write_binary_qtree_content(
    file: BinaryIO,
    header_length: bytes,
    header_bytes: bytes,
    body_bytes: bytes,
) -> None:
    """
    Menulis konten binary ke file .qtree.

    Fungsi ini dipisah agar struktur penulisan file lebih mudah dibaca.
    """
    file.write(MAGIC_BYTES)
    file.write(header_length)
    file.write(header_bytes)
    file.write(body_bytes)


# ============================================================
# MAIN COMPRESSION FUNCTIONS
# ============================================================

def compress_image(
    image_path: Path,
    output_path: Path,
    mode: str,
    threshold: float,
) -> CompressionResult:
    """
    Mengompresi satu gambar BMP menjadi file .qtree.

    Args:
        image_path (Path): Path gambar BMP input.
        output_path (Path): Path file .qtree output.
        mode (str): Mode kompresi, yaitu lossless atau lossy.
        threshold (float): Threshold Quadtree.

    Returns:
        CompressionResult: Ringkasan hasil kompresi.
    """
    validate_compression_mode(mode)

    start_time = time.perf_counter()

    # Baca gambar sebagai array RGB.
    image_array = load_image_as_array(image_path)

    # Ambil ukuran gambar.
    height, width, channels = image_array.shape

    if channels != 3:
        raise ValueError("Gambar harus RGB dengan 3 channel.")

    # Bangun struktur Quadtree.
    root = build_quadtree(
        image_array=image_array,
        threshold=threshold,
        min_block_size=MIN_BLOCK_SIZE,
    )

    # Buat metadata untuk file .qtree.
    metadata = build_metadata(
        image_path=image_path,
        mode=mode,
        width=width,
        height=height,
        threshold=threshold,
        root=root,
    )

    # Simpan Quadtree ke file .qtree.
    write_qtree_file(
        output_path=output_path,
        metadata=metadata,
        root=root,
    )

    end_time = time.perf_counter()

    return CompressionResult(
        input_path=image_path,
        output_path=output_path,
        mode=mode,
        width=width,
        height=height,
        threshold=threshold,
        original_size_bytes=get_file_size_bytes(image_path),
        compressed_size_bytes=get_file_size_bytes(output_path),
        total_nodes=metadata["total_nodes"],
        leaf_nodes=metadata["leaf_nodes"],
        tree_depth=metadata["tree_depth"],
        processing_time_seconds=end_time - start_time,
    )


def compress_lossless(image_path: Path) -> CompressionResult:
    """
    Mengompresi gambar BMP menggunakan mode lossless.

    Lossless:
    - threshold = 0
    - blok hanya digabung jika semua piksel benar-benar identik.
    """
    output_path = build_output_path(
        input_path=image_path,
        output_directory=COMPRESSED_LOSSLESS_DIR,
        suffix="_lossless",
        extension=COMPRESSED_FILE_EXTENSION,
    )

    return compress_image(
        image_path=image_path,
        output_path=output_path,
        mode="lossless",
        threshold=LOSSLESS_THRESHOLD,
    )


def compress_lossy(image_path: Path) -> CompressionResult:
    """
    Mengompresi gambar BMP menggunakan mode lossy.

    Lossy:
    - threshold > 0
    - blok boleh digabung jika perbedaan warnanya masih di bawah threshold.
    """
    output_path = build_output_path(
        input_path=image_path,
        output_directory=COMPRESSED_LOSSY_DIR,
        suffix="_lossy",
        extension=COMPRESSED_FILE_EXTENSION,
    )

    return compress_image(
        image_path=image_path,
        output_path=output_path,
        mode="lossy",
        threshold=LOSSY_THRESHOLD,
    )


def compress_images(
    image_paths: Iterable[Path],
    mode: str,
) -> list[CompressionResult]:
    """
    Mengompresi banyak gambar sekaligus.

    Args:
        image_paths (Iterable[Path]): Daftar gambar BMP.
        mode (str): Mode kompresi, lossless atau lossy.

    Returns:
        list[CompressionResult]: Daftar hasil kompresi.
    """
    validate_compression_mode(mode)

    results: list[CompressionResult] = []

    for image_path in image_paths:
        if mode == "lossless":
            result = compress_lossless(image_path)
        else:
            result = compress_lossy(image_path)

        results.append(result)

    return results