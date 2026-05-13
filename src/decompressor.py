"""
decompressor.py

File ini berisi proses dekompresi file .qtree menjadi gambar BMP.

Tugas utama file ini:
1. Membaca file hasil kompresi .qtree.
2. Mengecek apakah file .qtree valid.
3. Membaca metadata gambar.
4. Membangun kembali struktur Quadtree dari data binary.
5. Merekonstruksi gambar dari Quadtree.
6. Menyimpan hasil rekonstruksi sebagai file .bmp.

Catatan:
- File .qtree dibuat oleh compressor.py.
- Format yang dipakai adalah gzip + header JSON + binary preorder tree.
"""

from __future__ import annotations

import gzip
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Any

import numpy as np

from config import (
    COMPRESSED_FILE_EXTENSION,
    RECONSTRUCTED_LOSSLESS_DIR,
    RECONSTRUCTED_LOSSY_DIR,
)

from compressor import (
    FORMAT_VERSION,
    HEADER_LENGTH_STRUCT,
    MAGIC_BYTES,
    NODE_INTERNAL,
    NODE_LEAF,
)

from quadtree import (
    QuadTreeNode,
    reconstruct_image_from_quadtree,
    split_region,
)

from utils import (
    build_output_path,
    get_file_size_bytes,
    save_array_as_bmp,
)


# ============================================================
# RESULT DATACLASS
# ============================================================

@dataclass(slots=True)
class DecompressionResult:
    """
    Menyimpan ringkasan hasil dekompresi satu file .qtree.
    """

    input_path: Path
    output_path: Path
    mode: str
    width: int
    height: int
    compressed_size_bytes: int
    reconstructed_size_bytes: int
    processing_time_seconds: float


@dataclass(slots=True)
class QTreeFileData:
    """
    Menyimpan isi file .qtree setelah dibaca.

    metadata:
        Informasi gambar dan kompresi.

    root:
        Root node dari struktur Quadtree.
    """

    metadata: dict[str, Any]
    root: QuadTreeNode


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_qtree_path(qtree_path: Path) -> None:
    """
    Memastikan file input adalah file .qtree yang valid secara ekstensi.
    """

    if not qtree_path.exists():
        raise FileNotFoundError(f"File .qtree tidak ditemukan: {qtree_path}")

    if not qtree_path.is_file():
        raise ValueError(f"Path bukan file: {qtree_path}")

    if qtree_path.suffix.lower() != COMPRESSED_FILE_EXTENSION:
        raise ValueError(
            f"File harus berekstensi {COMPRESSED_FILE_EXTENSION}: {qtree_path}"
        )


def validate_metadata(metadata: dict[str, Any]) -> None:
    """
    Memastikan metadata di dalam file .qtree lengkap dan valid.
    """

    required_keys = {
        "format",
        "version",
        "algorithm",
        "mode",
        "original_filename",
        "width",
        "height",
        "channels",
        "threshold",
        "min_block_size",
        "total_nodes",
        "leaf_nodes",
        "tree_depth",
    }

    missing_keys = required_keys - set(metadata.keys())
    if missing_keys:
        raise ValueError(f"Metadata .qtree tidak lengkap: {missing_keys}")

    if metadata["format"] != "QTBMP":
        raise ValueError("Format file .qtree tidak valid.")

    if int(metadata["version"]) != FORMAT_VERSION:
        raise ValueError(
            f"Versi file .qtree tidak didukung: {metadata['version']}"
        )

    if metadata["mode"] not in {"lossless", "lossy"}:
        raise ValueError(f"Mode kompresi tidak valid: {metadata['mode']}")

    if int(metadata["channels"]) != 3:
        raise ValueError("File .qtree harus berisi gambar RGB 3 channel.")

    if int(metadata["width"]) <= 0 or int(metadata["height"]) <= 0:
        raise ValueError("Ukuran gambar pada metadata tidak valid.")


# ============================================================
# FILE READING HELPERS
# ============================================================

def read_exact(file: BinaryIO, size: int) -> bytes:
    """
    Membaca data binary dengan jumlah byte tertentu.

    Fungsi ini dibuat supaya pembacaan file lebih aman.
    Jika byte yang terbaca kurang dari yang diminta, berarti file rusak
    atau tidak lengkap.
    """

    data = file.read(size)

    if len(data) != size:
        raise ValueError("File .qtree rusak atau tidak lengkap.")

    return data


def read_qtree_file(qtree_path: Path) -> QTreeFileData:
    """
    Membaca file .qtree dan mengubahnya kembali menjadi metadata + Quadtree.

    Struktur file:
        gzip(
            MAGIC_BYTES
            HEADER_LENGTH
            HEADER_JSON
            BINARY_TREE_BODY
        )
    """

    validate_qtree_path(qtree_path)

    with gzip.open(qtree_path, "rb") as file:
        magic = read_exact(file, len(MAGIC_BYTES))

        if magic != MAGIC_BYTES:
            raise ValueError("Magic bytes tidak cocok. File bukan .qtree valid.")

        header_length_bytes = read_exact(file, struct.calcsize(HEADER_LENGTH_STRUCT))
        header_length = struct.unpack(HEADER_LENGTH_STRUCT, header_length_bytes)[0]

        header_bytes = read_exact(file, header_length)
        metadata = json.loads(header_bytes.decode("utf-8"))

        validate_metadata(metadata)

        body_bytes = file.read()

    width = int(metadata["width"])
    height = int(metadata["height"])

    root, final_offset = decode_node_preorder(
        data=body_bytes,
        offset=0,
        x=0,
        y=0,
        width=width,
        height=height,
    )

    if final_offset != len(body_bytes):
        raise ValueError(
            "File .qtree memiliki data tambahan yang tidak terbaca. "
            "Kemungkinan file rusak atau format tidak sesuai."
        )

    return QTreeFileData(metadata=metadata, root=root)


# ============================================================
# BINARY DESERIALIZATION
# ============================================================

def decode_node_preorder(
    data: bytes,
    offset: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[QuadTreeNode, int]:
    """
    Membaca satu node Quadtree dari data binary secara preorder.

    Format leaf:
        [NODE_LEAF][R][G][B]

    Format internal:
        [NODE_INTERNAL][JUMLAH_CHILD][CHILD_1][CHILD_2]...

    Args:
        data:
            Data binary tree.

        offset:
            Posisi byte yang sedang dibaca.

        x, y, width, height:
            Koordinat blok node saat ini.

    Returns:
        tuple[QuadTreeNode, int]:
            Node hasil decode dan offset terbaru.
    """

    if offset >= len(data):
        raise ValueError("Data binary .qtree habis sebelum node selesai dibaca.")

    node_type = data[offset]
    offset += 1

    if node_type == NODE_LEAF:
        return decode_leaf_node(
            data=data,
            offset=offset,
            x=x,
            y=y,
            width=width,
            height=height,
        )

    if node_type == NODE_INTERNAL:
        return decode_internal_node(
            data=data,
            offset=offset,
            x=x,
            y=y,
            width=width,
            height=height,
        )

    raise ValueError(f"Tipe node tidak dikenal: {node_type}")


def decode_leaf_node(
    data: bytes,
    offset: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[QuadTreeNode, int]:
    """
    Membaca node leaf dari data binary.

    Node leaf menyimpan satu warna RGB.
    """

    if offset + 3 > len(data):
        raise ValueError("Data warna leaf node tidak lengkap.")

    red = int(data[offset])
    green = int(data[offset + 1])
    blue = int(data[offset + 2])
    offset += 3

    node = QuadTreeNode(
        x=x,
        y=y,
        width=width,
        height=height,
        color=(red, green, blue),
        children=None,
    )

    return node, offset


def decode_internal_node(
    data: bytes,
    offset: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[QuadTreeNode, int]:
    """
    Membaca node internal dari data binary.

    Node internal punya beberapa child.
    Posisi dan ukuran child dihitung ulang memakai split_region().
    """

    if offset >= len(data):
        raise ValueError("Data jumlah child node internal tidak lengkap.")

    child_count = int(data[offset])
    offset += 1

    child_regions = split_region(x, y, width, height)

    if child_count != len(child_regions):
        raise ValueError(
            f"Jumlah child tidak cocok. "
            f"Ditemukan {child_count}, seharusnya {len(child_regions)}."
        )

    children: list[QuadTreeNode] = []

    for child_x, child_y, child_width, child_height in child_regions:
        child_node, offset = decode_node_preorder(
            data=data,
            offset=offset,
            x=child_x,
            y=child_y,
            width=child_width,
            height=child_height,
        )
        children.append(child_node)

    # Warna parent tidak dipakai untuk rekonstruksi.
    # Kita isi (0, 0, 0) karena warna asli parent tidak disimpan dalam binary.
    node = QuadTreeNode(
        x=x,
        y=y,
        width=width,
        height=height,
        color=(0, 0, 0),
        children=children,
    )

    return node, offset


# ============================================================
# DECOMPRESSION FUNCTIONS
# ============================================================

def decompress_qtree_file(
    qtree_path: Path,
    output_path: Path,
) -> DecompressionResult:
    """
    Mendekompresi satu file .qtree menjadi gambar BMP.
    """

    start_time = time.perf_counter()

    qtree_data = read_qtree_file(qtree_path)

    metadata = qtree_data.metadata
    root = qtree_data.root

    width = int(metadata["width"])
    height = int(metadata["height"])
    mode = str(metadata["mode"])

    image_shape = (height, width, 3)

    reconstructed_image = reconstruct_image_from_quadtree(
        root=root,
        image_shape=image_shape,
    )

    save_array_as_bmp(
        image_array=reconstructed_image,
        output_path=output_path,
    )

    end_time = time.perf_counter()

    return DecompressionResult(
        input_path=qtree_path,
        output_path=output_path,
        mode=mode,
        width=width,
        height=height,
        compressed_size_bytes=get_file_size_bytes(qtree_path),
        reconstructed_size_bytes=get_file_size_bytes(output_path),
        processing_time_seconds=end_time - start_time,
    )


def decompress_lossless(qtree_path: Path) -> DecompressionResult:
    """
    Mendekompresi file .qtree mode lossless menjadi BMP.
    """

    output_path = build_output_path(
        input_path=qtree_path,
        output_directory=RECONSTRUCTED_LOSSLESS_DIR,
        suffix="",
        extension=".bmp",
    )

    return decompress_qtree_file(
        qtree_path=qtree_path,
        output_path=output_path,
    )


def decompress_lossy(qtree_path: Path) -> DecompressionResult:
    """
    Mendekompresi file .qtree mode lossy menjadi BMP.
    """

    output_path = build_output_path(
        input_path=qtree_path,
        output_directory=RECONSTRUCTED_LOSSY_DIR,
        suffix="",
        extension=".bmp",
    )

    return decompress_qtree_file(
        qtree_path=qtree_path,
        output_path=output_path,
    )


def decompress_by_mode(qtree_path: Path) -> DecompressionResult:
    """
    Mendekompresi file .qtree otomatis berdasarkan metadata mode.

    Jika metadata mode = lossless:
        output masuk ke reconstructed_lossless.

    Jika metadata mode = lossy:
        output masuk ke reconstructed_lossy.
    """

    qtree_data = read_qtree_file(qtree_path)
    mode = str(qtree_data.metadata["mode"])

    if mode == "lossless":
        output_path = build_output_path(
            input_path=qtree_path,
            output_directory=RECONSTRUCTED_LOSSLESS_DIR,
            suffix="",
            extension=".bmp",
        )
    elif mode == "lossy":
        output_path = build_output_path(
            input_path=qtree_path,
            output_directory=RECONSTRUCTED_LOSSY_DIR,
            suffix="",
            extension=".bmp",
        )
    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")

    reconstructed_image = reconstruct_image_from_quadtree(
        root=qtree_data.root,
        image_shape=(
            int(qtree_data.metadata["height"]),
            int(qtree_data.metadata["width"]),
            3,
        ),
    )

    start_time = time.perf_counter()

    save_array_as_bmp(
        image_array=reconstructed_image,
        output_path=output_path,
    )

    end_time = time.perf_counter()

    return DecompressionResult(
        input_path=qtree_path,
        output_path=output_path,
        mode=mode,
        width=int(qtree_data.metadata["width"]),
        height=int(qtree_data.metadata["height"]),
        compressed_size_bytes=get_file_size_bytes(qtree_path),
        reconstructed_size_bytes=get_file_size_bytes(output_path),
        processing_time_seconds=end_time - start_time,
    )