"""
quadtree.py

File ini berisi inti algoritma Quadtree Image Compression.

Konsep Quadtree:
1. Gambar dibagi menjadi blok.
2. Setiap blok dicek apakah warnanya cukup seragam.
3. Jika seragam, blok disimpan sebagai satu warna.
4. Jika belum seragam, blok dipecah menjadi 4 bagian:
   - kiri atas
   - kanan atas
   - kiri bawah
   - kanan bawah
5. Proses ini dilakukan berulang sampai blok cukup seragam
   atau ukuran blok sudah mencapai batas minimum.

Mode lossless:
- threshold = 0
- satu blok hanya dianggap seragam jika semua pikselnya benar-benar sama.

Mode lossy:
- threshold > 0
- satu blok boleh dianggap seragam jika perbedaan warnanya masih kecil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import MIN_BLOCK_SIZE


# ============================================================
# QUADTREE NODE
# ============================================================

@dataclass(slots=True)
class QuadTreeNode:
    """
    Merepresentasikan satu node pada struktur Quadtree.

    Atribut:
        x (int):
            Posisi kolom awal blok pada gambar.

        y (int):
            Posisi baris awal blok pada gambar.

        width (int):
            Lebar blok.

        height (int):
            Tinggi blok.

        color (tuple[int, int, int]):
            Warna rata-rata atau warna tunggal blok dalam format RGB.

        children (list[QuadTreeNode] | None):
            Anak node jika blok masih dipecah.
            Jika None, berarti node ini adalah leaf node.
    """

    x: int
    y: int
    width: int
    height: int
    color: tuple[int, int, int]
    children: list["QuadTreeNode"] | None = None

    @property
    def is_leaf(self) -> bool:
        """
        Mengecek apakah node ini adalah leaf node.

        Leaf node berarti blok tidak dipecah lagi dan akan direkonstruksi
        menggunakan satu warna saja.
        """
        return self.children is None or len(self.children) == 0


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_block_coordinates(
    image_array: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    """
    Memastikan koordinat blok masih berada di dalam batas gambar.

    Args:
        image_array (np.ndarray): Array gambar RGB.
        x (int): Posisi kolom awal.
        y (int): Posisi baris awal.
        width (int): Lebar blok.
        height (int): Tinggi blok.
    """
    image_height, image_width, channels = image_array.shape

    if channels != 3:
        raise ValueError("Gambar harus memiliki 3 channel RGB.")

    if x < 0 or y < 0:
        raise ValueError("Koordinat x dan y tidak boleh negatif.")

    if width <= 0 or height <= 0:
        raise ValueError("Width dan height harus lebih dari 0.")

    if x + width > image_width or y + height > image_height:
        raise ValueError("Koordinat blok keluar dari batas gambar.")


def validate_threshold(threshold: float) -> None:
    """
    Memastikan nilai threshold valid.

    Args:
        threshold (float): Nilai batas toleransi warna.
    """
    if threshold < 0:
        raise ValueError("Threshold tidak boleh negatif.")


# ============================================================
# COLOR AND ERROR HELPERS
# ============================================================

def get_block(
    image_array: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Mengambil potongan blok dari gambar.

    Args:
        image_array (np.ndarray): Array gambar RGB.
        x (int): Posisi kolom awal.
        y (int): Posisi baris awal.
        width (int): Lebar blok.
        height (int): Tinggi blok.

    Returns:
        np.ndarray: Blok gambar.
    """
    validate_block_coordinates(image_array, x, y, width, height)
    return image_array[y:y + height, x:x + width]


def calculate_mean_color(block: np.ndarray) -> tuple[int, int, int]:
    """
    Menghitung warna rata-rata dari sebuah blok.

    Args:
        block (np.ndarray): Blok gambar RGB.

    Returns:
        tuple[int, int, int]: Warna rata-rata RGB.

    Catatan:
        Warna dibulatkan ke integer karena nilai RGB harus 0-255.
    """
    mean_color = np.mean(block, axis=(0, 1))
    mean_color = np.clip(np.round(mean_color), 0, 255).astype(np.uint8)

    return tuple(int(value) for value in mean_color)


def calculate_block_rmse(
    block: np.ndarray,
    color: tuple[int, int, int],
) -> float:
    """
    Menghitung error warna blok terhadap satu warna representatif.

    RMSE = Root Mean Squared Error.

    Semakin kecil RMSE:
    - warna dalam blok semakin seragam,
    - blok semakin layak disimpan sebagai satu warna.

    Semakin besar RMSE:
    - warna dalam blok semakin bervariasi,
    - blok sebaiknya dipecah lagi.

    Args:
        block (np.ndarray): Blok gambar RGB.
        color (tuple[int, int, int]): Warna pembanding RGB.

    Returns:
        float: Nilai RMSE blok.
    """
    color_array = np.array(color, dtype=np.float32)
    block_float = block.astype(np.float32)

    difference = block_float - color_array
    mse = np.mean(difference ** 2)

    return float(np.sqrt(mse))


def is_lossless_uniform(block: np.ndarray) -> tuple[bool, tuple[int, int, int]]:
    """
    Mengecek apakah blok seragam secara lossless.

    Pada mode lossless, semua piksel harus benar-benar sama.
    Jika ada satu piksel saja yang berbeda, blok tidak boleh digabung.

    Args:
        block (np.ndarray): Blok gambar RGB.

    Returns:
        tuple[bool, tuple[int, int, int]]:
            - True jika semua piksel identik.
            - Warna RGB piksel pertama sebagai warna blok.
    """
    first_color = block[0, 0]
    is_uniform = bool(np.all(block == first_color))

    color = tuple(int(value) for value in first_color)
    return is_uniform, color


def is_lossy_uniform(
    block: np.ndarray,
    threshold: float,
) -> tuple[bool, tuple[int, int, int], float]:
    """
    Mengecek apakah blok cukup seragam untuk mode lossy.

    Pada mode lossy, blok boleh digabung jika nilai RMSE <= threshold.

    Args:
        block (np.ndarray): Blok gambar RGB.
        threshold (float): Batas toleransi error warna.

    Returns:
        tuple[bool, tuple[int, int, int], float]:
            - True jika blok cukup seragam.
            - Warna rata-rata blok.
            - Nilai RMSE blok.
    """
    mean_color = calculate_mean_color(block)
    rmse = calculate_block_rmse(block, mean_color)

    return rmse <= threshold, mean_color, rmse


# ============================================================
# SPLIT HELPER
# ============================================================

def split_region(
    x: int,
    y: int,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    """
    Membagi satu blok menjadi maksimal 4 bagian.

    Fungsi ini aman untuk gambar dengan ukuran ganjil.
    Contoh:
        width = 5 akan dibagi menjadi 2 dan 3.
        height = 7 akan dibagi menjadi 3 dan 4.

    Args:
        x (int): Posisi kolom awal blok.
        y (int): Posisi baris awal blok.
        width (int): Lebar blok.
        height (int): Tinggi blok.

    Returns:
        list[tuple[int, int, int, int]]:
            Daftar blok dalam format (x, y, width, height).
    """
    half_width = width // 2
    half_height = height // 2

    right_width = width - half_width
    bottom_height = height - half_height

    regions: list[tuple[int, int, int, int]] = []

    # Kiri atas
    if half_width > 0 and half_height > 0:
        regions.append((x, y, half_width, half_height))

    # Kanan atas
    if right_width > 0 and half_height > 0:
        regions.append((x + half_width, y, right_width, half_height))

    # Kiri bawah
    if half_width > 0 and bottom_height > 0:
        regions.append((x, y + half_height, half_width, bottom_height))

    # Kanan bawah
    if right_width > 0 and bottom_height > 0:
        regions.append((x + half_width, y + half_height, right_width, bottom_height))

    return regions


# ============================================================
# QUADTREE BUILDING
# ============================================================

def build_quadtree(
    image_array: np.ndarray,
    x: int = 0,
    y: int = 0,
    width: int | None = None,
    height: int | None = None,
    threshold: float = 0.0,
    min_block_size: int = MIN_BLOCK_SIZE,
) -> QuadTreeNode:
    """
    Membangun struktur Quadtree dari gambar.

    Args:
        image_array (np.ndarray):
            Array gambar RGB.

        x (int):
            Posisi kolom awal blok.

        y (int):
            Posisi baris awal blok.

        width (int | None):
            Lebar blok. Jika None, pakai lebar gambar.

        height (int | None):
            Tinggi blok. Jika None, pakai tinggi gambar.

        threshold (float):
            Batas toleransi warna.
            - 0.0 untuk lossless.
            - > 0.0 untuk lossy.

        min_block_size (int):
            Ukuran minimum blok yang boleh dibuat.

    Returns:
        QuadTreeNode:
            Root node dari struktur Quadtree.
    """
    validate_threshold(threshold)

    image_height, image_width, channels = image_array.shape

    if channels != 3:
        raise ValueError("Gambar harus RGB dengan 3 channel.")

    if width is None:
        width = image_width

    if height is None:
        height = image_height

    validate_block_coordinates(image_array, x, y, width, height)

    block = get_block(image_array, x, y, width, height)

    # Mode lossless:
    # threshold 0 berarti blok hanya boleh digabung jika semua piksel identik.
    if threshold == 0:
        is_uniform, block_color = is_lossless_uniform(block)
        block_error = 0.0
    else:
        is_uniform, block_color, block_error = is_lossy_uniform(block, threshold)

    # Kondisi berhenti:
    # 1. Blok sudah seragam.
    # 2. Blok sudah mencapai ukuran minimum di kedua sisi.
    #
    # Penting:
    # Jangan pakai OR di sini.
    # Kalau width = 1 tapi height masih > 1, blok masih bisa dipecah vertikal.
    # Kalau height = 1 tapi width masih > 1, blok masih bisa dipecah horizontal.
    # Blok benar-benar tidak bisa dipecah lagi hanya saat width dan height
    # sama-sama sudah mencapai min_block_size.
    reached_min_size = width <= min_block_size and height <= min_block_size

    if is_uniform or reached_min_size:
        return QuadTreeNode(
            x=x,
            y=y,
            width=width,
            height=height,
            color=block_color,
            children=None,
        )

    # Jika belum seragam, blok dipecah menjadi beberapa region.
    child_regions = split_region(x, y, width, height)

    children = [
        build_quadtree(
            image_array=image_array,
            x=child_x,
            y=child_y,
            width=child_width,
            height=child_height,
            threshold=threshold,
            min_block_size=min_block_size,
        )
        for child_x, child_y, child_width, child_height in child_regions
    ]

    # Warna parent tetap disimpan sebagai warna rata-rata blok.
    # Ini berguna untuk analisis/debugging, walaupun saat rekonstruksi
    # warna leaf node yang dipakai.
    return QuadTreeNode(
        x=x,
        y=y,
        width=width,
        height=height,
        color=block_color,
        children=children,
    )


# ============================================================
# IMAGE RECONSTRUCTION
# ============================================================

def reconstruct_image_from_quadtree(
    root: QuadTreeNode,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Merekonstruksi gambar dari struktur Quadtree.

    Args:
        root (QuadTreeNode):
            Root node Quadtree.

        image_shape (tuple[int, int, int]):
            Bentuk gambar asli dalam format (height, width, channels).

    Returns:
        np.ndarray:
            Gambar hasil rekonstruksi dalam format array RGB.
    """
    if len(image_shape) != 3 or image_shape[2] != 3:
        raise ValueError("image_shape harus berbentuk (height, width, 3).")

    reconstructed = np.zeros(image_shape, dtype=np.uint8)
    fill_node_region(reconstructed, root)

    return reconstructed


def fill_node_region(
    image_array: np.ndarray,
    node: QuadTreeNode,
) -> None:
    """
    Mengisi area gambar berdasarkan node Quadtree.

    Jika node adalah leaf:
        Area blok diisi dengan warna node.

    Jika node punya children:
        Fungsi dipanggil rekursif untuk setiap child.

    Args:
        image_array (np.ndarray): Array gambar hasil rekonstruksi.
        node (QuadTreeNode): Node Quadtree.
    """
    if node.is_leaf:
        image_array[
            node.y:node.y + node.height,
            node.x:node.x + node.width,
        ] = np.array(node.color, dtype=np.uint8)
        return

    if node.children is None:
        return

    for child in node.children:
        fill_node_region(image_array, child)


# ============================================================
# TREE STATISTICS
# ============================================================

def count_nodes(node: QuadTreeNode) -> int:
    """
    Menghitung total node pada Quadtree.

    Args:
        node (QuadTreeNode): Root node.

    Returns:
        int: Jumlah semua node.
    """
    if node.is_leaf:
        return 1

    return 1 + sum(count_nodes(child) for child in node.children or [])


def count_leaf_nodes(node: QuadTreeNode) -> int:
    """
    Menghitung jumlah leaf node pada Quadtree.

    Leaf node adalah blok akhir yang tidak dipecah lagi.

    Args:
        node (QuadTreeNode): Root node.

    Returns:
        int: Jumlah leaf node.
    """
    if node.is_leaf:
        return 1

    return sum(count_leaf_nodes(child) for child in node.children or [])


def calculate_tree_depth(node: QuadTreeNode) -> int:
    """
    Menghitung kedalaman maksimum Quadtree.

    Args:
        node (QuadTreeNode): Root node.

    Returns:
        int: Kedalaman maksimum tree.
    """
    if node.is_leaf:
        return 1

    return 1 + max(calculate_tree_depth(child) for child in node.children or [])


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def quadtree_to_dict(node: QuadTreeNode) -> dict[str, Any]:
    """
    Mengubah QuadtreeNode menjadi dictionary.

    Fungsi ini penting supaya struktur Quadtree bisa disimpan
    ke file kompresi custom seperti .qtree.

    Args:
        node (QuadTreeNode): Node Quadtree.

    Returns:
        dict[str, Any]: Representasi dictionary dari node.
    """
    return {
        "x": node.x,
        "y": node.y,
        "width": node.width,
        "height": node.height,
        "color": node.color,
        "children": (
            [quadtree_to_dict(child) for child in node.children]
            if node.children
            else None
        ),
    }


def dict_to_quadtree(data: dict[str, Any]) -> QuadTreeNode:
    """
    Mengubah dictionary kembali menjadi QuadtreeNode.

    Fungsi ini dipakai saat membaca file .qtree untuk dekompresi.

    Args:
        data (dict[str, Any]): Dictionary node.

    Returns:
        QuadTreeNode: Node Quadtree.
    """
    required_keys = {"x", "y", "width", "height", "color", "children"}

    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Data node tidak lengkap. Missing keys: {missing_keys}")

    children_data = data["children"]

    children = None
    if children_data is not None:
        children = [dict_to_quadtree(child_data) for child_data in children_data]

    color = tuple(int(value) for value in data["color"])

    return QuadTreeNode(
        x=int(data["x"]),
        y=int(data["y"]),
        width=int(data["width"]),
        height=int(data["height"]),
        color=color,
        children=children,
    )