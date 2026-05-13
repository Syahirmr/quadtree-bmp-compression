"""
metrics.py

File ini berisi fungsi-fungsi untuk menghitung hasil pengujian
kompresi gambar Quadtree.

Metrik yang dihitung:
1. Ukuran file asli.
2. Ukuran file hasil kompresi .qtree.
3. Compression ratio.
4. Saving percentage.
5. MSE.
6. RMSE.
7. MAE.
8. PSNR.
9. Status apakah hasil rekonstruksi identik dengan gambar asli.

Catatan penting:
- Ukuran hasil kompresi yang dipakai adalah ukuran file .qtree.
- Gambar hasil dekompresi BMP biasanya kembali besar karena sudah menjadi
  bitmap utuh lagi.
- Untuk lossless, MSE seharusnya 0 dan gambar harus identik.
- Untuk lossy, MSE biasanya > 0 karena ada penurunan kualitas.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_CSV_PATH
from utils import (
    get_file_size_bytes,
    load_image_as_array,
)


# ============================================================
# DATACLASS RESULT
# ============================================================

@dataclass(slots=True)
class QualityMetrics:
    """
    Menyimpan hasil perhitungan kualitas gambar.

    Atribut:
        mse:
            Mean Squared Error.
            Semakin kecil semakin bagus.

        rmse:
            Root Mean Squared Error.
            Akar dari MSE.

        mae:
            Mean Absolute Error.
            Rata-rata selisih absolut piksel.

        psnr:
            Peak Signal-to-Noise Ratio.
            Semakin besar semakin bagus.
            Jika gambar identik, nilainya infinity.

        is_identical:
            True jika gambar asli dan gambar rekonstruksi sama persis.
    """

    mse: float
    rmse: float
    mae: float
    psnr: float
    is_identical: bool


@dataclass(slots=True)
class EvaluationResult:
    """
    Menyimpan satu baris hasil evaluasi kompresi gambar.

    Dataclass ini nanti bisa langsung diubah menjadi dictionary
    lalu disimpan ke CSV.
    """

    image_name: str
    mode: str

    original_size_bytes: int
    compressed_size_bytes: int
    reconstructed_size_bytes: int

    compression_ratio: float
    saving_percentage: float

    mse: float
    rmse: float
    mae: float
    psnr: float
    is_identical: bool

    width: int
    height: int

    total_nodes: int | None
    leaf_nodes: int | None
    tree_depth: int | None

    compression_time_seconds: float | None
    decompression_time_seconds: float | None


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_same_shape(
    original_array: np.ndarray,
    reconstructed_array: np.ndarray,
) -> None:
    """
    Memastikan gambar asli dan gambar rekonstruksi punya ukuran sama.

    Args:
        original_array:
            Array gambar asli.

        reconstructed_array:
            Array gambar hasil dekompresi.
    """

    if original_array.shape != reconstructed_array.shape:
        raise ValueError(
            "Ukuran gambar asli dan rekonstruksi berbeda. "
            f"Original: {original_array.shape}, "
            f"Reconstructed: {reconstructed_array.shape}"
        )


def validate_file_exists(file_path: Path) -> None:
    """
    Memastikan file benar-benar ada sebelum dihitung.

    Args:
        file_path:
            Path file yang dicek.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path bukan file: {file_path}")


# ============================================================
# IMAGE QUALITY METRICS
# ============================================================

def calculate_mse(
    original_array: np.ndarray,
    reconstructed_array: np.ndarray,
) -> float:
    """
    Menghitung Mean Squared Error antara dua gambar.

    MSE menghitung rata-rata kuadrat selisih nilai piksel.

    Nilai:
        0     = gambar sama persis.
        > 0   = ada perbedaan antara gambar asli dan hasil rekonstruksi.
    """

    validate_same_shape(original_array, reconstructed_array)

    original_float = original_array.astype(np.float64)
    reconstructed_float = reconstructed_array.astype(np.float64)

    difference = original_float - reconstructed_float
    mse = np.mean(difference ** 2)

    return float(mse)


def calculate_rmse(mse: float) -> float:
    """
    Menghitung Root Mean Squared Error dari MSE.
    """

    if mse < 0:
        raise ValueError("MSE tidak boleh negatif.")

    return float(math.sqrt(mse))


def calculate_mae(
    original_array: np.ndarray,
    reconstructed_array: np.ndarray,
) -> float:
    """
    Menghitung Mean Absolute Error antara dua gambar.

    MAE menghitung rata-rata selisih absolut piksel.
    """

    validate_same_shape(original_array, reconstructed_array)

    original_float = original_array.astype(np.float64)
    reconstructed_float = reconstructed_array.astype(np.float64)

    mae = np.mean(np.abs(original_float - reconstructed_float))

    return float(mae)


def calculate_psnr(mse: float, max_pixel_value: float = 255.0) -> float:
    """
    Menghitung Peak Signal-to-Noise Ratio.

    PSNR umum dipakai untuk menilai kualitas hasil kompresi gambar.

    Interpretasi umum:
        - Semakin tinggi PSNR, kualitas semakin baik.
        - Jika MSE = 0, gambar identik dan PSNR dianggap infinity.

    Formula:
        PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))
    """

    if mse < 0:
        raise ValueError("MSE tidak boleh negatif.")

    if mse == 0:
        return float("inf")

    return float(20 * math.log10(max_pixel_value / math.sqrt(mse)))


def calculate_quality_metrics(
    original_image_path: Path,
    reconstructed_image_path: Path,
) -> QualityMetrics:
    """
    Menghitung semua metrik kualitas gambar.

    Args:
        original_image_path:
            Path gambar BMP asli.

        reconstructed_image_path:
            Path gambar BMP hasil dekompresi.

    Returns:
        QualityMetrics:
            Hasil MSE, RMSE, MAE, PSNR, dan status identik.
    """

    validate_file_exists(original_image_path)
    validate_file_exists(reconstructed_image_path)

    original_array = load_image_as_array(original_image_path)
    reconstructed_array = load_image_as_array(reconstructed_image_path)

    validate_same_shape(original_array, reconstructed_array)

    mse = calculate_mse(original_array, reconstructed_array)
    rmse = calculate_rmse(mse)
    mae = calculate_mae(original_array, reconstructed_array)
    psnr = calculate_psnr(mse)

    is_identical = bool(np.array_equal(original_array, reconstructed_array))

    return QualityMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        psnr=psnr,
        is_identical=is_identical,
    )


# ============================================================
# COMPRESSION METRICS
# ============================================================

def calculate_compression_ratio(
    original_size_bytes: int,
    compressed_size_bytes: int,
) -> float:
    """
    Menghitung compression ratio.

    Formula:
        compression_ratio = ukuran_asli / ukuran_kompresi

    Contoh:
        2.77 berarti file asli 2.77 kali lebih besar dari file kompresi.
    """

    if original_size_bytes <= 0:
        raise ValueError("Ukuran file asli harus lebih dari 0.")

    if compressed_size_bytes <= 0:
        raise ValueError("Ukuran file kompresi harus lebih dari 0.")

    return float(original_size_bytes / compressed_size_bytes)


def calculate_saving_percentage(
    original_size_bytes: int,
    compressed_size_bytes: int,
) -> float:
    """
    Menghitung persentase penghematan ukuran file.

    Formula:
        saving = (1 - ukuran_kompresi / ukuran_asli) * 100

    Contoh:
        63.95 berarti ukuran file berhasil dikurangi 63.95%.
    """

    if original_size_bytes <= 0:
        raise ValueError("Ukuran file asli harus lebih dari 0.")

    if compressed_size_bytes <= 0:
        raise ValueError("Ukuran file kompresi harus lebih dari 0.")

    return float((1 - (compressed_size_bytes / original_size_bytes)) * 100)


# ============================================================
# EVALUATION BUILDER
# ============================================================

def build_evaluation_result(
    original_image_path: Path,
    compressed_file_path: Path,
    reconstructed_image_path: Path,
    mode: str,
    total_nodes: int | None = None,
    leaf_nodes: int | None = None,
    tree_depth: int | None = None,
    compression_time_seconds: float | None = None,
    decompression_time_seconds: float | None = None,
) -> EvaluationResult:
    """
    Membuat satu hasil evaluasi lengkap untuk satu gambar dan satu mode.

    Args:
        original_image_path:
            Path gambar asli.

        compressed_file_path:
            Path file hasil kompresi .qtree.

        reconstructed_image_path:
            Path gambar hasil dekompresi.

        mode:
            Mode kompresi, yaitu lossless atau lossy.

        total_nodes, leaf_nodes, tree_depth:
            Statistik Quadtree.

        compression_time_seconds:
            Waktu kompresi.

        decompression_time_seconds:
            Waktu dekompresi.

    Returns:
        EvaluationResult:
            Satu baris hasil evaluasi.
    """

    validate_file_exists(original_image_path)
    validate_file_exists(compressed_file_path)
    validate_file_exists(reconstructed_image_path)

    original_array = load_image_as_array(original_image_path)
    height, width, _ = original_array.shape

    original_size = get_file_size_bytes(original_image_path)
    compressed_size = get_file_size_bytes(compressed_file_path)
    reconstructed_size = get_file_size_bytes(reconstructed_image_path)

    compression_ratio = calculate_compression_ratio(
        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,
    )

    saving_percentage = calculate_saving_percentage(
        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,
    )

    quality_metrics = calculate_quality_metrics(
        original_image_path=original_image_path,
        reconstructed_image_path=reconstructed_image_path,
    )

    return EvaluationResult(
        image_name=original_image_path.name,
        mode=mode,

        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,
        reconstructed_size_bytes=reconstructed_size,

        compression_ratio=compression_ratio,
        saving_percentage=saving_percentage,

        mse=quality_metrics.mse,
        rmse=quality_metrics.rmse,
        mae=quality_metrics.mae,
        psnr=quality_metrics.psnr,
        is_identical=quality_metrics.is_identical,

        width=width,
        height=height,

        total_nodes=total_nodes,
        leaf_nodes=leaf_nodes,
        tree_depth=tree_depth,

        compression_time_seconds=compression_time_seconds,
        decompression_time_seconds=decompression_time_seconds,
    )


# ============================================================
# CSV HELPERS
# ============================================================

def evaluation_results_to_dataframe(
    evaluation_results: list[EvaluationResult],
) -> pd.DataFrame:
    """
    Mengubah list EvaluationResult menjadi pandas DataFrame.

    DataFrame ini nanti bisa:
    - ditampilkan,
    - dianalisis,
    - disimpan ke CSV.
    """

    rows = [asdict(result) for result in evaluation_results]
    dataframe = pd.DataFrame(rows)

    return dataframe


def save_evaluation_results_to_csv(
    evaluation_results: list[EvaluationResult],
    output_csv_path: Path = RESULTS_CSV_PATH,
) -> None:
    """
    Menyimpan hasil evaluasi ke file CSV.

    Args:
        evaluation_results:
            List hasil evaluasi.

        output_csv_path:
            Path file CSV tujuan.
    """

    if not evaluation_results:
        raise ValueError("Tidak ada hasil evaluasi untuk disimpan.")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = evaluation_results_to_dataframe(evaluation_results)
    dataframe.to_csv(output_csv_path, index=False)


def summarize_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat ringkasan rata-rata hasil pengujian berdasarkan mode.

    Ringkasan ini berguna untuk laporan, misalnya:
    - rata-rata compression ratio lossless,
    - rata-rata compression ratio lossy,
    - rata-rata MSE,
    - rata-rata PSNR.
    """

    if dataframe.empty:
        raise ValueError("DataFrame kosong, tidak bisa dibuat ringkasan.")

    summary = dataframe.groupby("mode", as_index=False).agg(
        image_count=("image_name", "count"),
        avg_original_size_bytes=("original_size_bytes", "mean"),
        avg_compressed_size_bytes=("compressed_size_bytes", "mean"),
        avg_compression_ratio=("compression_ratio", "mean"),
        avg_saving_percentage=("saving_percentage", "mean"),
        avg_mse=("mse", "mean"),
        avg_rmse=("rmse", "mean"),
        avg_mae=("mae", "mean"),
        avg_psnr=("psnr", "mean"),
        avg_compression_time_seconds=("compression_time_seconds", "mean"),
        avg_decompression_time_seconds=("decompression_time_seconds", "mean"),
    )

    return summary