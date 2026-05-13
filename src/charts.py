"""
charts.py

File ini berisi fungsi untuk membuat grafik hasil pengujian
Quadtree BMP Compression.

Grafik yang dibuat:
1. Perbandingan ukuran file asli dan hasil kompresi.
2. Compression ratio.
3. Saving percentage.
4. PSNR.
5. Waktu kompresi dan dekompresi.

Catatan:
- Grafik dibuat dari DataFrame hasil evaluasi.
- File grafik disimpan ke folder results/charts.
- Tidak memakai seaborn, cukup matplotlib agar dependency tetap ringan.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Backend Agg dipakai agar matplotlib bisa menyimpan gambar
# tanpa perlu membuka window GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

DPI: int = 150
FIGURE_WIDTH: int = 12
FIGURE_HEIGHT: int = 6


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_chart_directory(output_directory: Path) -> None:
    """
    Membuat folder charts jika belum ada.
    """
    output_directory.mkdir(parents=True, exist_ok=True)


def validate_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Memastikan DataFrame tidak kosong dan memiliki kolom penting.
    """
    if dataframe.empty:
        raise ValueError("DataFrame kosong, tidak bisa membuat grafik.")

    required_columns = {
        "image_name",
        "mode",
        "original_size_bytes",
        "compressed_size_bytes",
        "compression_ratio",
        "saving_percentage",
        "psnr",
        "compression_time_seconds",
        "decompression_time_seconds",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Kolom DataFrame tidak lengkap untuk membuat grafik: "
            + ", ".join(sorted(missing_columns))
        )


def bytes_to_kilobytes(size_bytes: pd.Series) -> pd.Series:
    """
    Mengubah ukuran bytes menjadi KB.
    """
    return size_bytes / 1024


def get_image_labels(dataframe: pd.DataFrame) -> list[str]:
    """
    Mengambil label nama gambar tanpa ekstensi agar grafik lebih rapi.
    """
    image_names = dataframe["image_name"].drop_duplicates().tolist()

    return [Path(image_name).stem for image_name in image_names]


def prepare_pivot(
    dataframe: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """
    Membuat pivot table dengan index image_name dan kolom mode.

    Contoh hasil:
        image_name | lossless | lossy
        img_01    | 2.0      | 100.0
    """
    return dataframe.pivot_table(
        index="image_name",
        columns="mode",
        values=value_column,
        aggfunc="mean",
    )


def save_current_figure(output_path: Path) -> None:
    """
    Menyimpan figure matplotlib aktif ke file.
    """
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


# ============================================================
# CHART FUNCTIONS
# ============================================================

def create_file_size_comparison_chart(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """
    Membuat grafik perbandingan ukuran file asli dan hasil kompresi.

    Ukuran file ditampilkan dalam KB agar mudah dibaca.
    """
    output_path = output_directory / "file_size_comparison.png"

    chart_data = dataframe.copy()
    chart_data["original_size_kb"] = bytes_to_kilobytes(
        chart_data["original_size_bytes"]
    )
    chart_data["compressed_size_kb"] = bytes_to_kilobytes(
        chart_data["compressed_size_bytes"]
    )

    original_size = (
        chart_data.groupby("image_name")["original_size_kb"]
        .first()
        .sort_index()
    )

    compressed_pivot = prepare_pivot(
        dataframe=chart_data,
        value_column="compressed_size_kb",
    ).sort_index()

    labels = [Path(name).stem for name in original_size.index]
    x_positions = np.arange(len(labels))
    bar_width = 0.25

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    plt.bar(
        x_positions - bar_width,
        original_size.values,
        width=bar_width,
        label="Original BMP",
    )

    offset = 0

    if "lossless" in compressed_pivot.columns:
        plt.bar(
            x_positions + offset,
            compressed_pivot["lossless"].values,
            width=bar_width,
            label="Lossless .qtree",
        )
        offset += bar_width

    if "lossy" in compressed_pivot.columns:
        plt.bar(
            x_positions + offset,
            compressed_pivot["lossy"].values,
            width=bar_width,
            label="Lossy .qtree",
        )

    plt.title("Perbandingan Ukuran File")
    plt.xlabel("Gambar")
    plt.ylabel("Ukuran File (KB)")
    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)

    return output_path


def create_compression_ratio_chart(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """
    Membuat grafik compression ratio untuk lossless dan lossy.
    """
    output_path = output_directory / "compression_ratio.png"

    pivot = prepare_pivot(
        dataframe=dataframe,
        value_column="compression_ratio",
    ).sort_index()

    labels = [Path(name).stem for name in pivot.index]
    x_positions = np.arange(len(labels))
    bar_width = 0.35

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    if "lossless" in pivot.columns:
        plt.bar(
            x_positions - bar_width / 2,
            pivot["lossless"].values,
            width=bar_width,
            label="Lossless",
        )

    if "lossy" in pivot.columns:
        plt.bar(
            x_positions + bar_width / 2,
            pivot["lossy"].values,
            width=bar_width,
            label="Lossy",
        )

    plt.title("Compression Ratio")
    plt.xlabel("Gambar")
    plt.ylabel("Compression Ratio (x)")
    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)

    return output_path


def create_saving_percentage_chart(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """
    Membuat grafik persentase penghematan ukuran file.
    """
    output_path = output_directory / "saving_percentage.png"

    pivot = prepare_pivot(
        dataframe=dataframe,
        value_column="saving_percentage",
    ).sort_index()

    labels = [Path(name).stem for name in pivot.index]
    x_positions = np.arange(len(labels))
    bar_width = 0.35

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    if "lossless" in pivot.columns:
        plt.bar(
            x_positions - bar_width / 2,
            pivot["lossless"].values,
            width=bar_width,
            label="Lossless",
        )

    if "lossy" in pivot.columns:
        plt.bar(
            x_positions + bar_width / 2,
            pivot["lossy"].values,
            width=bar_width,
            label="Lossy",
        )

    plt.title("Saving Percentage")
    plt.xlabel("Gambar")
    plt.ylabel("Penghematan Ukuran (%)")
    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)

    return output_path


def create_psnr_chart(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """
    Membuat grafik PSNR.

    Catatan:
    - Lossless biasanya PSNR infinity karena MSE = 0.
    - Nilai infinity tidak bisa diplot dengan baik.
    - Karena itu, nilai infinity diganti menjadi NaN agar tidak merusak grafik.
    """
    output_path = output_directory / "psnr_comparison.png"

    chart_data = dataframe.copy()
    chart_data["psnr"] = chart_data["psnr"].replace([np.inf, -np.inf], np.nan)

    pivot = prepare_pivot(
        dataframe=chart_data,
        value_column="psnr",
    ).sort_index()

    labels = [Path(name).stem for name in pivot.index]
    x_positions = np.arange(len(labels))
    bar_width = 0.35

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    if "lossless" in pivot.columns:
        plt.bar(
            x_positions - bar_width / 2,
            pivot["lossless"].values,
            width=bar_width,
            label="Lossless",
        )

    if "lossy" in pivot.columns:
        plt.bar(
            x_positions + bar_width / 2,
            pivot["lossy"].values,
            width=bar_width,
            label="Lossy",
        )

    plt.title("PSNR Hasil Rekonstruksi")
    plt.xlabel("Gambar")
    plt.ylabel("PSNR (dB)")
    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)

    return output_path


def create_processing_time_chart(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> Path:
    """
    Membuat grafik waktu kompresi dan dekompresi.

    Waktu ditampilkan dalam detik.
    """
    output_path = output_directory / "processing_time.png"

    chart_data = dataframe.copy()
    chart_data["total_time_seconds"] = (
        chart_data["compression_time_seconds"].fillna(0)
        + chart_data["decompression_time_seconds"].fillna(0)
    )

    pivot = prepare_pivot(
        dataframe=chart_data,
        value_column="total_time_seconds",
    ).sort_index()

    labels = [Path(name).stem for name in pivot.index]
    x_positions = np.arange(len(labels))
    bar_width = 0.35

    plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    if "lossless" in pivot.columns:
        plt.bar(
            x_positions - bar_width / 2,
            pivot["lossless"].values,
            width=bar_width,
            label="Lossless",
        )

    if "lossy" in pivot.columns:
        plt.bar(
            x_positions + bar_width / 2,
            pivot["lossy"].values,
            width=bar_width,
            label="Lossy",
        )

    plt.title("Waktu Proses Kompresi + Dekompresi")
    plt.xlabel("Gambar")
    plt.ylabel("Waktu (detik)")
    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(output_path)

    return output_path


# ============================================================
# MAIN CHART GENERATOR
# ============================================================

def generate_all_charts(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """
    Membuat semua grafik hasil pengujian.

    Args:
        dataframe:
            DataFrame hasil evaluasi.

        output_directory:
            Folder tujuan penyimpanan grafik.

    Returns:
        list[Path]:
            Daftar path file grafik yang berhasil dibuat.
    """
    validate_dataframe(dataframe)
    ensure_chart_directory(output_directory)

    chart_paths = [
        create_file_size_comparison_chart(dataframe, output_directory),
        create_compression_ratio_chart(dataframe, output_directory),
        create_saving_percentage_chart(dataframe, output_directory),
        create_psnr_chart(dataframe, output_directory),
        create_processing_time_chart(dataframe, output_directory),
    ]

    return chart_paths