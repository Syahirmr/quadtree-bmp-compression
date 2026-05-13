"""
main.py

File utama untuk menjalankan seluruh pipeline project Quadtree BMP Compression.

Pipeline:
1. Validasi dataset.
2. User memilih gambar yang ingin diproses.
3. User memilih mode kompresi.
4. User memilih apakah output lama dibersihkan.
5. Kompresi gambar menggunakan Quadtree.
6. Dekompresi file .qtree kembali menjadi BMP.
7. Evaluasi hasil kompresi.
8. Menyimpan hasil detail dan ringkasan ke CSV.

Desain:
- Clean          : struktur fungsi dipisah jelas.
- Secure         : input user divalidasi sebelum diproses.
- Maintainable   : setiap fungsi punya tanggung jawab spesifik.
- Performance    : hanya gambar yang dipilih user yang diproses.
"""

from __future__ import annotations

import sys
from pathlib import Path

from config import (
    COMPRESSED_LOSSLESS_DIR,
    COMPRESSED_LOSSY_DIR,
    RAW_BMP_DIR,
    RECONSTRUCTED_LOSSLESS_DIR,
    RECONSTRUCTED_LOSSY_DIR,
    RESULTS_CHARTS_DIR,
    RESULTS_CSV_PATH,
    RESULTS_TABLES_DIR,
    ensure_output_directories,
    print_config_summary,
    validate_dataset,
)

from compressor import compress_lossless, compress_lossy
from decompressor import decompress_lossless, decompress_lossy
from metrics import (
    EvaluationResult,
    build_evaluation_result,
    evaluation_results_to_dataframe,
    save_evaluation_results_to_csv,
    summarize_results,
)
from charts import generate_all_charts
from utils import clear_directory, format_file_size, get_bmp_files


# ============================================================
# CONSTANTS
# ============================================================

SUMMARY_CSV_PATH: Path = RESULTS_TABLES_DIR / "ringkasan_hasil.csv"

VALID_DATA_CHOICES: set[str] = {"1", "2", "3"}
VALID_MODE_CHOICES: set[str] = {"1", "2", "3"}
VALID_YES_NO: set[str] = {"y", "n"}

MODE_MAP: dict[str, str] = {
    "1": "both",
    "2": "lossless",
    "3": "lossy",
}


# ============================================================
# INPUT HELPERS
# ============================================================

def ask_choice(prompt: str, valid_choices: set[str]) -> str:
    """
    Meminta input pilihan dari user sampai input valid.

    Args:
        prompt:
            Teks pertanyaan yang ditampilkan ke terminal.

        valid_choices:
            Set pilihan yang dianggap valid.

    Returns:
        str:
            Pilihan user yang sudah valid.
    """
    while True:
        choice = input(prompt).strip()

        if choice in valid_choices:
            return choice

        print(
            f"Input tidak valid. Pilihan yang tersedia: "
            f"{', '.join(sorted(valid_choices))}"
        )


def ask_yes_no(prompt: str) -> bool:
    """
    Meminta input y/n dari user.

    Returns:
        bool:
            True jika user memilih y.
            False jika user memilih n.
    """
    choice = ask_choice(prompt=prompt, valid_choices=VALID_YES_NO)
    return choice == "y"


def ask_positive_integer(prompt: str, max_value: int | None = None) -> int:
    """
    Meminta input angka positif dari user.

    Args:
        prompt:
            Teks pertanyaan.

        max_value:
            Batas maksimal angka yang diperbolehkan.

    Returns:
        int:
            Angka valid dari user.
    """
    while True:
        raw_value = input(prompt).strip()

        if not raw_value.isdigit():
            print("Input harus berupa angka positif.")
            continue

        value = int(raw_value)

        if value <= 0:
            print("Angka harus lebih dari 0.")
            continue

        if max_value is not None and value > max_value:
            print(f"Angka tidak boleh lebih dari {max_value}.")
            continue

        return value


# ============================================================
# IMAGE SELECTION HELPERS
# ============================================================

def print_available_images(image_paths: list[Path]) -> None:
    """
    Menampilkan daftar gambar yang tersedia dengan nomor urut.

    Nomor urut ini dipakai untuk memilih beberapa gambar tertentu
    atau memilih gambar berdasarkan rentang.
    """
    print("\n=== Daftar Gambar Tersedia ===")

    for index, image_path in enumerate(image_paths, start=1):
        print(f"{index:02d}. {image_path.name}")


def get_image_by_number(image_paths: list[Path], image_number: int) -> Path:
    """
    Mengambil path gambar berdasarkan nomor urut 1-based.

    Contoh:
        image_number = 1 berarti mengambil gambar pertama.
    """
    if image_number < 1 or image_number > len(image_paths):
        raise ValueError(
            f"Nomor gambar harus di antara 1 sampai {len(image_paths)}."
        )

    return image_paths[image_number - 1]


def select_images_by_numbers(
    all_image_paths: list[Path],
    selected_numbers: list[int],
) -> list[Path]:
    """
    Memilih gambar berdasarkan beberapa nomor.

    Contoh:
        1 5 10
        berarti memilih gambar nomor 1, 5, dan 10.
    """
    selected_paths: list[Path] = []
    seen_numbers: set[int] = set()

    for number in selected_numbers:
        if number in seen_numbers:
            continue

        selected_paths.append(
            get_image_by_number(
                image_paths=all_image_paths,
                image_number=number,
            )
        )
        seen_numbers.add(number)

    return selected_paths


def select_images_by_range(
    all_image_paths: list[Path],
    start_number: int,
    end_number: int,
) -> list[Path]:
    """
    Memilih gambar berdasarkan rentang nomor.

    Contoh:
        start_number = 5
        end_number   = 10

    Maka yang diproses adalah gambar nomor 5 sampai 10.
    """
    total_images = len(all_image_paths)

    if start_number < 1 or end_number < 1:
        raise ValueError("Nomor gambar harus dimulai dari 1.")

    if start_number > total_images or end_number > total_images:
        raise ValueError(
            f"Nomor gambar tidak boleh lebih dari jumlah gambar: {total_images}."
        )

    if start_number > end_number:
        raise ValueError("Nomor awal tidak boleh lebih besar dari nomor akhir.")

    start_index = start_number - 1
    end_index = end_number

    return all_image_paths[start_index:end_index]


def parse_selected_numbers(raw_input: str, max_number: int) -> list[int]:
    """
    Mengubah input user menjadi list nomor gambar.

    Format input yang didukung:
        1 5 10
        1,5,10
        1, 5, 10

    Args:
        raw_input:
            Input mentah dari terminal.

        max_number:
            Nomor gambar maksimal yang tersedia.

    Returns:
        list[int]:
            Daftar nomor gambar valid.
    """
    cleaned_input = raw_input.replace(",", " ")
    tokens = cleaned_input.split()

    if not tokens:
        raise ValueError("Input nomor gambar tidak boleh kosong.")

    selected_numbers: list[int] = []

    for token in tokens:
        if not token.isdigit():
            raise ValueError(
                f"Input tidak valid: {token}. "
                "Gunakan nomor gambar, contoh: 1 5 10"
            )

        number = int(token)

        if number < 1 or number > max_number:
            raise ValueError(
                f"Nomor {number} tidak valid. "
                f"Gunakan angka 1 sampai {max_number}."
            )

        selected_numbers.append(number)

    return selected_numbers


def ask_selected_images(all_image_paths: list[Path]) -> list[Path]:
    """
    Menanyakan gambar mana saja yang ingin diproses.

    Pilihan:
        1. Semua gambar.
        2. Beberapa gambar tertentu berdasarkan nomor.
        3. Rentang gambar tertentu berdasarkan nomor awal dan akhir.
    """
    print_available_images(all_image_paths)

    print("\n=== Pilihan Data Uji ===")
    print("1. Proses semua gambar")
    print("2. Proses beberapa gambar tertentu")
    print("3. Proses rentang gambar tertentu")

    choice = ask_choice(
        prompt="Mau proses gambar apa? Pilih [1/2/3]: ",
        valid_choices=VALID_DATA_CHOICES,
    )

    if choice == "1":
        return all_image_paths

    if choice == "2":
        print("\nContoh input:")
        print("1 5 10")
        print("atau")
        print("1,5,10")

        raw_numbers = input("Masukkan nomor gambar yang mau diproses: ").strip()

        selected_numbers = parse_selected_numbers(
            raw_input=raw_numbers,
            max_number=len(all_image_paths),
        )

        return select_images_by_numbers(
            all_image_paths=all_image_paths,
            selected_numbers=selected_numbers,
        )

    start_number = ask_positive_integer(
        prompt="Dari nomor gambar: ",
        max_value=len(all_image_paths),
    )

    end_number = ask_positive_integer(
        prompt="Sampai nomor gambar: ",
        max_value=len(all_image_paths),
    )

    return select_images_by_range(
        all_image_paths=all_image_paths,
        start_number=start_number,
        end_number=end_number,
    )


def ask_selected_mode() -> str:
    """
    Menanyakan mode kompresi yang ingin dijalankan.

    Returns:
        str:
            both, lossless, atau lossy.
    """
    print("\n=== Pilihan Mode Kompresi ===")
    print("1. Lossless dan Lossy")
    print("2. Lossless saja")
    print("3. Lossy saja")

    choice = ask_choice(
        prompt="Pilih mode [1/2/3]: ",
        valid_choices=VALID_MODE_CHOICES,
    )

    return MODE_MAP[choice]


def ask_interactive_inputs(all_image_paths: list[Path]) -> tuple[list[Path], str, bool]:
    """
    Mengumpulkan semua input interaktif dari user.

    Returns:
        tuple:
            selected_image_paths:
                Daftar gambar yang akan diproses.

            selected_mode:
                Mode kompresi: both, lossless, atau lossy.

            clean_output:
                True jika output lama dibersihkan.
    """
    selected_image_paths = ask_selected_images(all_image_paths)
    selected_mode = ask_selected_mode()

    clean_output = ask_yes_no(
        prompt="\nBersihkan output lama sebelum proses? [y/n]: "
    )

    return selected_image_paths, selected_mode, clean_output


# ============================================================
# PREPARATION HELPERS
# ============================================================

def prepare_workspace(clean_output: bool) -> None:
    """
    Menyiapkan workspace sebelum proses dimulai.

    Args:
        clean_output:
            Jika True, output lama akan dihapus.
            Jika False, output lama tetap dibiarkan.
    """
    ensure_output_directories()

    if not clean_output:
        return

    directories_to_clear = [
        COMPRESSED_LOSSLESS_DIR,
        COMPRESSED_LOSSY_DIR,
        RECONSTRUCTED_LOSSLESS_DIR,
        RECONSTRUCTED_LOSSY_DIR,
        RESULTS_TABLES_DIR,
        RESULTS_CHARTS_DIR,
    ]

    for directory in directories_to_clear:
        clear_directory(directory)


# ============================================================
# PROCESSING HELPERS
# ============================================================

def process_lossless_image(image_path: Path) -> EvaluationResult:
    """
    Memproses satu gambar dengan mode lossless:
    kompresi -> dekompresi -> evaluasi.
    """
    compression_result = compress_lossless(image_path)
    decompression_result = decompress_lossless(compression_result.output_path)

    return build_evaluation_result(
        original_image_path=image_path,
        compressed_file_path=compression_result.output_path,
        reconstructed_image_path=decompression_result.output_path,
        mode="lossless",
        total_nodes=compression_result.total_nodes,
        leaf_nodes=compression_result.leaf_nodes,
        tree_depth=compression_result.tree_depth,
        compression_time_seconds=compression_result.processing_time_seconds,
        decompression_time_seconds=decompression_result.processing_time_seconds,
    )


def process_lossy_image(image_path: Path) -> EvaluationResult:
    """
    Memproses satu gambar dengan mode lossy:
    kompresi -> dekompresi -> evaluasi.
    """
    compression_result = compress_lossy(image_path)
    decompression_result = decompress_lossy(compression_result.output_path)

    return build_evaluation_result(
        original_image_path=image_path,
        compressed_file_path=compression_result.output_path,
        reconstructed_image_path=decompression_result.output_path,
        mode="lossy",
        total_nodes=compression_result.total_nodes,
        leaf_nodes=compression_result.leaf_nodes,
        tree_depth=compression_result.tree_depth,
        compression_time_seconds=compression_result.processing_time_seconds,
        decompression_time_seconds=decompression_result.processing_time_seconds,
    )


def process_single_image(image_path: Path, mode: str) -> list[EvaluationResult]:
    """
    Memproses satu gambar berdasarkan mode yang dipilih.

    Args:
        image_path:
            Path gambar BMP yang akan diproses.

        mode:
            both, lossless, atau lossy.

    Returns:
        list[EvaluationResult]:
            Hasil evaluasi dari mode yang diproses.
    """
    results: list[EvaluationResult] = []

    if mode in {"both", "lossless"}:
        results.append(process_lossless_image(image_path))

    if mode in {"both", "lossy"}:
        results.append(process_lossy_image(image_path))

    return results


def process_images(
    image_paths: list[Path],
    mode: str,
) -> list[EvaluationResult]:
    """
    Memproses semua gambar yang sudah dipilih user.

    Args:
        image_paths:
            Daftar gambar BMP yang diproses.

        mode:
            Mode kompresi.

    Returns:
        list[EvaluationResult]:
            Semua hasil evaluasi.
    """
    all_results: list[EvaluationResult] = []

    for index, image_path in enumerate(image_paths, start=1):
        print("\n" + "=" * 60)
        print(f"Memproses gambar [{index}/{len(image_paths)}]: {image_path.name}")
        print("=" * 60)

        image_results = process_single_image(
            image_path=image_path,
            mode=mode,
        )

        for result in image_results:
            print_result_brief(result)
            print("-" * 40)

        all_results.extend(image_results)

    return all_results


# ============================================================
# PRINT HELPERS
# ============================================================

def format_psnr(psnr_value: float) -> str:
    """
    Memformat nilai PSNR agar mudah dibaca.
    """
    if psnr_value == float("inf"):
        return "infinity"

    return f"{psnr_value:.2f} dB"


def print_selected_run_summary(
    all_image_paths: list[Path],
    selected_image_paths: list[Path],
    selected_mode: str,
    clean_output: bool,
) -> None:
    """
    Menampilkan ringkasan pilihan user sebelum proses dimulai.
    """
    print("\n=== Ringkasan Pilihan ===")
    print(f"Mode yang dipakai              : {selected_mode}")
    print(f"Output lama dibersihkan?       : {clean_output}")
    print(f"Total gambar tersedia          : {len(all_image_paths)}")
    print(f"Total gambar yang akan diproses: {len(selected_image_paths)}")

    print("\nDaftar gambar yang diproses:")
    for image_path in selected_image_paths:
        print(f"- {image_path.name}")


def print_result_brief(result: EvaluationResult) -> None:
    """
    Menampilkan ringkasan singkat hasil evaluasi satu mode.
    """
    print(f"  Mode               : {result.mode}")
    print(f"  Ukuran asli        : {format_file_size(result.original_size_bytes)}")
    print(f"  Ukuran .qtree      : {format_file_size(result.compressed_size_bytes)}")
    print(f"  Compression ratio  : {result.compression_ratio:.2f}x")
    print(f"  Saving percentage  : {result.saving_percentage:.2f}%")
    print(f"  MSE                : {result.mse:.6f}")
    print(f"  PSNR               : {format_psnr(result.psnr)}")
    print(f"  Identik?           : {result.is_identical}")
    print(f"  Total node         : {result.total_nodes}")
    print(f"  Leaf node          : {result.leaf_nodes}")
    print(f"  Tree depth         : {result.tree_depth}")
    print(f"  Waktu kompresi     : {result.compression_time_seconds:.2f} detik")
    print(f"  Waktu dekompresi   : {result.decompression_time_seconds:.2f} detik")


def print_summary(summary_dataframe) -> None:
    """
    Menampilkan ringkasan rata-rata hasil pengujian per mode.
    """
    print("\n=== Ringkasan Rata-Rata per Mode ===")

    for _, row in summary_dataframe.iterrows():
        print(f"\nMode: {row['mode']}")
        print(f"  Jumlah data              : {int(row['image_count'])}")
        print(f"  Avg original size        : {format_file_size(int(row['avg_original_size_bytes']))}")
        print(f"  Avg compressed size      : {format_file_size(int(row['avg_compressed_size_bytes']))}")
        print(f"  Avg compression ratio    : {row['avg_compression_ratio']:.2f}x")
        print(f"  Avg saving percentage    : {row['avg_saving_percentage']:.2f}%")
        print(f"  Avg MSE                  : {row['avg_mse']:.6f}")
        print(f"  Avg RMSE                 : {row['avg_rmse']:.6f}")
        print(f"  Avg MAE                  : {row['avg_mae']:.6f}")

        avg_psnr = row["avg_psnr"]
        print(f"  Avg PSNR                 : {format_psnr(avg_psnr)}")

        print(
            f"  Avg compression time     : "
            f"{row['avg_compression_time_seconds']:.2f} detik"
        )
        print(
            f"  Avg decompression time   : "
            f"{row['avg_decompression_time_seconds']:.2f} detik"
        )


# ============================================================
# SAVE HELPERS
# ============================================================

def save_all_results(evaluation_results: list[EvaluationResult]) -> None:
    """
    Menyimpan hasil evaluasi detail, ringkasan, dan grafik ke file.
    """
    if not evaluation_results:
        raise ValueError("Tidak ada hasil evaluasi untuk disimpan.")

    save_evaluation_results_to_csv(
        evaluation_results=evaluation_results,
        output_csv_path=RESULTS_CSV_PATH,
    )

    dataframe = evaluation_results_to_dataframe(evaluation_results)
    summary_dataframe = summarize_results(dataframe)

    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_dataframe.to_csv(SUMMARY_CSV_PATH, index=False)

    chart_paths = generate_all_charts(
        dataframe=dataframe,
        output_directory=RESULTS_CHARTS_DIR,
    )

    print("\n=== Preview Hasil Detail ===")
    print(dataframe.head())

    print_summary(summary_dataframe)

    print(f"\nCSV detail berhasil disimpan di : {RESULTS_CSV_PATH}")
    print(f"CSV ringkasan berhasil disimpan : {SUMMARY_CSV_PATH}")

    print("\nGrafik berhasil disimpan:")
    for chart_path in chart_paths:
        print(f"- {chart_path}")

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline() -> None:
    """
    Menjalankan seluruh pipeline utama program.
    """
    print("=== Quadtree BMP Compression ===")

    validate_dataset()
    all_image_paths = get_bmp_files(RAW_BMP_DIR)

    selected_image_paths, selected_mode, clean_output = ask_interactive_inputs(
        all_image_paths=all_image_paths,
    )

    print("\n=== Persiapan Project ===")
    prepare_workspace(clean_output=clean_output)
    print_config_summary()

    print_selected_run_summary(
        all_image_paths=all_image_paths,
        selected_image_paths=selected_image_paths,
        selected_mode=selected_mode,
        clean_output=clean_output,
    )

    print("\n=== Mulai Proses ===")
    evaluation_results = process_images(
        image_paths=selected_image_paths,
        mode=selected_mode,
    )

    print("\n=== Menyimpan Hasil ===")
    save_all_results(evaluation_results)

    print("\n=== Selesai ===")
    print("Seluruh proses kompresi, dekompresi, dan evaluasi telah selesai.")


def main() -> None:
    """
    Entry point program.

    Fungsi ini menangani error umum supaya pesan di terminal lebih rapi.
    """
    try:
        run_pipeline()

    except KeyboardInterrupt:
        print("\n\nProses dibatalkan oleh user.")
        sys.exit(1)

    except Exception as error:
        print(f"\nTerjadi error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()