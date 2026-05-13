"""
clean_outputs.py

Script untuk membersihkan semua hasil output program secara aman.

Yang dibersihkan:
- output/compressed_lossless
- output/compressed_lossy
- output/reconstructed_lossless
- output/reconstructed_lossy
- results/tables
- results/charts

Yang TIDAK disentuh:
- data/raw_bmp
- src
- report
- README.md
- requirements.txt
"""

from config import (
    COMPRESSED_LOSSLESS_DIR,
    COMPRESSED_LOSSY_DIR,
    RECONSTRUCTED_LOSSLESS_DIR,
    RECONSTRUCTED_LOSSY_DIR,
    RESULTS_TABLES_DIR,
    RESULTS_CHARTS_DIR,
    ensure_output_directories,
)

from utils import clear_directory


def main() -> None:
    """
    Membersihkan folder output dan result secara aman.
    """
    ensure_output_directories()

    directories_to_clear = [
        COMPRESSED_LOSSLESS_DIR,
        COMPRESSED_LOSSY_DIR,
        RECONSTRUCTED_LOSSLESS_DIR,
        RECONSTRUCTED_LOSSY_DIR,
        RESULTS_TABLES_DIR,
        RESULTS_CHARTS_DIR,
    ]

    print("Folder yang akan dibersihkan:")
    for directory in directories_to_clear:
        print(f"- {directory}")

    confirmation = input("\nYakin ingin menghapus semua isi folder di atas? ketik YES: ")

    if confirmation != "YES":
        print("Dibatalkan. Tidak ada file yang dihapus.")
        return

    for directory in directories_to_clear:
        clear_directory(directory)

    print("\nSelesai. Semua output dan hasil pengujian berhasil dibersihkan.")


if __name__ == "__main__":
    main()