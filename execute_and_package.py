import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import os
import zipfile

print("1. Membaca notebook...")
with open("SpamMailPrediction.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("2. Menjalankan seluruh cell notebook (Execution)...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "."}})

print("3. Menyimpan notebook yang sudah memiliki output hasil RUN...")
with open("SpamMailPrediction.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("4. Mengekspor notebook ke format HTML...")
html_exporter = HTMLExporter()
(body, resources) = html_exporter.from_notebook_node(nb)
with open("SpamMailPrediction.html", "w", encoding="utf-8") as f:
    f.write(body)

print("5. Membuat file ZIP tugas...")
zip_filename = "Tugas_Percobaan_Mendeteksi_Email_SPAM_MuhammadAryaFatthurahman_2023230006.zip"
files_to_zip = [
    "SpamMailPrediction.ipynb",
    "SpamMailPrediction.py",
    "SpamMailPrediction.html",
    "mail_data.csv",
    "README.md"
]

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(file, arcname=file)
            print(f"   + Menambahkan {file} ke dalam zip")

print(f"\nSEMUA PROSES SELESAI DENGAN SUKSES!")
print(f"File zip siap dikumpulkan: {zip_filename} ({os.path.getsize(zip_filename)} bytes)")
