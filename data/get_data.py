"""Download the input CSV dataset"""
import requests
from tqdm import tqdm
import zipfile
import os

def download_csv(url, output_path):
    # Stream the content to handle large files
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the request was successful

    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))

    # Open the output file in write-binary mode and download the file with progress bar
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    print(f"File downloaded successfully and saved to {output_path}")

def unzip_file(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"File unzipped successfully and extracted to {extract_to}")

# URL of the CSV file
url = "https://hub.dkfz.de/s/a7jJkMBYqEeTpGj/download/GenomeCRISPR_full.csv.zip"

# Output path where the zipped file will be saved
zip_output_path = "GenomeCRISPR.zip"
csv_output_path = "GenomeCRISPR.csv"

# Download the zipped file
download_csv(url, zip_output_path)

# Unzip the downloaded file
unzip_file(zip_output_path)

# Optional: Remove the zip file after extraction
os.remove(zip_output_path)
print(f"Zipped file {zip_output_path} removed.")