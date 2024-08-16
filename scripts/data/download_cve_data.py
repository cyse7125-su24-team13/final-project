import requests
import zipfile
import os

# URL of the CVE data zip file
url = "https://github.com/CVEProject/cvelistV5/archive/refs/heads/main.zip"
zip_file_path = "cvelist.zip"
extract_dir = "cvelist/"

# Function to download the file
def download_cve_data(url, save_path):
    print(f"Downloading CVE data from {url}...")
    response = requests.get(url)
    with open(save_path, "wb") as file:
        file.write(response.content)
    print(f"Downloaded CVE data and saved to {save_path}.")

# Function to extract the downloaded zip file
def extract_zip_file(zip_file_path, extract_to):
    print(f"Extracting {zip_file_path} to {extract_to}...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction completed.")

# Function to clean up the zip file after extraction
def clean_up(zip_file_path):
    os.remove(zip_file_path)
    print(f"Removed the zip file: {zip_file_path}")

if __name__ == "__main__":
    # Step 1: Download the CVE data zip file
    download_cve_data(url, zip_file_path)

    # Step 2: Extract the downloaded zip file
    extract_zip_file(zip_file_path, extract_dir)

    # Step 3: Clean up the downloaded zip file
    clean_up(zip_file_path)

    print("CVE data download and extraction process is complete.")
