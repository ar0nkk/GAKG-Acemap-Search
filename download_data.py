import os
import requests
from tqdm import tqdm
import urllib.request
from config import DATA_DIR
import glob

# Configuration
BASE_URL = "https://huggingface.co/datasets/aronkk/gakg/resolve/main"

def download_file(url, dest_path):
    print(f"Attempting to download to {dest_path}...")
    
    # Retry strategies: Default, Mirror, No Verify, Mirror+No Verify
    urls_to_try = [
        (url, True, "Default Source"),
        (url.replace("huggingface.co", "hf-mirror.com"), True, "HF Mirror"),
        (url, False, "SSL Verify Disabled"),
        (url.replace("huggingface.co", "hf-mirror.com"), False, "HF Mirror (No SSL Verify)")
    ]

    proxies = urllib.request.getproxies()
    if proxies and 'https' in proxies and proxies['https'].startswith('https://'):
         proxies['https'] = proxies['https'].replace('https://', 'http://')

    success = False
    for attempt_url, verify_ssl, desc in urls_to_try:
        print(f"\nTrying: {desc} -> {attempt_url}")
        try:
            response = requests.get(attempt_url, stream=True, timeout=30, proxies=proxies, verify=verify_ssl)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 64
            
            with open(dest_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)
            ) as bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))
            print(f"Successfully downloaded {dest_path}")
            success = True
            break
        except Exception as e:
            print(f"  Failed ({desc}): {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
    
    if not success:
        print(f"All download attempts failed. Please manually download {url}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    # Auto-identify: Check if ANY parquet file exists locally
    existing_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    if existing_files:
        print(f"Found existing data files in {DATA_DIR}:")
        for f in existing_files:
            print(f"  - {os.path.basename(f)}")
        print("Skipping download.")
        return

    # If empty, download the default dataset file
    print("No parquet files found. Starting download...")
    
    # We download the 'cleaned' version from the repo but save it as 'gakg.parquet'
    # so that the preprocess script can recognize it as input data to apply further local filters.
    local_filename = "gakg.parquet"
    remote_filename = "gakg_cleaned.parquet"
    
    dest_path = os.path.join(DATA_DIR, local_filename)
    url = f"{BASE_URL}/{remote_filename}"
    download_file(url, dest_path)

if __name__ == "__main__":
    main()
