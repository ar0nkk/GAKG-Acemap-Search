import os
import requests
from tqdm import tqdm

# Configuration
BASE_URL = "https://huggingface.co/datasets/aronkk/gakg/resolve/main"
FILES = [
    "gakg_full_chunk_0000.parquet",
    "gakg_full_chunk_0001.parquet"
]
DATA_DIR = "data"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        # 配置你的代理地址
        proxies = {
            'http': 'http://127.0.0.1:7890',  # 修改为你的代理地址和端口
            'https': 'http://127.0.0.1:7890',  # 修改为你的代理地址和端口
        }
        
        response = requests.get(url, stream=True, timeout=30, proxies=proxies)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(dest_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))
        print(f"Successfully downloaded {dest_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    for file_name in FILES:
        dest_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(dest_path):
            print(f"File already exists: {dest_path}, skipping.")
            continue
        
        url = f"{BASE_URL}/{file_name}"
        download_file(url, dest_path)

if __name__ == "__main__":
    main()
