import requests
from tqdm import tqdm
import os

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total,
        unit='B',
        unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

os.makedirs("data", exist_ok=True)

print("Downloading TWiki_Diffsets...")
download_file(
    "https://huggingface.co/datasets/seonghyeonye/TemporalWiki/resolve/main/TWiki_Diffsets.zip",
    "data/TWiki_Diffsets.zip"
)

print("Downloading TWiki_Probes...")
download_file(
    "https://huggingface.co/datasets/seonghyeonye/TemporalWiki/resolve/main/TWiki_Probes.zip",
    "data/TWiki_Probes.zip"
)

print("Done.")