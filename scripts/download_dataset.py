"""Download tgbl-flight dataset with progress tracking."""
import os
import requests
from tqdm import tqdm

url = "https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbl-flight-v2.zip"
dest_dir = os.path.join("datasets", "tgbl-flight")
os.makedirs(dest_dir, exist_ok=True)
dest_file = os.path.join(dest_dir, "tgbl-flight-v2.zip")

if os.path.exists(dest_file):
    print(f"File already exists: {dest_file}")
else:
    print(f"Downloading tgbl-flight-v2.zip (1.28 GB)...")
    r = requests.get(url, stream=True, timeout=60)
    total = int(r.headers.get("Content-Length", 0))
    with open(dest_file, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="tgbl-flight") as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Downloaded to {dest_file}")

# Extract
import zipfile
extract_dir = dest_dir
marker = os.path.join(extract_dir, "_extracted")
if not os.path.exists(marker):
    print("Extracting...")
    with zipfile.ZipFile(dest_file, "r") as z:
        z.extractall(extract_dir)
    open(marker, "w").close()
    print("Extraction complete.")
else:
    print("Already extracted.")

# List extracted files
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        fp = os.path.join(root, f)
        size_mb = os.path.getsize(fp) / 1e6
        print(f"  {fp} ({size_mb:.1f} MB)")
