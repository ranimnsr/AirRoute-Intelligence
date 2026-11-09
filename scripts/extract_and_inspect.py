"""Extract the zip and load the dataset via TGB."""
import os
import zipfile

zip_path = os.path.join("datasets", "tgbl-flight", "tgbl-flight-v2.zip")
extract_dir = os.path.join("datasets", "tgbl-flight")

print("Extracting zip file...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)
print("Done extracting.")

print("\nFiles after extraction:")
for root, dirs, files in os.walk(extract_dir):
    level = root.replace(extract_dir, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        fp = os.path.join(root, f)
        size_mb = os.path.getsize(fp) / 1e6
        print(f"{indent}  {f} ({size_mb:.1f} MB)")
