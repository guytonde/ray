import os
import zipfile
import urllib.request


DOWNLOADS = [
    (
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "DIV2K_train_HR.zip",
        "DIV2K/DIV2K_train_HR"
    ),
    (
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "DIV2K_valid_HR.zip",
        "DIV2K/DIV2K_valid_HR"
    ),
]

def is_valid_zip(path):
    try:
        with zipfile.ZipFile(path, "r"):
            return True
    except zipfile.BadZipFile:
        return False


def download(url, dest):
    print(f"\nDownloading {os.path.basename(dest)} ...")
    if os.path.exists(dest):
        os.remove(dest)


    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 8192

            with open(dest, "wb") as file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    file.write(buffer)
                    downloaded += len(buffer)

                    if total > 0:
                        percent = downloaded * 100 / total
                        print(f"\r  Progress: {percent:5.1f}%", end="")

        print("\n  Download complete.")

    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

    if not is_valid_zip(dest):
        print("ERROR: Downloaded file is not a valid zip.")
        if os.path.exists(dest):
            os.remove(dest)
        return False

    return True

def extract(zip_path, extract_to):
    print(f"\nExtracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"  Extracted to {extract_to}/")


if __name__ == "__main__":
    os.makedirs("DIV2K", exist_ok=True)
    for url, zip_name, out_folder in DOWNLOADS:
        if os.path.isdir(out_folder):
            count = len([f for f in os.listdir(out_folder) if f.endswith(".png")])
            print(f"\nAlready exists, skipping: {out_folder}/ ({count} images)")
            continue

        needs_download = not os.path.exists(zip_name) or not is_valid_zip(zip_name)
        if needs_download:
            success = download(url, zip_name)
            if not success:
                continue
        else:
            print(f"\nValid zip already present, skipping download: {zip_name}")

        extract(zip_name, "DIV2K")

        os.remove(zip_name)
        print(f"  Deleted {zip_name}")

    print("\nDone! Folder structure:")
    for _, _, folder in DOWNLOADS:
        if os.path.isdir(folder):
            count = len([f for f in os.listdir(folder) if f.endswith(".png")])
            print(f"  {folder}/  ({count} images)")



