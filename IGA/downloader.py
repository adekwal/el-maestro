import os
import re
import requests
import gdown
from tqdm import tqdm


def download_file(url, dest_folder="data", overwrite=False):
    os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(dest_folder, filename)

    if os.path.exists(filepath) and not overwrite:
        print(f"1st file already exists: {filepath}")
        return filepath

    print(f"Downloading 1st file...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filepath, 'wb') as f, tqdm(
            desc=filename, total=total_size, unit='B', unit_scale=True, ncols=80
    ) as pbar:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"1st file saved in: {filepath}")
    return filepath


def download_from_drive(drive_url, dest_folder="data", filename="file.h5", overwrite=False):
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, filename)

    # Check if file exists
    if os.path.exists(dest_path) and not overwrite:
        print(f"2nd file already exists: {dest_path}")
        return dest_path

    # Address exchange
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_url)
    if not match:
        raise ValueError("Google Drive link is not correct")
    file_id = match.group(1)

    # Build URL address
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"Downloading 2nd file...")
    gdown.download(url, dest_path, quiet=False)

    # Verification
    if os.path.exists(dest_path):
        print(f"2nd file saved in: {dest_path}")
        return dest_path
    else:
        raise FileNotFoundError(f"Download error: {url}")
