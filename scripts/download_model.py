import os
import sys
import time
import argparse
import urllib.request
import urllib.error

DEFAULT_URL = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DEFAULT_FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DEST = os.path.join(BASE_DIR, "models", "1b", DEFAULT_FILENAME)

def verify_gguf(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "rb") as f:
            return f.read(4) == b"GGUF"
    except Exception:
        return False

def format_bytes(size):
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def download(url=DEFAULT_URL, destination=DEFAULT_DEST, force=False):
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if os.path.exists(destination) and not force:
        if verify_gguf(destination):
            print(f"Model already verified at: {destination}")
            return True

    temp_path = destination + ".part"
    headers = {"User-Agent": "Mozilla/5.0"}
    existing_bytes = 0

    if os.path.exists(temp_path):
        existing_bytes = os.path.getsize(temp_path)
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)

    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            existing_bytes = 0
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            response = urllib.request.urlopen(req, timeout=30)
        else:
            print(f"Download failed: HTTP {e.code} {e.reason}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

    content_len = response.headers.get("Content-Length")
    is_resume = response.status == 206
    total_bytes = existing_bytes + int(content_len) if content_len and is_resume else (int(content_len) if content_len else 0)

    print(f"Downloading from Hugging Face: {os.path.basename(destination)}")
    if total_bytes > 0:
        print(f"Target size: {format_bytes(total_bytes)}")

    mode = "ab" if is_resume else "wb"
    downloaded = existing_bytes
    start_time = time.time()
    last_update = 0

    try:
        with open(temp_path, mode) as out:
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_update >= 0.2:
                    last_update = now
                    elapsed = max(0.001, now - start_time)
                    speed = (downloaded - existing_bytes) / elapsed
                    pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                    sys.stdout.write(f"\rProgress: {pct:5.1f}% | {format_bytes(downloaded)}/{format_bytes(total_bytes)} | {format_bytes(speed)}/s")
                    sys.stdout.flush()

        sys.stdout.write("\n")
    except Exception as e:
        print(f"\nDownload interrupted: {e}")
        return False

    if not verify_gguf(temp_path):
        print("Header verification failed. Download may be incomplete.")
        return False

    if os.path.exists(destination):
        os.remove(destination)
    os.rename(temp_path, destination)
    print(f"Saved successfully to: {destination}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Hugging Face GGUF Model Downloader")
    parser.add_argument("--url", default=DEFAULT_URL, help="Hugging Face download URL")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Destination file path")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()

    success = download(args.url, args.dest, args.force)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
